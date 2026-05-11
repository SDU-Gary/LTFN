#include "logger.h"
#include "ltfn.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ltfn {

struct RuntimeDatasets {
    Dataset train;
    Dataset test;
};

RuntimeDatasets load_runtime_datasets(const ExperimentConfig& config) {
    if (config.mode == RunMode::SmokeTest) {
        RuntimeDatasets datasets;
        datasets.train = generate_smoke_dataset(config.max_train_samples == 0 ? 64 : config.max_train_samples, config.seed);
        datasets.test = generate_smoke_dataset(std::max<std::size_t>(config.eval_samples, 16U), config.seed + 1U);
        return datasets;
    }

    RuntimeDatasets datasets;
    datasets.test = load_mnist_images(config.data_dir / "t10k-images-idx3-ubyte", config.eval_samples);
    if (config.mode == RunMode::Train) {
        datasets.train = load_mnist_images(config.data_dir / "train-images-idx3-ubyte", config.max_train_samples);
    }
    return datasets;
}

void save_eval_reconstructions(
    const Logger& logger,
    const Dataset& dataset,
    const EvaluationResult& evaluation,
    std::size_t samples_seen) {
    logger.save_eval_reconstructions(dataset, evaluation, samples_seen);
}

void print_status_line(
    std::size_t epoch,
    std::size_t samples_seen,
    const MetricRow& metric,
    const StepDiagnostics& last_step) {
    std::cout << "[epoch " << epoch
              << "] samples=" << samples_seen
              << " train_window_mse=" << std::fixed << std::setprecision(6) << metric.train_window_mse
              << " eval_mse=" << metric.eval_mse
              << " eval_energy=" << metric.eval_energy
              << " probe_energy=" << metric.probe_energy
              << " img/s=" << metric.images_per_sec
              << " weight_norm=" << metric.avg_weight_norm
              << " grad_norm=" << metric.avg_gradient_norm
              << " last_energy=" << last_step.energy
              << '\n';
}

RelaxationResult run_probe_sample(
    ILTFNModel& model,
    Logger& logger,
    const Eigen::VectorXd& input,
    int steps,
    std::size_t epoch,
    std::size_t samples_seen,
    std::size_t sample_index);

BatchTrainOptions make_batch_train_options(const Logger& logger, int steps) {
    BatchTrainOptions options;
    const bool log_relaxation = logger.is_enabled(LogTag::Energy) ||
        logger.is_enabled(LogTag::LayerErrors) ||
        logger.is_enabled(LogTag::Gradients);
    if (!log_relaxation) {
        return options;
    }

    options.logged_relax_steps.push_back(0);
    for (int relax_step = 1; relax_step <= steps; ++relax_step) {
        if (logger.should_log_step(relax_step)) {
            options.logged_relax_steps.push_back(relax_step);
        }
    }
    return options;
}

double compute_current_learning_rate(
    const ExperimentConfig& config,
    std::size_t samples_seen,
    std::size_t total_target_samples) {
    if (!config.cosine_lr_schedule || total_target_samples == 0) {
        return config.lr_w;
    }

    const double progress = std::clamp(
        static_cast<double>(samples_seen) / static_cast<double>(total_target_samples),
        0.0,
        1.0);
    const double cosine_term = 0.5 * (1.0 + std::cos(progress * 3.14159265358979323846));
    return config.lr_w_final + (config.lr_w - config.lr_w_final) * cosine_term;
}

MetricRow run_eval_pass(
    ILTFNModel& model,
    const Dataset& test_dataset,
    const Dataset& probe_dataset,
    const ExperimentConfig& config,
    Logger& logger,
    std::size_t epoch,
    std::size_t samples_seen,
    double train_window_mse,
    double images_per_sec,
    double elapsed_sec,
    const StepDiagnostics& last_step) {
    EvaluationResult evaluation = evaluate_model(
        model, test_dataset, config.steps, config.eval_samples, config.recon_samples_to_save);
    save_eval_reconstructions(logger, test_dataset, evaluation, samples_seen);
    logger.log_eval_latents(evaluation, epoch, samples_seen);
    logger.log_eval_effective_dimensions(evaluation, epoch, samples_seen);
    logger.log_eval_error_variances(evaluation, epoch, samples_seen);

    const std::size_t probe_index = std::min(config.probe_index, probe_dataset.images.size() - 1);
    RelaxationResult probe = run_probe_sample(
        model, logger, probe_dataset.images[probe_index], config.steps, epoch, samples_seen, probe_index);

    MetricRow metric;
    metric.epoch = epoch;
    metric.samples_seen = samples_seen;
    metric.train_window_mse = train_window_mse;
    metric.eval_mse = evaluation.average_mse;
    metric.eval_energy = evaluation.average_energy;
    metric.probe_mse = probe.mse;
    metric.probe_energy = probe.final_energy;
    metric.images_per_sec = images_per_sec;
    metric.elapsed_sec = elapsed_sec;
    metric.avg_weight_norm = !last_step.weight_norms.empty()
        ? average_norm(last_step.weight_norms)
        : average_weight_norm(model.weights());
    metric.avg_gradient_norm = average_norm(last_step.weight_gradient_norms);

    logger.log_eval_summary(metric);
    logger.log_event(
        "evaluation",
        "{\"samples_seen\":" + std::to_string(samples_seen) +
            ",\"eval_mse\":" + std::to_string(metric.eval_mse) +
            ",\"probe_energy\":" + std::to_string(metric.probe_energy) +
            (!evaluation.effective_dimensions.empty()
                ? ",\"deepest_effective_dim\":" +
                    std::to_string(evaluation.effective_dimensions.back().effective_dim_90)
                : "") +
            (!evaluation.error_variances.empty()
                ? ",\"deepest_error_variance\":" +
                    std::to_string(evaluation.error_variances.back().error_variance)
                : "") +
            "}");
    print_status_line(epoch, samples_seen, metric, last_step);
    if (!evaluation.effective_dimensions.empty() && !evaluation.error_variances.empty()) {
        const EffectiveDimensionRow& deepest_repr = evaluation.effective_dimensions.back();
        const ErrorVarianceRow& deepest_error = evaluation.error_variances.back();
        std::cout << "  deepest_effective_dim=" << deepest_repr.effective_dim_90
                  << "/" << deepest_repr.width
                  << " deepest_error_variance=" << std::fixed << std::setprecision(6)
                  << deepest_error.error_variance << '\n';
    }
    return metric;
}

RelaxationResult run_train_sample(
    ILTFNModel& model,
    Logger& logger,
    const Eigen::VectorXd& input,
    int steps,
    std::size_t epoch,
    std::size_t samples_seen,
    std::size_t sample_index) {
    model.reset_states(input);
    const bool log_relaxation = logger.is_enabled(LogTag::Energy) ||
        logger.is_enabled(LogTag::LayerErrors) ||
        logger.is_enabled(LogTag::Gradients);
    StepDiagnostics diagnostics;
    if (log_relaxation) {
        diagnostics = model.current_diagnostics();
        logger.log_relaxation_step("train", epoch, samples_seen, sample_index, 0, diagnostics);
    }
    for (int relax_step = 1; relax_step <= steps; ++relax_step) {
        const bool should_log = log_relaxation && logger.should_log_step(relax_step);
        const bool need_diagnostics = relax_step == steps || should_log;
        if (need_diagnostics) {
            diagnostics = model.step_current(true);
            if (should_log) {
                logger.log_relaxation_step("train", epoch, samples_seen, sample_index, relax_step, diagnostics);
            }
        } else {
            model.advance_current(true);
        }
    }

    RelaxationResult result;
    result.reconstruction = model.current_reconstruction();
    result.final_energy = diagnostics.energy;
    result.mse = compute_mse(input, result.reconstruction);
    result.final_error_norms = diagnostics.error_norms;
    result.final_weight_gradient_norms = diagnostics.weight_gradient_norms;
    return result;
}

BatchTrainResult run_train_batch(
    ILTFNModel& model,
    Logger& logger,
    const std::vector<const Eigen::VectorXd*>& inputs,
    int steps,
    std::size_t epoch,
    std::size_t samples_seen,
    std::size_t batch_index) {
    BatchTrainResult result = model.train_batch(inputs, steps, make_batch_train_options(logger, steps));
    for (const LoggedRelaxationStep& entry : result.logged_steps) {
        logger.log_relaxation_step("train", epoch, samples_seen, batch_index, entry.relax_step, entry.diagnostics);
    }
    return result;
}

RelaxationResult run_probe_sample(
    ILTFNModel& model,
    Logger& logger,
    const Eigen::VectorXd& input,
    int steps,
    std::size_t epoch,
    std::size_t samples_seen,
    std::size_t sample_index) {
    model.reset_states(input);
    const bool log_relaxation = logger.is_enabled(LogTag::Energy) ||
        logger.is_enabled(LogTag::LayerErrors) ||
        logger.is_enabled(LogTag::Gradients);
    StepDiagnostics diagnostics;
    if (log_relaxation) {
        diagnostics = model.current_diagnostics();
        logger.log_relaxation_step("probe", epoch, samples_seen, sample_index, 0, diagnostics);
    }
    for (int relax_step = 1; relax_step <= steps; ++relax_step) {
        const bool should_log = log_relaxation && logger.should_log_step(relax_step);
        const bool need_diagnostics = relax_step == steps || should_log;
        if (need_diagnostics) {
            diagnostics = model.step_current(false);
            if (should_log) {
                logger.log_relaxation_step("probe", epoch, samples_seen, sample_index, relax_step, diagnostics);
            }
        } else {
            model.advance_current(false);
        }
    }

    RelaxationResult result;
    result.reconstruction = model.current_reconstruction();
    result.final_energy = diagnostics.energy;
    result.mse = compute_mse(input, result.reconstruction);
    result.final_error_norms = diagnostics.error_norms;
    result.final_weight_gradient_norms = diagnostics.weight_gradient_norms;
    return result;
}

int run_train(const ExperimentConfig& config, int argc, char** argv) {
    RuntimeDatasets datasets = load_runtime_datasets(config);
    Logger logger(config, argc, argv);

    LTFNConfig model_config;
    model_config.dims = config.dims;
    model_config.tau_r = config.tau_r;
    model_config.lr_w = config.lr_w;
    model_config.dt_r = config.dt_r;
    model_config.dt_w = config.dt_w;
    model_config.visible_loss = config.visible_loss;
    model_config.use_biases = config.use_biases;
    model_config.visible_unit_precision = config.visible_unit_precision;
    model_config.state_init = config.state_init;
    model_config.error_precision_beta = config.error_precision_beta;
    model_config.error_precision_epsilon = config.error_precision_epsilon;
    model_config.error_precision_min = config.error_precision_min;
    model_config.error_precision_max = config.error_precision_max;
    model_config.transient_gate_tau = config.transient_gate_tau;
    model_config.sequential_inference = config.sequential_inference;
    model_config.layer_adapt_beta = config.layer_adapt_beta;
    model_config.layer_adapt_epsilon = config.layer_adapt_epsilon;
    model_config.decorrelation_lambda = config.decorrelation_lambda;
    std::unique_ptr<ILTFNModel> model = create_model(model_config, config.seed, config.backend);
    model->set_weight_momentum(config.momentum_beta);

    CheckpointState checkpoint_state;
    checkpoint_state.seed = config.seed;
    if (!config.resume_path.empty()) {
        std::string error;
        if (!load_checkpoint(config.resume_path, *model, checkpoint_state, error)) {
            throw std::runtime_error("Failed to load checkpoint: " + error);
        }
        logger.log_event(
            "checkpoint_loaded",
            "{\"path\":\"" + config.resume_path.generic_string() +
                "\",\"samples_seen\":" + std::to_string(checkpoint_state.samples_seen) + "}");
    }

    if (!config.save_initial_checkpoint_path.empty()) {
        std::string error;
        if (!save_checkpoint(config.save_initial_checkpoint_path, *model, checkpoint_state, error)) {
            throw std::runtime_error("Failed to save initial checkpoint: " + error);
        }
        logger.log_event(
            "initial_checkpoint_saved",
            "{\"path\":\"" + config.save_initial_checkpoint_path.generic_string() +
                "\",\"samples_seen\":" + std::to_string(checkpoint_state.samples_seen) + "}");
    }

    std::vector<std::size_t> indices(datasets.train.images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 generator(config.seed);
    const std::size_t total_target_samples = config.max_train_samples != 0
        ? config.max_train_samples
        : config.max_epochs * datasets.train.images.size();

    const auto start_time = std::chrono::steady_clock::now();
    auto window_start = start_time;

    std::size_t samples_seen = checkpoint_state.samples_seen;
    std::size_t last_evaluated_samples = 0;
    std::size_t next_eval_target = config.eval_interval == 0
        ? 0
        : ((samples_seen / config.eval_interval) + 1U) * config.eval_interval;
    std::size_t next_checkpoint_target = config.checkpoint_interval == 0
        ? 0
        : ((samples_seen / config.checkpoint_interval) + 1U) * config.checkpoint_interval;
    double window_mse_total = 0.0;
    std::size_t window_count = 0;
    StepDiagnostics last_step;

    for (std::size_t epoch = checkpoint_state.epochs_completed; epoch < config.max_epochs; ++epoch) {
        if (config.shuffle) {
            std::shuffle(indices.begin(), indices.end(), generator);
        }

        for (std::size_t offset = 0; offset < indices.size();) {
            if (config.max_train_samples != 0 && samples_seen >= config.max_train_samples) {
                break;
            }

            const std::size_t remaining = indices.size() - offset;
            const std::size_t sample_cap_remaining = config.max_train_samples == 0
                ? remaining
                : std::min(remaining, config.max_train_samples - samples_seen);
            if (sample_cap_remaining == 0) {
                break;
            }

            const std::size_t batch_count = std::min(config.batch_size, sample_cap_remaining);
            std::vector<const Eigen::VectorXd*> batch_inputs;
            batch_inputs.reserve(batch_count);
            for (std::size_t batch_offset = 0; batch_offset < batch_count; ++batch_offset) {
                batch_inputs.push_back(&datasets.train.images[indices[offset + batch_offset]]);
            }

            model->set_learning_rate(compute_current_learning_rate(config, samples_seen, total_target_samples));
            const std::size_t batch_samples_seen = samples_seen + batch_count;
            const BatchTrainResult result = run_train_batch(
                *model,
                logger,
                batch_inputs,
                config.steps,
                epoch + 1,
                batch_samples_seen,
                indices[offset]);
            last_step = result.final_diagnostics;
            last_step.mse = result.average_mse;

            samples_seen = batch_samples_seen;
            window_mse_total += result.average_mse * static_cast<double>(batch_count);
            window_count += batch_count;
            offset += batch_count;

            const bool hit_sample_cap = config.max_train_samples != 0 && samples_seen >= config.max_train_samples;
            const bool should_eval =
                (config.eval_interval != 0 && samples_seen >= next_eval_target) || hit_sample_cap;
            if (should_eval) {
                const auto now = std::chrono::steady_clock::now();
                const double elapsed_sec =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                const double window_elapsed =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - window_start).count();
                const double images_per_sec = window_elapsed > 0.0
                    ? static_cast<double>(window_count) / window_elapsed
                    : 0.0;
                const double train_window_mse = window_count > 0
                    ? window_mse_total / static_cast<double>(window_count)
                    : 0.0;

                run_eval_pass(
                    *model,
                    datasets.test,
                    datasets.test,
                    config,
                    logger,
                    epoch + 1,
                    samples_seen,
                    train_window_mse,
                    images_per_sec,
                    elapsed_sec,
                    last_step);

                last_evaluated_samples = samples_seen;
                window_start = now;
                window_mse_total = 0.0;
                window_count = 0;
                while (config.eval_interval != 0 && next_eval_target <= samples_seen) {
                    next_eval_target += config.eval_interval;
                }
            }

            if (config.checkpoint_interval != 0 && samples_seen >= next_checkpoint_target) {
                checkpoint_state.samples_seen = samples_seen;
                checkpoint_state.epochs_completed = epoch;
                std::string error;
                if (!save_checkpoint(config.checkpoint_path, *model, checkpoint_state, error)) {
                    throw std::runtime_error("Failed to save checkpoint: " + error);
                }
                logger.log_event(
                    "checkpoint_saved",
                    "{\"path\":\"" + config.checkpoint_path.generic_string() +
                        "\",\"samples_seen\":" + std::to_string(samples_seen) + "}");
                while (config.checkpoint_interval != 0 && next_checkpoint_target <= samples_seen) {
                    next_checkpoint_target += config.checkpoint_interval;
                }
            }

            if (hit_sample_cap) {
                break;
            }
        }

        checkpoint_state.epochs_completed = epoch + 1;
        if (config.max_train_samples != 0 && samples_seen >= config.max_train_samples) {
            break;
        }
    }

    if (samples_seen > 0 && samples_seen != last_evaluated_samples) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_sec =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
        const double window_elapsed =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - window_start).count();
        const double images_per_sec = window_elapsed > 0.0
            ? static_cast<double>(window_count) / window_elapsed
            : 0.0;
        const double train_window_mse = window_count > 0
            ? window_mse_total / static_cast<double>(window_count)
            : 0.0;

        run_eval_pass(
            *model,
            datasets.test,
            datasets.test,
            config,
            logger,
            checkpoint_state.epochs_completed,
            samples_seen,
            train_window_mse,
            images_per_sec,
            elapsed_sec,
            last_step);
    }

    checkpoint_state.samples_seen = samples_seen;
    std::string error;
    if (!save_checkpoint(config.checkpoint_path, *model, checkpoint_state, error)) {
        throw std::runtime_error("Failed to save final checkpoint: " + error);
    }
    logger.log_event(
        "run_finished",
        "{\"samples_seen\":" + std::to_string(samples_seen) +
            ",\"checkpoint\":\"" + config.checkpoint_path.generic_string() + "\"}");
    return 0;
}

int run_eval_only(const ExperimentConfig& config, int argc, char** argv) {
    if (config.resume_path.empty()) {
        throw std::runtime_error("--resume is required for --mode eval.");
    }

    RuntimeDatasets datasets = load_runtime_datasets(config);
    Logger logger(config, argc, argv);

    LTFNConfig model_config;
    model_config.dims = config.dims;
    model_config.tau_r = config.tau_r;
    model_config.lr_w = config.lr_w;
    model_config.dt_r = config.dt_r;
    model_config.dt_w = config.dt_w;
    model_config.visible_loss = config.visible_loss;
    model_config.use_biases = config.use_biases;
    model_config.visible_unit_precision = config.visible_unit_precision;
    model_config.state_init = config.state_init;
    model_config.error_precision_beta = config.error_precision_beta;
    model_config.error_precision_epsilon = config.error_precision_epsilon;
    model_config.error_precision_min = config.error_precision_min;
    model_config.error_precision_max = config.error_precision_max;
    model_config.transient_gate_tau = config.transient_gate_tau;
    model_config.sequential_inference = config.sequential_inference;
    model_config.layer_adapt_beta = config.layer_adapt_beta;
    model_config.layer_adapt_epsilon = config.layer_adapt_epsilon;
    model_config.decorrelation_lambda = config.decorrelation_lambda;
    std::unique_ptr<ILTFNModel> model = create_model(model_config, config.seed, config.backend);

    CheckpointState checkpoint_state;
    std::string error;
    if (!load_checkpoint(config.resume_path, *model, checkpoint_state, error)) {
        throw std::runtime_error("Failed to load checkpoint: " + error);
    }

    StepDiagnostics diagnostics;

    run_eval_pass(
        *model,
        datasets.test,
        datasets.test,
        config,
        logger,
        checkpoint_state.epochs_completed,
        checkpoint_state.samples_seen,
        0.0,
        0.0,
        0.0,
        diagnostics);

    logger.log_event(
        "eval_finished",
        "{\"checkpoint\":\"" + config.resume_path.generic_string() + "\"}");
    return 0;
}
}  // namespace ltfn

int main(int argc, char** argv) {
    try {
        ltfn::ExperimentConfig config;
        std::string error_message;
        const bool parsed = ltfn::parse_args(argc, argv, config, error_message);
        if (!parsed) {
            if (!error_message.empty()) {
                std::cerr << "Argument error: " << error_message << "\n\n";
            }
            std::cout << ltfn::usage_text(argv[0]);
            return error_message.empty() ? 0 : 1;
        }

        switch (config.mode) {
            case ltfn::RunMode::Train:
            case ltfn::RunMode::SmokeTest:
                return ltfn::run_train(config, argc, argv);
            case ltfn::RunMode::Eval:
                return ltfn::run_eval_only(config, argc, argv);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << '\n';
        return 1;
    }

    std::cerr << "Fatal error: unreachable run mode.\n";
    return 1;
}
