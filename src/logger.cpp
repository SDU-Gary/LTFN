#include "logger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ltfn {

namespace {

std::string current_timestamp_utc() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t time = clock::to_time_t(now);

    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif

    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::string escape_json(const std::string& value) {
    std::ostringstream escaped;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << ch;
                break;
        }
    }
    return escaped.str();
}

std::string format_double(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

std::string dims_to_json(const std::vector<int>& dims) {
    std::ostringstream stream;
    stream << "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            stream << ",";
        }
        stream << dims[i];
    }
    stream << "]";
    return stream.str();
}

std::string path_to_json(const fs::path& path) {
    return "\"" + escape_json(path.generic_string()) + "\"";
}

std::string bool_to_json(bool value) {
    return value ? "true" : "false";
}

std::string trim_copy(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string lowercase_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

void enable_core_tags(std::array<bool, static_cast<std::size_t>(LogTag::Count)>& enabled) {
    enabled[static_cast<std::size_t>(LogTag::Energy)] = true;
    enabled[static_cast<std::size_t>(LogTag::LayerErrors)] = true;
    enabled[static_cast<std::size_t>(LogTag::Gradients)] = true;
    enabled[static_cast<std::size_t>(LogTag::Eval)] = true;
    enabled[static_cast<std::size_t>(LogTag::Latents)] = true;
}

std::array<bool, static_cast<std::size_t>(LogTag::Count)> parse_tags(const std::string& raw_tags) {
    std::array<bool, static_cast<std::size_t>(LogTag::Count)> enabled{};

    std::stringstream stream(raw_tags);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = lowercase_copy(trim_copy(token));
        if (token.empty()) {
            continue;
        }
        if (token == "all") {
            enabled.fill(true);
            continue;
        }
        if (token == "none") {
            enabled.fill(false);
            continue;
        }
        if (token == "core") {
            enable_core_tags(enabled);
            continue;
        }
        if (token == "energy") {
            enabled[static_cast<std::size_t>(LogTag::Energy)] = true;
            continue;
        }
        if (token == "layer-errors" || token == "layer_errors") {
            enabled[static_cast<std::size_t>(LogTag::LayerErrors)] = true;
            continue;
        }
        if (token == "gradients" || token == "gradient-norms" || token == "gradient_norms") {
            enabled[static_cast<std::size_t>(LogTag::Gradients)] = true;
            continue;
        }
        if (token == "eval" || token == "metrics") {
            enabled[static_cast<std::size_t>(LogTag::Eval)] = true;
            continue;
        }
        if (token == "latents" || token == "latent-top" || token == "latent_top" || token == "representations") {
            enabled[static_cast<std::size_t>(LogTag::Latents)] = true;
            continue;
        }
        if (token == "reconstructions" || token == "recons") {
            enabled[static_cast<std::size_t>(LogTag::Reconstructions)] = true;
            continue;
        }
        if (token == "events") {
            enabled[static_cast<std::size_t>(LogTag::Events)] = true;
            continue;
        }
        throw std::invalid_argument("Unknown logger tag: " + token);
    }

    if (raw_tags.empty()) {
        enabled.fill(true);
    }
    return enabled;
}

}  // namespace

Logger::Logger(const ExperimentConfig& config, int argc, char** argv)
    : config_(config),
      root_dir_(config.output_dir),
      reconstructions_dir_(config.output_dir / "reconstructions"),
      checkpoints_dir_(config.output_dir / "checkpoints"),
      step_interval_(std::max<std::size_t>(1U, config.logger_step_interval)),
      enabled_(parse_tags(config.logger_tags)) {
    ensure_directory(root_dir_);
    ensure_directory(reconstructions_dir_);
    ensure_directory(checkpoints_dir_);
    open_streams();
    write_headers();
    write_config_snapshot(argc, argv);
    log_event(
        "run_started",
        "{\"mode\":\"" + mode_to_string(config.mode) +
            "\",\"backend\":\"" + backend_to_string(config.backend) + "\"}");
}

bool Logger::is_enabled(LogTag tag) const noexcept {
    return enabled_[static_cast<std::size_t>(tag)];
}

bool Logger::should_log_step(int relax_step) const noexcept {
    return relax_step == 0 || (relax_step > 0 && static_cast<std::size_t>(relax_step) % step_interval_ == 0);
}

const fs::path& Logger::root_dir() const noexcept {
    return root_dir_;
}

const fs::path& Logger::checkpoints_dir() const noexcept {
    return checkpoints_dir_;
}

const fs::path& Logger::reconstructions_dir() const noexcept {
    return reconstructions_dir_;
}

void Logger::log_event(const std::string& type, const std::string& payload_json) {
    if (!is_enabled(LogTag::Events) || !events_jsonl_) {
        return;
    }
    events_jsonl_
        << "{\"timestamp_utc\":\"" << current_timestamp_utc()
        << "\",\"type\":\"" << escape_json(type)
        << "\",\"payload\":" << payload_json
        << "}\n";
    events_jsonl_.flush();
}

void Logger::log_eval_summary(const MetricRow& row) {
    if (!is_enabled(LogTag::Eval) || !metrics_csv_) {
        return;
    }
    metrics_csv_
        << current_timestamp_utc() << ','
        << row.epoch << ','
        << row.samples_seen << ','
        << format_double(row.train_window_mse) << ','
        << format_double(row.eval_mse) << ','
        << format_double(row.eval_energy) << ','
        << format_double(row.probe_mse) << ','
        << format_double(row.probe_energy) << ','
        << format_double(row.images_per_sec) << ','
        << format_double(row.elapsed_sec) << ','
        << format_double(row.avg_weight_norm) << ','
        << format_double(row.avg_gradient_norm)
        << '\n';
    metrics_csv_.flush();
}

void Logger::log_relaxation_step(
    const std::string& phase,
    std::size_t epoch,
    std::size_t samples_seen,
    std::size_t sample_index,
    int relax_step,
    const StepDiagnostics& diagnostics) {
    if (!should_log_step(relax_step)) {
        return;
    }

    const std::string timestamp = current_timestamp_utc();
    if (is_enabled(LogTag::Energy) && energy_csv_) {
        energy_csv_
            << timestamp << ','
            << phase << ','
            << epoch << ','
            << samples_seen << ','
            << sample_index << ','
            << relax_step << ','
            << format_double(diagnostics.energy) << ','
            << format_double(diagnostics.mse)
            << '\n';
        energy_csv_.flush();
    }

    if (is_enabled(LogTag::LayerErrors) && layer_errors_csv_) {
        for (std::size_t layer = 0; layer < diagnostics.error_norms.size(); ++layer) {
            const double norm = diagnostics.error_norms[layer];
            layer_errors_csv_
                << timestamp << ','
                << phase << ','
                << epoch << ','
                << samples_seen << ','
                << sample_index << ','
                << relax_step << ','
                << layer << ','
                << format_double(norm) << ','
                << format_double(norm * norm)
                << '\n';
        }
        layer_errors_csv_.flush();
    }

    if (is_enabled(LogTag::Gradients) && grad_norms_csv_) {
        const std::size_t count = diagnostics.weight_gradient_norms.size();
        for (std::size_t layer = 0; layer < count; ++layer) {
            const double gradient_norm = diagnostics.weight_gradient_norms[layer];
            const double state_update_norm =
                layer < diagnostics.state_update_norms.size() ? diagnostics.state_update_norms[layer] : 0.0;
            const double update_norm =
                layer < diagnostics.weight_update_norms.size() ? diagnostics.weight_update_norms[layer] : 0.0;
            const double weight_norm =
                layer < diagnostics.weight_norms.size() ? diagnostics.weight_norms[layer] : 0.0;
            grad_norms_csv_
                << timestamp << ','
                << phase << ','
                << epoch << ','
                << samples_seen << ','
                << sample_index << ','
                << relax_step << ','
                << layer << ','
                << format_double(gradient_norm) << ','
                << format_double(state_update_norm) << ','
                << format_double(update_norm) << ','
                << format_double(weight_norm)
                << '\n';
        }
        grad_norms_csv_.flush();
    }
}

void Logger::save_eval_reconstructions(
    const Dataset& dataset,
    const EvaluationResult& evaluation,
    std::size_t samples_seen) const {
    if (!is_enabled(LogTag::Reconstructions)) {
        return;
    }

    const std::size_t count = std::min(evaluation.reconstructions.size(), dataset.images.size());
    for (std::size_t i = 0; i < count; ++i) {
        std::ostringstream file_name;
        file_name << "step_" << std::setw(8) << std::setfill('0') << samples_seen
                  << "_idx_" << std::setw(4) << std::setfill('0') << i
                  << ".pgm";
        write_side_by_side_pgm(
            reconstructions_dir_ / file_name.str(),
            dataset.images[i],
            evaluation.reconstructions[i],
            dataset.rows,
            dataset.cols);
    }
}

void Logger::log_eval_latents(
    const EvaluationResult& evaluation,
    std::size_t epoch,
    std::size_t samples_seen) {
    if (!is_enabled(LogTag::Latents) || !latent_top_csv_) {
        return;
    }

    const std::string timestamp = current_timestamp_utc();
    for (std::size_t i = 0; i < evaluation.top_representations.size(); ++i) {
        const Eigen::VectorXd& latent = evaluation.top_representations[i];
        latent_top_csv_
            << timestamp << ','
            << epoch << ','
            << samples_seen << ','
            << i;
        for (Eigen::Index dim = 0; dim < latent.size(); ++dim) {
            latent_top_csv_ << ',' << format_double(latent(dim));
        }
        latent_top_csv_ << '\n';
    }
    latent_top_csv_.flush();
}

void Logger::log_eval_effective_dimensions(
    const EvaluationResult& evaluation,
    std::size_t epoch,
    std::size_t samples_seen) {
    if (!is_enabled(LogTag::Eval) || !effective_dims_csv_) {
        return;
    }

    const std::string timestamp = current_timestamp_utc();
    for (const EffectiveDimensionRow& row : evaluation.effective_dimensions) {
        effective_dims_csv_
            << timestamp << ','
            << epoch << ','
            << samples_seen << ','
            << row.layer << ','
            << row.width << ','
            << row.samples << ','
            << row.effective_dim_90 << ','
            << format_double(row.effective_dim_ratio) << ','
            << format_double(row.largest_singular) << ','
            << format_double(row.smallest_singular) << ','
            << format_double(row.spectrum_ratio)
            << '\n';
    }
    effective_dims_csv_.flush();
}

void Logger::log_eval_error_variances(
    const EvaluationResult& evaluation,
    std::size_t epoch,
    std::size_t samples_seen) {
    if (!is_enabled(LogTag::Eval) || !error_variances_csv_) {
        return;
    }

    const std::string timestamp = current_timestamp_utc();
    for (const ErrorVarianceRow& row : evaluation.error_variances) {
        error_variances_csv_
            << timestamp << ','
            << epoch << ','
            << samples_seen << ','
            << row.layer << ','
            << row.width << ','
            << row.samples << ','
            << format_double(row.error_mean) << ','
            << format_double(row.error_variance) << ','
            << format_double(row.error_mean_square)
            << '\n';
    }
    error_variances_csv_.flush();
}

void Logger::open_streams() {
    if (is_enabled(LogTag::Eval)) {
        metrics_csv_.open(root_dir_ / "metrics.csv", std::ios::out | std::ios::trunc);
        effective_dims_csv_.open(root_dir_ / "effective_dims.csv", std::ios::out | std::ios::trunc);
        error_variances_csv_.open(root_dir_ / "error_variances.csv", std::ios::out | std::ios::trunc);
    }
    if (is_enabled(LogTag::Events)) {
        events_jsonl_.open(root_dir_ / "events.jsonl", std::ios::out | std::ios::trunc);
    }
    if (is_enabled(LogTag::Energy)) {
        energy_csv_.open(root_dir_ / "energy.csv", std::ios::out | std::ios::trunc);
    }
    if (is_enabled(LogTag::LayerErrors)) {
        layer_errors_csv_.open(root_dir_ / "layer_errors.csv", std::ios::out | std::ios::trunc);
    }
    if (is_enabled(LogTag::Gradients)) {
        grad_norms_csv_.open(root_dir_ / "grad_norms.csv", std::ios::out | std::ios::trunc);
    }
    if (is_enabled(LogTag::Latents)) {
        latent_top_csv_.open(root_dir_ / "latent_top.csv", std::ios::out | std::ios::trunc);
    }

    if ((is_enabled(LogTag::Eval) && !metrics_csv_) ||
        (is_enabled(LogTag::Eval) && !effective_dims_csv_) ||
        (is_enabled(LogTag::Eval) && !error_variances_csv_) ||
        (is_enabled(LogTag::Events) && !events_jsonl_) ||
        (is_enabled(LogTag::Energy) && !energy_csv_) ||
        (is_enabled(LogTag::LayerErrors) && !layer_errors_csv_) ||
        (is_enabled(LogTag::Gradients) && !grad_norms_csv_) ||
        (is_enabled(LogTag::Latents) && !latent_top_csv_)) {
        throw std::runtime_error("Failed to open one or more logger output files.");
    }
}

void Logger::write_config_snapshot(int argc, char** argv) {
    std::ofstream config_file(root_dir_ / "config.json", std::ios::out | std::ios::trunc);
    if (!config_file) {
        throw std::runtime_error("Failed to write logger config snapshot.");
    }

    config_file << "{\n"
                << "  \"timestamp_utc\": \"" << current_timestamp_utc() << "\",\n"
                << "  \"mode\": \"" << mode_to_string(config_.mode) << "\",\n"
                << "  \"backend\": \"" << backend_to_string(config_.backend) << "\",\n"
                << "  \"data_dir\": " << path_to_json(config_.data_dir) << ",\n"
                << "  \"output_dir\": " << path_to_json(config_.output_dir) << ",\n"
                << "  \"checkpoint_path\": " << path_to_json(config_.checkpoint_path) << ",\n"
                << "  \"resume_path\": " << path_to_json(config_.resume_path) << ",\n"
                << "  \"save_initial_checkpoint_path\": " << path_to_json(config_.save_initial_checkpoint_path) << ",\n"
                << "  \"dims\": " << dims_to_json(config_.dims) << ",\n"
                << "  \"tau_r\": " << config_.tau_r << ",\n"
                << "  \"lr_w\": " << config_.lr_w << ",\n"
                << "  \"lr_w_final\": " << config_.lr_w_final << ",\n"
                << "  \"dt_r\": " << config_.dt_r << ",\n"
                << "  \"dt_w\": " << config_.dt_w << ",\n"
                << "  \"visible_loss\": \"" << visible_loss_to_string(config_.visible_loss) << "\",\n"
                << "  \"momentum_beta\": " << config_.momentum_beta << ",\n"
                << "  \"layer_adapt_beta\": " << config_.layer_adapt_beta << ",\n"
                << "  \"layer_adapt_epsilon\": " << config_.layer_adapt_epsilon << ",\n"
                << "  \"decorrelation_lambda\": " << config_.decorrelation_lambda << ",\n"
                << "  \"steps\": " << config_.steps << ",\n"
                << "  \"batch_size\": " << config_.batch_size << ",\n"
                << "  \"max_epochs\": " << config_.max_epochs << ",\n"
                << "  \"max_train_samples\": " << config_.max_train_samples << ",\n"
                << "  \"eval_samples\": " << config_.eval_samples << ",\n"
                << "  \"eval_interval\": " << config_.eval_interval << ",\n"
                << "  \"checkpoint_interval\": " << config_.checkpoint_interval << ",\n"
                << "  \"probe_index\": " << config_.probe_index << ",\n"
                << "  \"recon_samples_to_save\": " << config_.recon_samples_to_save << ",\n"
                << "  \"logger_step_interval\": " << config_.logger_step_interval << ",\n"
                << "  \"logger_tags\": \"" << escape_json(config_.logger_tags) << "\",\n"
                << "  \"cosine_lr_schedule\": " << bool_to_json(config_.cosine_lr_schedule) << ",\n"
                << "  \"shuffle\": " << bool_to_json(config_.shuffle) << ",\n"
                << "  \"seed\": " << config_.seed << ",\n"
                << "  \"command_line\": \"" << escape_json(join_command_line(argc, argv)) << "\"\n"
                << "}\n";
}

void Logger::write_headers() {
    if (metrics_csv_) {
        metrics_csv_
            << "timestamp_utc,epoch,samples_seen,train_window_mse,eval_mse,eval_energy,probe_mse,probe_energy,"
               "images_per_sec,elapsed_sec,avg_weight_norm,avg_gradient_norm\n";
    }
    if (effective_dims_csv_) {
        effective_dims_csv_
            << "timestamp_utc,epoch,samples_seen,layer,width,samples,effective_dim_90,effective_dim_ratio,"
               "largest_singular,smallest_singular,spectrum_ratio\n";
    }
    if (error_variances_csv_) {
        error_variances_csv_
            << "timestamp_utc,epoch,samples_seen,layer,width,samples,error_mean,error_variance,error_mean_square\n";
    }
    if (energy_csv_) {
        energy_csv_ << "timestamp_utc,phase,epoch,samples_seen,sample_index,relax_step,energy,mse\n";
    }
    if (layer_errors_csv_) {
        layer_errors_csv_
            << "timestamp_utc,phase,epoch,samples_seen,sample_index,relax_step,layer,error_norm,error_squared\n";
    }
    if (grad_norms_csv_) {
        grad_norms_csv_
            << "timestamp_utc,phase,epoch,samples_seen,sample_index,relax_step,layer,gradient_norm,"
               "state_update_norm,"
               "weight_update_norm,weight_norm\n";
    }
    if (latent_top_csv_) {
        latent_top_csv_ << "timestamp_utc,epoch,samples_seen,eval_index";
        for (int dim = 0; dim < config_.dims.back(); ++dim) {
            latent_top_csv_ << ",latent_" << dim;
        }
        latent_top_csv_ << '\n';
    }
}

}  // namespace ltfn
