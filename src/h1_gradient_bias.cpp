#include "ltfn.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ltfn {
namespace h1_detail {

struct H1Config {
    fs::path checkpoint_path;
    fs::path data_dir;
    fs::path output_path{"runs/h1_gradient_bias.json"};
    ComputeBackend backend{ComputeBackend::Cuda};
    std::vector<int> dims{784, 512, 256, 128, 64};
    double tau_r{0.1};
    double lr_w{3.2e-5};
    double dt_r{0.1};
    double dt_w{1.0};
    VisibleLoss visible_loss{VisibleLoss::Mse};
    std::size_t batch_size{128};
    std::uint32_t seed{123U};
    std::vector<int> steps_list{200, 500, 1000, 2000, 5000};
};

std::vector<int> parse_int_list(const std::string& value) {
    std::vector<int> values;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    if (values.empty()) {
        throw std::invalid_argument("Expected a non-empty comma-separated integer list.");
    }
    return values;
}

ComputeBackend parse_backend(const std::string& value) {
    if (value == "cpu") {
        return ComputeBackend::Cpu;
    }
    if (value == "cuda") {
        return ComputeBackend::Cuda;
    }
    throw std::invalid_argument("Unknown backend: " + value);
}

VisibleLoss parse_visible_loss(const std::string& value) {
    if (value == "mse") {
        return VisibleLoss::Mse;
    }
    if (value == "bce") {
        return VisibleLoss::Bce;
    }
    throw std::invalid_argument("Unknown visible loss: " + value);
}

std::string h1_usage_text(const char* program_name) {
    std::ostringstream stream;
    stream
        << "Usage: " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --checkpoint PATH              Checkpoint to analyze (required)\n"
        << "  --data-dir PATH                MNIST data directory (required)\n"
        << "  --output PATH                  Output JSON path (default: runs/h1_gradient_bias.json)\n"
        << "  --backend cpu|cuda             Backend for relaxation (default: cuda)\n"
        << "  --dims LIST                    Layer dimensions (default: 784,512,256,128,64)\n"
        << "  --tau-r FLOAT                  State time constant (default: 0.1)\n"
        << "  --lr-w FLOAT                   Checkpoint learning rate for config match (default: 3.2e-5)\n"
        << "  --dt-r FLOAT                   State step size (default: 0.1)\n"
        << "  --dt-w FLOAT                   Weight step size (default: 1.0)\n"
        << "  --visible-loss mse|bce         Visible loss type (default: mse)\n"
        << "  --batch-size INT               Number of fixed samples (default: 128)\n"
        << "  --seed INT                     Sampling seed (default: 123)\n"
        << "  --steps-list LIST              Relax steps to compare (default: 200,500,1000,2000,5000)\n"
        << "  --help                         Show this help message\n";
    return stream.str();
}

bool parse_h1_args(int argc, char** argv, H1Config& config, std::string& error_message) {
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            auto require_value = [&](std::string& value) {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("Missing value for " + key);
                }
                value = argv[++i];
            };

            if (key == "--help" || key == "-h") {
                error_message.clear();
                return false;
            }

            std::string value;
            if (key == "--checkpoint") {
                require_value(value);
                config.checkpoint_path = value;
            } else if (key == "--data-dir") {
                require_value(value);
                config.data_dir = value;
            } else if (key == "--output") {
                require_value(value);
                config.output_path = value;
            } else if (key == "--backend") {
                require_value(value);
                config.backend = parse_backend(value);
            } else if (key == "--dims") {
                require_value(value);
                config.dims = parse_int_list(value);
            } else if (key == "--tau-r") {
                require_value(value);
                config.tau_r = std::stod(value);
            } else if (key == "--lr-w") {
                require_value(value);
                config.lr_w = std::stod(value);
            } else if (key == "--dt-r") {
                require_value(value);
                config.dt_r = std::stod(value);
            } else if (key == "--dt-w") {
                require_value(value);
                config.dt_w = std::stod(value);
            } else if (key == "--visible-loss") {
                require_value(value);
                config.visible_loss = parse_visible_loss(value);
            } else if (key == "--batch-size") {
                require_value(value);
                config.batch_size = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--seed") {
                require_value(value);
                config.seed = static_cast<std::uint32_t>(std::stoul(value));
            } else if (key == "--steps-list") {
                require_value(value);
                config.steps_list = parse_int_list(value);
            } else {
                throw std::invalid_argument("Unknown argument: " + key);
            }
        }

        if (config.checkpoint_path.empty()) {
            throw std::invalid_argument("--checkpoint is required.");
        }
        if (config.data_dir.empty()) {
            throw std::invalid_argument("--data-dir is required.");
        }
        if (config.batch_size == 0) {
            throw std::invalid_argument("--batch-size must be positive.");
        }
        if (config.steps_list.empty()) {
            throw std::invalid_argument("--steps-list must not be empty.");
        }
        for (int step : config.steps_list) {
            if (step <= 0) {
                throw std::invalid_argument("All entries in --steps-list must be positive.");
            }
        }
        return true;
    } catch (const std::exception& ex) {
        error_message = ex.what();
        return false;
    }
}

double matrix_dot(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs) {
    return (lhs.array() * rhs.array()).sum();
}

struct StepComparison {
    int steps{0};
    double final_energy{0.0};
    double final_mse{0.0};
    double overall_cosine{0.0};
    double overall_norm_ratio{0.0};
    std::vector<double> layer_cosines;
    std::vector<double> layer_norm_ratios;
};

std::string format_double(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(8) << value;
    return stream.str();
}

}  // namespace h1_detail

int run_h1_gradient_bias(const h1_detail::H1Config& config) {
    Dataset train = load_mnist_images(config.data_dir / "train-images-idx3-ubyte", 0);
    if (train.images.size() < config.batch_size) {
        throw std::runtime_error("Dataset is smaller than the requested batch size.");
    }

    std::vector<std::size_t> all_indices(train.images.size());
    std::iota(all_indices.begin(), all_indices.end(), 0U);
    std::mt19937 generator(config.seed);
    std::shuffle(all_indices.begin(), all_indices.end(), generator);
    all_indices.resize(config.batch_size);
    std::sort(all_indices.begin(), all_indices.end());

    std::vector<const Eigen::VectorXd*> inputs;
    inputs.reserve(config.batch_size);
    for (std::size_t index : all_indices) {
        inputs.push_back(&train.images[index]);
    }

    std::vector<int> steps_list = config.steps_list;
    std::sort(steps_list.begin(), steps_list.end());
    steps_list.erase(std::unique(steps_list.begin(), steps_list.end()), steps_list.end());
    const int reference_steps = steps_list.back();

    LTFNConfig model_config;
    model_config.dims = config.dims;
    model_config.tau_r = config.tau_r;
    model_config.lr_w = config.lr_w;
    model_config.dt_r = config.dt_r;
    model_config.dt_w = config.dt_w;
    model_config.visible_loss = config.visible_loss;

    std::vector<BatchTrainResult> batch_results;
    batch_results.reserve(steps_list.size());

    for (int steps : steps_list) {
        std::unique_ptr<ILTFNModel> model = create_model(model_config, 42U, config.backend);
        CheckpointState checkpoint_state;
        std::string checkpoint_error;
        if (!load_checkpoint(config.checkpoint_path, *model, checkpoint_state, checkpoint_error)) {
            throw std::runtime_error("Failed to load checkpoint: " + checkpoint_error);
        }

        BatchTrainOptions options;
        options.update_weights = false;
        options.capture_final_gradients = true;
        BatchTrainResult result = model->train_batch(inputs, steps, options);
        if (result.final_weight_gradients.empty()) {
            throw std::runtime_error("Gradient capture failed for steps=" + std::to_string(steps));
        }
        batch_results.push_back(std::move(result));
    }

    const std::vector<Eigen::MatrixXd>& reference_gradients = batch_results.back().final_weight_gradients;
    std::vector<h1_detail::StepComparison> comparisons;
    comparisons.reserve(batch_results.size());

    for (std::size_t idx = 0; idx < batch_results.size(); ++idx) {
        const BatchTrainResult& result = batch_results[idx];
        h1_detail::StepComparison comparison;
        comparison.steps = steps_list[idx];
        comparison.final_energy = result.final_diagnostics.energy;
        comparison.final_mse = result.final_diagnostics.mse;

        double global_dot = 0.0;
        double global_norm = 0.0;
        double reference_norm = 0.0;
        comparison.layer_cosines.reserve(result.final_weight_gradients.size());
        comparison.layer_norm_ratios.reserve(result.final_weight_gradients.size());

        for (std::size_t layer = 0; layer < result.final_weight_gradients.size(); ++layer) {
            const Eigen::MatrixXd& gradient = result.final_weight_gradients[layer];
            const Eigen::MatrixXd& reference = reference_gradients[layer];
            const double dot = h1_detail::matrix_dot(gradient, reference);
            const double norm = gradient.norm();
            const double ref_norm = reference.norm();
            const double cosine = (norm > 0.0 && ref_norm > 0.0) ? (dot / (norm * ref_norm)) : 0.0;
            const double norm_ratio = ref_norm > 0.0 ? (norm / ref_norm) : 0.0;
            comparison.layer_cosines.push_back(cosine);
            comparison.layer_norm_ratios.push_back(norm_ratio);

            global_dot += dot;
            global_norm += norm * norm;
            reference_norm += ref_norm * ref_norm;
        }

        comparison.overall_cosine = (global_norm > 0.0 && reference_norm > 0.0)
            ? (global_dot / (std::sqrt(global_norm) * std::sqrt(reference_norm)))
            : 0.0;
        comparison.overall_norm_ratio =
            reference_norm > 0.0 ? (std::sqrt(global_norm) / std::sqrt(reference_norm)) : 0.0;
        comparisons.push_back(std::move(comparison));
    }

    ensure_directory(config.output_path.parent_path());
    std::ofstream output(config.output_path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + config.output_path.string());
    }

    output << "{\n"
           << "  \"checkpoint\": \"" << config.checkpoint_path.generic_string() << "\",\n"
           << "  \"data_dir\": \"" << config.data_dir.generic_string() << "\",\n"
           << "  \"backend\": \"" << backend_to_string(config.backend) << "\",\n"
           << "  \"visible_loss\": \"" << visible_loss_to_string(config.visible_loss) << "\",\n"
           << "  \"batch_size\": " << config.batch_size << ",\n"
           << "  \"seed\": " << config.seed << ",\n"
           << "  \"reference_steps\": " << reference_steps << ",\n"
           << "  \"sample_indices\": [";
    for (std::size_t i = 0; i < all_indices.size(); ++i) {
        if (i > 0) {
            output << ",";
        }
        output << all_indices[i];
    }
    output << "],\n"
           << "  \"comparisons\": [\n";

    for (std::size_t i = 0; i < comparisons.size(); ++i) {
        const h1_detail::StepComparison& comparison = comparisons[i];
        output << "    {\n"
               << "      \"steps\": " << comparison.steps << ",\n"
               << "      \"final_energy\": " << h1_detail::format_double(comparison.final_energy) << ",\n"
               << "      \"final_mse\": " << h1_detail::format_double(comparison.final_mse) << ",\n"
               << "      \"overall_cosine\": " << h1_detail::format_double(comparison.overall_cosine) << ",\n"
               << "      \"overall_norm_ratio\": " << h1_detail::format_double(comparison.overall_norm_ratio) << ",\n"
               << "      \"layer_cosines\": [";
        for (std::size_t layer = 0; layer < comparison.layer_cosines.size(); ++layer) {
            if (layer > 0) {
                output << ",";
            }
            output << h1_detail::format_double(comparison.layer_cosines[layer]);
        }
        output << "],\n"
               << "      \"layer_norm_ratios\": [";
        for (std::size_t layer = 0; layer < comparison.layer_norm_ratios.size(); ++layer) {
            if (layer > 0) {
                output << ",";
            }
            output << h1_detail::format_double(comparison.layer_norm_ratios[layer]);
        }
        output << "]\n"
               << "    }";
        if (i + 1 < comparisons.size()) {
            output << ",";
        }
        output << "\n";
    }

    output << "  ]\n"
           << "}\n";

    std::cout << "H1 gradient-bias diagnostic written to " << config.output_path << "\n";
    std::cout << "reference_steps=" << reference_steps << "\n";
    for (const h1_detail::StepComparison& comparison : comparisons) {
        std::cout << "steps=" << comparison.steps
                  << " overall_cosine=" << h1_detail::format_double(comparison.overall_cosine)
                  << " overall_norm_ratio=" << h1_detail::format_double(comparison.overall_norm_ratio)
                  << " final_mse=" << h1_detail::format_double(comparison.final_mse)
                  << " final_energy=" << h1_detail::format_double(comparison.final_energy)
                  << "\n";
    }

    return 0;
}

}  // namespace ltfn

int main(int argc, char** argv) {
    try {
        ltfn::h1_detail::H1Config config;
        std::string error_message;
        const bool parsed = ltfn::h1_detail::parse_h1_args(argc, argv, config, error_message);
        if (!parsed) {
            if (!error_message.empty()) {
                std::cerr << "Argument error: " << error_message << "\n\n";
            }
            std::cout << ltfn::h1_detail::h1_usage_text(argv[0]);
            return error_message.empty() ? 0 : 1;
        }
        return ltfn::run_h1_gradient_bias(config);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << '\n';
        return 1;
    }
}
