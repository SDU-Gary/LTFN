#include "ltfn.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ltfn {
namespace h3_detail {

struct Config {
    fs::path checkpoint_path;
    fs::path data_dir;
    fs::path output_dir{"runs/h3_repr_dump"};
    ComputeBackend backend{ComputeBackend::Cuda};
    std::vector<int> dims{784, 512, 256, 128, 64};
    double tau_r{0.1};
    double lr_w{3.2e-5};
    double dt_r{0.1};
    double dt_w{1.0};
    VisibleLoss visible_loss{VisibleLoss::Bce};
    int steps{2000};
    std::size_t sample_count{1000};
    std::size_t batch_size{64};
    std::uint32_t seed{42U};
    std::string split{"test"};
};

std::vector<int> parse_int_list(const std::string& value) {
    std::vector<int> dims;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            dims.push_back(std::stoi(token));
        }
    }
    if (dims.empty()) {
        throw std::invalid_argument("Expected a non-empty integer list.");
    }
    return dims;
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

bool parse_args(int argc, char** argv, Config& config, std::string& error_message) {
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
            } else if (key == "--output-dir") {
                require_value(value);
                config.output_dir = value;
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
            } else if (key == "--steps") {
                require_value(value);
                config.steps = std::stoi(value);
            } else if (key == "--samples") {
                require_value(value);
                config.sample_count = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--batch-size") {
                require_value(value);
                config.batch_size = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--seed") {
                require_value(value);
                config.seed = static_cast<std::uint32_t>(std::stoul(value));
            } else if (key == "--split") {
                require_value(value);
                config.split = value;
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
        if (config.steps <= 0) {
            throw std::invalid_argument("--steps must be positive.");
        }
        if (config.sample_count == 0 || config.batch_size == 0) {
            throw std::invalid_argument("--samples and --batch-size must be positive.");
        }
        if (config.split != "test" && config.split != "train") {
            throw std::invalid_argument("--split must be test or train.");
        }
        return true;
    } catch (const std::exception& ex) {
        error_message = ex.what();
        return false;
    }
}

std::string usage_text(const char* program_name) {
    std::ostringstream stream;
    stream
        << "Usage: " << program_name << " [options]\n\n"
        << "  --checkpoint PATH\n"
        << "  --data-dir PATH\n"
        << "  --output-dir PATH\n"
        << "  --backend cpu|cuda\n"
        << "  --dims LIST\n"
        << "  --tau-r FLOAT\n"
        << "  --lr-w FLOAT\n"
        << "  --dt-r FLOAT\n"
        << "  --dt-w FLOAT\n"
        << "  --visible-loss mse|bce\n"
        << "  --steps INT\n"
        << "  --samples INT\n"
        << "  --batch-size INT\n"
        << "  --split train|test\n";
    return stream.str();
}

void append_rows(std::ofstream& stream, const Eigen::MatrixXd& matrix) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            if (row > 0) {
                stream << ',';
            }
            stream << matrix(row, col);
        }
        stream << '\n';
    }
}

}  // namespace h3_detail

int run_h3_repr_dump(const h3_detail::Config& config) {
    const fs::path image_path = config.split == "test"
        ? (config.data_dir / "t10k-images-idx3-ubyte")
        : (config.data_dir / "train-images-idx3-ubyte");
    Dataset dataset = load_mnist_images(image_path, config.sample_count);
    if (dataset.images.empty()) {
        throw std::runtime_error("Requested dataset split is empty.");
    }

    LTFNConfig model_config;
    model_config.dims = config.dims;
    model_config.tau_r = config.tau_r;
    model_config.lr_w = config.lr_w;
    model_config.dt_r = config.dt_r;
    model_config.dt_w = config.dt_w;
    model_config.visible_loss = config.visible_loss;

    std::unique_ptr<ILTFNModel> model = create_model(model_config, config.seed, config.backend);
    CheckpointState checkpoint_state;
    std::string checkpoint_error;
    if (!load_checkpoint(config.checkpoint_path, *model, checkpoint_state, checkpoint_error)) {
        throw std::runtime_error("Failed to load checkpoint: " + checkpoint_error);
    }

    ensure_directory(config.output_dir);
    std::vector<std::ofstream> layer_streams;
    layer_streams.reserve(config.dims.size() - 1);
    for (std::size_t layer = 1; layer < config.dims.size(); ++layer) {
        layer_streams.emplace_back(config.output_dir / ("layer_" + std::to_string(layer) + ".csv"), std::ios::out | std::ios::trunc);
        if (!layer_streams.back()) {
            throw std::runtime_error("Failed to open layer output CSV.");
        }
    }

    for (std::size_t offset = 0; offset < dataset.images.size(); offset += config.batch_size) {
        const std::size_t count = std::min(config.batch_size, dataset.images.size() - offset);
        std::vector<const Eigen::VectorXd*> batch_inputs;
        batch_inputs.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            batch_inputs.push_back(&dataset.images[offset + i]);
        }

        BatchTrainOptions options;
        options.update_weights = false;
        options.capture_final_states = true;
        BatchTrainResult result = model->train_batch(batch_inputs, config.steps, options);
        if (result.final_batch_states.size() != config.dims.size()) {
            throw std::runtime_error("State capture failed.");
        }

        for (std::size_t layer = 1; layer < result.final_batch_states.size(); ++layer) {
            h3_detail::append_rows(layer_streams[layer - 1], result.final_batch_states[layer]);
        }

        std::cout << "processed " << std::min(offset + count, dataset.images.size())
                  << "/" << dataset.images.size() << "\n";
    }

    std::ofstream meta(config.output_dir / "metadata.json", std::ios::out | std::ios::trunc);
    meta << "{\n"
         << "  \"checkpoint\": \"" << config.checkpoint_path.generic_string() << "\",\n"
         << "  \"data_dir\": \"" << config.data_dir.generic_string() << "\",\n"
         << "  \"split\": \"" << config.split << "\",\n"
         << "  \"backend\": \"" << backend_to_string(config.backend) << "\",\n"
         << "  \"visible_loss\": \"" << visible_loss_to_string(config.visible_loss) << "\",\n"
         << "  \"steps\": " << config.steps << ",\n"
         << "  \"samples\": " << dataset.images.size() << ",\n"
         << "  \"batch_size\": " << config.batch_size << "\n"
         << "}\n";
    return 0;
}

}  // namespace ltfn

int main(int argc, char** argv) {
    try {
        ltfn::h3_detail::Config config;
        std::string error_message;
        if (!ltfn::h3_detail::parse_args(argc, argv, config, error_message)) {
            if (!error_message.empty()) {
                std::cerr << "Argument error: " << error_message << "\n\n";
            }
            std::cout << ltfn::h3_detail::usage_text(argv[0]);
            return error_message.empty() ? 0 : 1;
        }
        return ltfn::run_h3_repr_dump(config);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << '\n';
        return 1;
    }
}
