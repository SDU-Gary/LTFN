#include "utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#include <Eigen/SVD>

namespace ltfn {

namespace {

constexpr std::array<char, 8> kCheckpointMagic{'L', 'T', 'F', 'N', 'C', 'K', 'P', '1'};
constexpr std::uint32_t kCheckpointVersion = 1U;

std::uint32_t read_big_endian_u32(std::istream& stream) {
    std::array<unsigned char, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("Failed to read big-endian uint32.");
    }
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) |
           static_cast<std::uint32_t>(bytes[3]);
}

void write_binary(std::ostream& stream, const void* data, std::size_t size) {
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

template <typename T>
void write_pod(std::ostream& stream, const T& value) {
    write_binary(stream, &value, sizeof(T));
}

template <typename T>
void read_pod(std::istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
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

bool parse_bool_string(const std::string& value, bool& parsed) {
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        parsed = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        parsed = false;
        return true;
    }
    return false;
}

RunMode parse_mode(const std::string& value) {
    if (value == "train") {
        return RunMode::Train;
    }
    if (value == "eval") {
        return RunMode::Eval;
    }
    if (value == "smoke-test") {
        return RunMode::SmokeTest;
    }
    throw std::invalid_argument("Unknown mode: " + value);
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

std::vector<int> parse_dims(const std::string& value) {
    std::vector<int> dims;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        dims.push_back(std::stoi(token));
    }
    if (dims.empty()) {
        throw std::invalid_argument("Layer dimensions must not be empty.");
    }
    return dims;
}

void append_argument_value(int& index, int argc, char** argv, std::string& value, const std::string& key) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + key);
    }
    value = argv[++index];
}

std::string config_to_json(const ExperimentConfig& config, int argc, char** argv) {
    std::ostringstream stream;
    stream << "{\n"
           << "  \"timestamp_utc\": \"" << current_timestamp_utc() << "\",\n"
           << "  \"mode\": \"" << mode_to_string(config.mode) << "\",\n"
           << "  \"backend\": \"" << backend_to_string(config.backend) << "\",\n"
           << "  \"data_dir\": " << path_to_json(config.data_dir) << ",\n"
           << "  \"output_dir\": " << path_to_json(config.output_dir) << ",\n"
           << "  \"checkpoint_path\": " << path_to_json(config.checkpoint_path) << ",\n"
           << "  \"resume_path\": " << path_to_json(config.resume_path) << ",\n"
           << "  \"save_initial_checkpoint_path\": " << path_to_json(config.save_initial_checkpoint_path) << ",\n"
           << "  \"dims\": " << dims_to_json(config.dims) << ",\n"
           << "  \"tau_r\": " << config.tau_r << ",\n"
           << "  \"lr_w\": " << config.lr_w << ",\n"
           << "  \"lr_w_final\": " << config.lr_w_final << ",\n"
           << "  \"dt_r\": " << config.dt_r << ",\n"
           << "  \"dt_w\": " << config.dt_w << ",\n"
           << "  \"visible_loss\": \"" << visible_loss_to_string(config.visible_loss) << "\",\n"
           << "  \"momentum_beta\": " << config.momentum_beta << ",\n"
           << "  \"layer_adapt_beta\": " << config.layer_adapt_beta << ",\n"
           << "  \"layer_adapt_epsilon\": " << config.layer_adapt_epsilon << ",\n"
           << "  \"decorrelation_lambda\": " << config.decorrelation_lambda << ",\n"
           << "  \"steps\": " << config.steps << ",\n"
            << "  \"batch_size\": " << config.batch_size << ",\n"
           << "  \"max_epochs\": " << config.max_epochs << ",\n"
           << "  \"max_train_samples\": " << config.max_train_samples << ",\n"
           << "  \"eval_samples\": " << config.eval_samples << ",\n"
           << "  \"eval_interval\": " << config.eval_interval << ",\n"
           << "  \"checkpoint_interval\": " << config.checkpoint_interval << ",\n"
           << "  \"probe_index\": " << config.probe_index << ",\n"
           << "  \"recon_samples_to_save\": " << config.recon_samples_to_save << ",\n"
           << "  \"logger_step_interval\": " << config.logger_step_interval << ",\n"
           << "  \"logger_tags\": \"" << escape_json(config.logger_tags) << "\",\n"
           << "  \"cosine_lr_schedule\": " << bool_to_json(config.cosine_lr_schedule) << ",\n"
           << "  \"shuffle\": " << bool_to_json(config.shuffle) << ",\n"
           << "  \"seed\": " << config.seed << ",\n"
           << "  \"command_line\": \"" << escape_json(join_command_line(argc, argv)) << "\"\n"
           << "}\n";
    return stream.str();
}

std::string format_double(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

double clamp_sigmoid_input(double value) {
    return std::max(-500.0, std::min(500.0, value));
}

double sigmoid_scalar(double value) {
    return 1.0 / (1.0 + std::exp(-clamp_sigmoid_input(value)));
}

int effective_dimension_90(const Eigen::VectorXd& singular_values) {
    if (singular_values.size() == 0) {
        return 0;
    }
    const Eigen::VectorXd power = singular_values.array().square().matrix();
    const double total_power = power.sum();
    if (total_power <= 0.0) {
        return 0;
    }

    double cumulative_power = 0.0;
    for (Eigen::Index i = 0; i < power.size(); ++i) {
        cumulative_power += power(i);
        if (cumulative_power / total_power >= 0.9) {
            return static_cast<int>(i + 1);
        }
    }
    return static_cast<int>(power.size());
}

}  // namespace

bool parse_args(int argc, char** argv, ExperimentConfig& config, std::string& error_message) {
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            std::string value;

            if (key == "--help" || key == "-h") {
                error_message.clear();
                return false;
            }

            if (key == "--mode") {
                append_argument_value(i, argc, argv, value, key);
                config.mode = parse_mode(value);
            } else if (key == "--backend") {
                append_argument_value(i, argc, argv, value, key);
                config.backend = parse_backend(value);
            } else if (key == "--data-dir") {
                append_argument_value(i, argc, argv, value, key);
                config.data_dir = value;
            } else if (key == "--output-dir") {
                append_argument_value(i, argc, argv, value, key);
                config.output_dir = value;
            } else if (key == "--checkpoint") {
                append_argument_value(i, argc, argv, value, key);
                config.checkpoint_path = value;
            } else if (key == "--resume") {
                append_argument_value(i, argc, argv, value, key);
                config.resume_path = value;
            } else if (key == "--save-initial-checkpoint") {
                append_argument_value(i, argc, argv, value, key);
                config.save_initial_checkpoint_path = value;
            } else if (key == "--dims") {
                append_argument_value(i, argc, argv, value, key);
                config.dims = parse_dims(value);
            } else if (key == "--tau-r") {
                append_argument_value(i, argc, argv, value, key);
                config.tau_r = std::stod(value);
            } else if (key == "--lr-w") {
                append_argument_value(i, argc, argv, value, key);
                config.lr_w = std::stod(value);
            } else if (key == "--lr-final") {
                append_argument_value(i, argc, argv, value, key);
                config.lr_w_final = std::stod(value);
            } else if (key == "--dt-r") {
                append_argument_value(i, argc, argv, value, key);
                config.dt_r = std::stod(value);
            } else if (key == "--dt-w") {
                append_argument_value(i, argc, argv, value, key);
                config.dt_w = std::stod(value);
            } else if (key == "--visible-loss") {
                append_argument_value(i, argc, argv, value, key);
                config.visible_loss = parse_visible_loss(value);
            } else if (key == "--momentum-beta") {
                append_argument_value(i, argc, argv, value, key);
                config.momentum_beta = std::stod(value);
            } else if (key == "--layer-adapt-beta") {
                append_argument_value(i, argc, argv, value, key);
                config.layer_adapt_beta = std::stod(value);
            } else if (key == "--layer-adapt-eps") {
                append_argument_value(i, argc, argv, value, key);
                config.layer_adapt_epsilon = std::stod(value);
            } else if (key == "--decorrelation-lambda") {
                append_argument_value(i, argc, argv, value, key);
                config.decorrelation_lambda = std::stod(value);
            } else if (key == "--steps") {
                append_argument_value(i, argc, argv, value, key);
                config.steps = std::stoi(value);
            } else if (key == "--batch-size") {
                append_argument_value(i, argc, argv, value, key);
                config.batch_size = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--max-epochs") {
                append_argument_value(i, argc, argv, value, key);
                config.max_epochs = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--max-train-samples") {
                append_argument_value(i, argc, argv, value, key);
                config.max_train_samples = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--eval-samples") {
                append_argument_value(i, argc, argv, value, key);
                config.eval_samples = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--eval-interval") {
                append_argument_value(i, argc, argv, value, key);
                config.eval_interval = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--checkpoint-interval") {
                append_argument_value(i, argc, argv, value, key);
                config.checkpoint_interval = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--probe-index") {
                append_argument_value(i, argc, argv, value, key);
                config.probe_index = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--recon-samples") {
                append_argument_value(i, argc, argv, value, key);
                config.recon_samples_to_save = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--logger-step-interval") {
                append_argument_value(i, argc, argv, value, key);
                config.logger_step_interval = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--logger-tags") {
                append_argument_value(i, argc, argv, value, key);
                config.logger_tags = value;
            } else if (key == "--seed") {
                append_argument_value(i, argc, argv, value, key);
                config.seed = static_cast<std::uint32_t>(std::stoul(value));
            } else if (key == "--shuffle") {
                append_argument_value(i, argc, argv, value, key);
                bool shuffle = config.shuffle;
                if (!parse_bool_string(value, shuffle)) {
                    throw std::invalid_argument("Expected boolean for --shuffle, got: " + value);
                }
                config.shuffle = shuffle;
            } else if (key == "--cosine-lr") {
                append_argument_value(i, argc, argv, value, key);
                bool cosine_lr_schedule = config.cosine_lr_schedule;
                if (!parse_bool_string(value, cosine_lr_schedule)) {
                    throw std::invalid_argument("Expected boolean for --cosine-lr, got: " + value);
                }
                config.cosine_lr_schedule = cosine_lr_schedule;
            } else {
                throw std::invalid_argument("Unknown argument: " + key);
            }
        }

        if (config.steps <= 0) {
            throw std::invalid_argument("--steps must be positive.");
        }
        if (config.batch_size == 0) {
            throw std::invalid_argument("--batch-size must be positive.");
        }
        if (config.tau_r <= 0.0) {
            throw std::invalid_argument("--tau-r must be positive.");
        }
        if (config.lr_w < 0.0) {
            throw std::invalid_argument("--lr-w must be non-negative.");
        }
        if (config.lr_w_final < 0.0) {
            throw std::invalid_argument("--lr-final must be non-negative.");
        }
        if (config.momentum_beta < 0.0 || config.momentum_beta >= 1.0) {
            throw std::invalid_argument("--momentum-beta must be in [0, 1).");
        }
        if (config.layer_adapt_beta < 0.0 || config.layer_adapt_beta >= 1.0) {
            throw std::invalid_argument("--layer-adapt-beta must be in [0, 1).");
        }
        if (config.layer_adapt_epsilon <= 0.0) {
            throw std::invalid_argument("--layer-adapt-eps must be positive.");
        }
        if (config.decorrelation_lambda < 0.0) {
            throw std::invalid_argument("--decorrelation-lambda must be non-negative.");
        }
        if (config.cosine_lr_schedule && config.lr_w_final <= 0.0) {
            throw std::invalid_argument("--lr-final must be positive when --cosine-lr is enabled.");
        }
        if (config.logger_step_interval == 0) {
            throw std::invalid_argument("--logger-step-interval must be positive.");
        }
        if (config.dims.size() < 2 || config.dims.front() != 784) {
            throw std::invalid_argument("--dims must contain at least two layers and start with 784.");
        }
        if (config.mode != RunMode::SmokeTest && config.data_dir.empty()) {
            throw std::invalid_argument("--data-dir is required unless --mode smoke-test is used.");
        }
        if (config.checkpoint_path.empty()) {
            config.checkpoint_path = config.output_dir / "checkpoints" / "latest.ltfnckpt";
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
        << "Options:\n"
        << "  --mode train|eval|smoke-test    Run mode (default: train)\n"
        << "  --backend cpu|cuda              Compute backend (default: cpu)\n"
        << "  --data-dir PATH                 Directory containing MNIST ubyte files\n"
        << "  --output-dir PATH               Output root (default: runs/latest)\n"
        << "  --resume PATH                   Checkpoint to load before running\n"
        << "  --checkpoint PATH               Checkpoint output path\n"
        << "  --save-initial-checkpoint PATH  Save model state before any training step\n"
        << "  --dims 784,256,64,32            Layer dimensions\n"
        << "  --tau-r FLOAT                   State time constant (default: 0.1)\n"
        << "  --lr-w FLOAT                    Weight learning rate (default: 1e-5)\n"
        << "  --lr-final FLOAT                Final lr for cosine schedule (default: 1e-6)\n"
        << "  --dt-r FLOAT                    State step size (default: 0.1)\n"
        << "  --dt-w FLOAT                    Weight step size (default: 1.0)\n"
        << "  --visible-loss mse|bce          Visible-layer energy term (default: mse)\n"
        << "  --momentum-beta FLOAT           SGD momentum beta in [0,1) (default: 0.0)\n"
        << "  --layer-adapt-beta FLOAT        Per-layer RMS gradient EMA beta, 0 disables (default: 0.0)\n"
        << "  --layer-adapt-eps FLOAT         Epsilon for layer-adaptive scaling (default: 1e-8)\n"
        << "  --decorrelation-lambda FLOAT    Hidden-activation decorrelation strength (default: 0.0)\n"
        << "  --steps INT                     Relaxation steps per sample (default: 200)\n"
        << "  --batch-size INT                Training mini-batch size (default: 32)\n"
        << "  --max-epochs INT                Maximum epochs for train mode (default: 1)\n"
        << "  --max-train-samples INT         Training sample cap, 0 means full dataset\n"
        << "  --eval-samples INT              Number of samples used during evaluation\n"
        << "  --eval-interval INT             Evaluate every N training samples\n"
        << "  --checkpoint-interval INT       Save checkpoint every N training samples\n"
        << "  --probe-index INT               Fixed probe sample index for monitoring\n"
        << "  --recon-samples INT             Number of reconstructions saved per eval\n"
        << "  --logger-step-interval INT      Write step logs every N relaxation steps\n"
        << "  --logger-tags LIST              Comma-separated tags: all,core,energy,layer-errors,\n"
        << "                                  gradients,eval,reconstructions,events\n"
        << "  --cosine-lr true|false          Enable cosine lr schedule (default: false)\n"
        << "  --shuffle true|false            Shuffle training set each epoch (default: true)\n"
        << "  --seed INT                      Random seed (default: 42)\n"
        << "  --help                          Show this help message\n";
    return stream.str();
}

std::string mode_to_string(RunMode mode) {
    switch (mode) {
        case RunMode::Train:
            return "train";
        case RunMode::Eval:
            return "eval";
        case RunMode::SmokeTest:
            return "smoke-test";
    }
    return "unknown";
}

std::string join_command_line(int argc, char** argv) {
    std::ostringstream stream;
    for (int i = 0; i < argc; ++i) {
        if (i > 0) {
            stream << ' ';
        }
        stream << argv[i];
    }
    return stream.str();
}

Dataset load_mnist_images(const fs::path& image_file, std::size_t limit) {
    std::ifstream stream(image_file, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open MNIST file: " + image_file.string());
    }

    const std::uint32_t magic = read_big_endian_u32(stream);
    const std::uint32_t count = read_big_endian_u32(stream);
    const std::uint32_t rows = read_big_endian_u32(stream);
    const std::uint32_t cols = read_big_endian_u32(stream);

    if (magic != 2051U) {
        throw std::runtime_error("Invalid MNIST image magic number in " + image_file.string());
    }
    if (rows == 0U || cols == 0U) {
        throw std::runtime_error("MNIST image dimensions must be non-zero.");
    }

    const std::size_t image_size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t to_read = limit == 0 ? count : std::min<std::size_t>(count, limit);

    Dataset dataset;
    dataset.rows = rows;
    dataset.cols = cols;
    dataset.images.reserve(to_read);

    std::vector<unsigned char> buffer(image_size);
    for (std::size_t index = 0; index < to_read; ++index) {
        stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        if (!stream) {
            throw std::runtime_error("Failed to read image data from " + image_file.string());
        }

        Eigen::VectorXd image(static_cast<Eigen::Index>(image_size));
        for (std::size_t pixel = 0; pixel < image_size; ++pixel) {
            image(static_cast<Eigen::Index>(pixel)) = static_cast<double>(buffer[pixel]) / 255.0;
        }
        dataset.images.push_back(std::move(image));
    }

    return dataset;
}

Dataset generate_smoke_dataset(std::size_t count, std::uint32_t seed) {
    Dataset dataset;
    dataset.rows = 28;
    dataset.cols = 28;
    dataset.images.reserve(count);

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> jitter(-0.05, 0.05);

    const auto make_canvas = []() {
        return Eigen::VectorXd::Zero(28 * 28);
    };

    std::vector<Eigen::VectorXd> patterns;

    {
        Eigen::VectorXd img = make_canvas();
        for (int r = 6; r < 22; ++r) {
            img(r * 28 + 14) = 1.0;
        }
        patterns.push_back(std::move(img));
    }
    {
        Eigen::VectorXd img = make_canvas();
        for (int c = 6; c < 22; ++c) {
            img(14 * 28 + c) = 1.0;
        }
        patterns.push_back(std::move(img));
    }
    {
        Eigen::VectorXd img = make_canvas();
        for (int d = 6; d < 22; ++d) {
            img(d * 28 + d) = 1.0;
        }
        patterns.push_back(std::move(img));
    }
    {
        Eigen::VectorXd img = make_canvas();
        for (int d = 6; d < 22; ++d) {
            img(d * 28 + (27 - d)) = 1.0;
        }
        patterns.push_back(std::move(img));
    }
    {
        Eigen::VectorXd img = make_canvas();
        for (int r = 8; r < 20; ++r) {
            for (int c = 8; c < 20; ++c) {
                if (r == 8 || r == 19 || c == 8 || c == 19) {
                    img(r * 28 + c) = 1.0;
                }
            }
        }
        patterns.push_back(std::move(img));
    }

    for (std::size_t i = 0; i < count; ++i) {
        Eigen::VectorXd image = patterns[i % patterns.size()];
        for (Eigen::Index idx = 0; idx < image.size(); ++idx) {
            image(idx) = std::clamp(image(idx) + jitter(generator), 0.0, 1.0);
        }
        dataset.images.push_back(std::move(image));
    }

    return dataset;
}

void ensure_directory(const fs::path& path) {
    if (!path.empty()) {
        fs::create_directories(path);
    }
}

MonitoringContext create_monitoring_context(const ExperimentConfig& config, int argc, char** argv) {
    MonitoringContext monitoring;
    monitoring.root_dir = config.output_dir;
    monitoring.reconstructions_dir = config.output_dir / "reconstructions";
    monitoring.checkpoints_dir = config.output_dir / "checkpoints";

    ensure_directory(monitoring.root_dir);
    ensure_directory(monitoring.reconstructions_dir);
    ensure_directory(monitoring.checkpoints_dir);

    monitoring.metrics_csv.open(monitoring.root_dir / "metrics.csv", std::ios::out | std::ios::trunc);
    monitoring.events_jsonl.open(monitoring.root_dir / "events.jsonl", std::ios::out | std::ios::trunc);
    if (!monitoring.metrics_csv || !monitoring.events_jsonl) {
        throw std::runtime_error("Failed to open monitoring output files under " + config.output_dir.string());
    }

    monitoring.metrics_csv
        << "timestamp_utc,epoch,samples_seen,train_window_mse,eval_mse,eval_energy,probe_mse,probe_energy,"
           "images_per_sec,elapsed_sec,avg_weight_norm,avg_gradient_norm\n";

    write_config_snapshot(monitoring, config, argc, argv);
    append_event(monitoring, "run_started", "{\"mode\":\"" + mode_to_string(config.mode) + "\"}");
    return monitoring;
}

void write_config_snapshot(const MonitoringContext& monitoring, const ExperimentConfig& config, int argc, char** argv) {
    std::ofstream config_file(monitoring.root_dir / "config.json", std::ios::out | std::ios::trunc);
    if (!config_file) {
        throw std::runtime_error("Failed to write config snapshot.");
    }
    config_file << config_to_json(config, argc, argv);
}

void append_metric(MonitoringContext& monitoring, const MetricRow& row) {
    monitoring.metrics_csv
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
    monitoring.metrics_csv.flush();
}

void append_event(MonitoringContext& monitoring, const std::string& type, const std::string& payload_json) {
    monitoring.events_jsonl
        << "{\"timestamp_utc\":\"" << current_timestamp_utc()
        << "\",\"type\":\"" << escape_json(type)
        << "\",\"payload\":" << payload_json
        << "}\n";
    monitoring.events_jsonl.flush();
}

void write_pgm(const fs::path& file_path, const Eigen::VectorXd& image, std::size_t rows, std::size_t cols) {
    const std::size_t expected_size = rows * cols;
    if (static_cast<std::size_t>(image.size()) != expected_size) {
        throw std::invalid_argument("Image size does not match the requested PGM shape.");
    }

    std::ofstream stream(file_path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Failed to open PGM output: " + file_path.string());
    }

    stream << "P5\n" << cols << ' ' << rows << "\n255\n";
    for (Eigen::Index i = 0; i < image.size(); ++i) {
        const double clamped = std::clamp(image(i), 0.0, 1.0);
        const auto pixel = static_cast<unsigned char>(std::lround(clamped * 255.0));
        stream.write(reinterpret_cast<const char*>(&pixel), 1);
    }
}

void write_side_by_side_pgm(
    const fs::path& file_path,
    const Eigen::VectorXd& left,
    const Eigen::VectorXd& right,
    std::size_t rows,
    std::size_t cols) {
    const std::size_t expected_size = rows * cols;
    if (static_cast<std::size_t>(left.size()) != expected_size ||
        static_cast<std::size_t>(right.size()) != expected_size) {
        throw std::invalid_argument("Image sizes do not match the requested side-by-side PGM shape.");
    }

    std::ofstream stream(file_path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Failed to open PGM output: " + file_path.string());
    }

    const std::size_t merged_cols = cols * 2;
    stream << "P5\n" << merged_cols << ' ' << rows << "\n255\n";
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const auto left_pixel = static_cast<unsigned char>(
                std::lround(std::clamp(left(static_cast<Eigen::Index>(row * cols + col)), 0.0, 1.0) * 255.0));
            stream.write(reinterpret_cast<const char*>(&left_pixel), 1);
        }
        for (std::size_t col = 0; col < cols; ++col) {
            const auto right_pixel = static_cast<unsigned char>(
                std::lround(std::clamp(right(static_cast<Eigen::Index>(row * cols + col)), 0.0, 1.0) * 255.0));
            stream.write(reinterpret_cast<const char*>(&right_pixel), 1);
        }
    }
}

double average_weight_norm(const std::vector<Eigen::MatrixXd>& weights) {
    if (weights.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& weight : weights) {
        total += weight.norm();
    }
    return total / static_cast<double>(weights.size());
}

double average_norm(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    return total / static_cast<double>(values.size());
}

bool save_checkpoint(
    const fs::path& file_path,
    const ILTFNModel& model,
    const CheckpointState& state,
    std::string& error_message) {
    try {
        ensure_directory(file_path.parent_path());
        std::ofstream stream(file_path, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("Failed to open checkpoint for writing: " + file_path.string());
        }

        write_binary(stream, kCheckpointMagic.data(), kCheckpointMagic.size());
        write_pod(stream, kCheckpointVersion);

        const auto& config = model.config();
        const std::uint32_t dim_count = static_cast<std::uint32_t>(config.dims.size());
        write_pod(stream, dim_count);
        for (int dim : config.dims) {
            const std::int32_t encoded = static_cast<std::int32_t>(dim);
            write_pod(stream, encoded);
        }

        write_pod(stream, config.tau_r);
        write_pod(stream, config.lr_w);
        write_pod(stream, config.dt_r);
        write_pod(stream, config.dt_w);
        write_pod(stream, state.samples_seen);
        write_pod(stream, state.epochs_completed);
        write_pod(stream, state.seed);

        const auto& weights = model.weights();
        const std::uint32_t weight_count = static_cast<std::uint32_t>(weights.size());
        write_pod(stream, weight_count);
        for (const auto& weight : weights) {
            const std::int32_t rows = static_cast<std::int32_t>(weight.rows());
            const std::int32_t cols = static_cast<std::int32_t>(weight.cols());
            write_pod(stream, rows);
            write_pod(stream, cols);
            write_binary(stream, weight.data(), sizeof(double) * static_cast<std::size_t>(rows * cols));
        }

        if (!stream) {
            throw std::runtime_error("Failed while writing checkpoint: " + file_path.string());
        }
        return true;
    } catch (const std::exception& ex) {
        error_message = ex.what();
        return false;
    }
}

bool load_checkpoint(
    const fs::path& file_path,
    ILTFNModel& model,
    CheckpointState& state,
    std::string& error_message) {
    try {
        std::ifstream stream(file_path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("Failed to open checkpoint: " + file_path.string());
        }

        std::array<char, 8> magic{};
        stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (magic != kCheckpointMagic) {
            throw std::runtime_error("Checkpoint magic number mismatch.");
        }

        std::uint32_t version = 0U;
        read_pod(stream, version);
        if (version != kCheckpointVersion) {
            throw std::runtime_error("Unsupported checkpoint version.");
        }

        std::uint32_t dim_count = 0U;
        read_pod(stream, dim_count);
        std::vector<int> dims;
        dims.reserve(dim_count);
        for (std::uint32_t i = 0; i < dim_count; ++i) {
            std::int32_t dim = 0;
            read_pod(stream, dim);
            dims.push_back(dim);
        }

        double tau_r = 0.0;
        double lr_w = 0.0;
        double dt_r = 0.0;
        double dt_w = 0.0;
        read_pod(stream, tau_r);
        read_pod(stream, lr_w);
        read_pod(stream, dt_r);
        read_pod(stream, dt_w);
        read_pod(stream, state.samples_seen);
        read_pod(stream, state.epochs_completed);
        read_pod(stream, state.seed);

        const auto& config = model.config();
        if (dims != config.dims ||
            std::abs(tau_r - config.tau_r) > 1e-12 ||
            std::abs(lr_w - config.lr_w) > 1e-12 ||
            std::abs(dt_r - config.dt_r) > 1e-12 ||
            std::abs(dt_w - config.dt_w) > 1e-12) {
            throw std::runtime_error("Checkpoint configuration does not match the current model.");
        }

        std::uint32_t weight_count = 0U;
        read_pod(stream, weight_count);
        std::vector<Eigen::MatrixXd> weights;
        weights.reserve(weight_count);

        for (std::uint32_t i = 0; i < weight_count; ++i) {
            std::int32_t rows = 0;
            std::int32_t cols = 0;
            read_pod(stream, rows);
            read_pod(stream, cols);
            Eigen::MatrixXd weight(rows, cols);
            stream.read(reinterpret_cast<char*>(weight.data()), sizeof(double) * static_cast<std::size_t>(rows * cols));
            if (!stream) {
                throw std::runtime_error("Failed while reading checkpoint weights.");
            }
            weights.push_back(std::move(weight));
        }

        model.set_weights(weights);
        return true;
    } catch (const std::exception& ex) {
        error_message = ex.what();
        return false;
    }
}

EvaluationResult evaluate_model(
    ILTFNModel& model,
    const Dataset& dataset,
    int steps,
    std::size_t limit,
    std::size_t reconstructions_to_keep) {
    if (dataset.images.empty()) {
        throw std::invalid_argument("Cannot evaluate on an empty dataset.");
    }

    const std::size_t eval_count = limit == 0 ? dataset.images.size() : std::min(limit, dataset.images.size());
    EvaluationResult result;
    result.reconstructions.reserve(std::min(reconstructions_to_keep, eval_count));
    result.top_representations.reserve(eval_count);

    const auto& config = model.config();
    const auto& weights = model.weights();
    const std::size_t hidden_layer_count = config.dims.size() - 1;
    std::vector<Eigen::MatrixXd> hidden_activations;
    hidden_activations.reserve(hidden_layer_count);
    for (std::size_t layer = 1; layer < config.dims.size(); ++layer) {
        hidden_activations.emplace_back(
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(eval_count), config.dims[layer]));
    }

    std::vector<double> error_sums(hidden_layer_count, 0.0);
    std::vector<double> error_square_sums(hidden_layer_count, 0.0);

    double mse_total = 0.0;
    double energy_total = 0.0;
    for (std::size_t i = 0; i < eval_count; ++i) {
        RelaxationResult sample = model.reconstruct(dataset.images[i], steps, false);
        mse_total += sample.mse;
        energy_total += sample.final_energy;
        const auto& states = model.states();
        result.top_representations.push_back(states.back());
        for (std::size_t layer = 1; layer < config.dims.size(); ++layer) {
            hidden_activations[layer - 1].row(static_cast<Eigen::Index>(i)) = states[layer].transpose();
        }
        for (std::size_t layer = 0; layer < weights.size(); ++layer) {
            const Eigen::VectorXd pre_activation =
                weights[layer] * states[layer + 1];
            const Eigen::VectorXd prediction =
                pre_activation.unaryExpr([](double value) { return sigmoid_scalar(value); });
            const Eigen::VectorXd error = states[layer] - prediction;
            error_sums[layer] += error.sum();
            error_square_sums[layer] += error.array().square().sum();
        }
        if (i < reconstructions_to_keep) {
            result.reconstructions.push_back(std::move(sample.reconstruction));
        }
    }

    result.average_mse = mse_total / static_cast<double>(eval_count);
    result.average_energy = energy_total / static_cast<double>(eval_count);
    result.effective_dimensions.reserve(hidden_layer_count);
    result.error_variances.reserve(weights.size());

    for (std::size_t layer = 1; layer < config.dims.size(); ++layer) {
        const Eigen::MatrixXd centered =
            hidden_activations[layer - 1].rowwise() - hidden_activations[layer - 1].colwise().mean();
        Eigen::BDCSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd singular_values = svd.singularValues();
        const int effective_dim = effective_dimension_90(singular_values);
        EffectiveDimensionRow row;
        row.layer = layer;
        row.width = config.dims[layer];
        row.samples = eval_count;
        row.effective_dim_90 = effective_dim;
        row.effective_dim_ratio =
            row.width > 0 ? static_cast<double>(effective_dim) / static_cast<double>(row.width) : 0.0;
        row.largest_singular = singular_values.size() > 0 ? singular_values(0) : 0.0;
        row.smallest_singular = singular_values.size() > 0 ? singular_values(singular_values.size() - 1) : 0.0;
        row.spectrum_ratio =
            row.smallest_singular > 1e-12 ? row.largest_singular / row.smallest_singular : std::numeric_limits<double>::infinity();
        result.effective_dimensions.push_back(row);
    }

    for (std::size_t layer = 0; layer < weights.size(); ++layer) {
        const double count = static_cast<double>(eval_count) * static_cast<double>(config.dims[layer]);
        const double mean = count > 0.0 ? error_sums[layer] / count : 0.0;
        const double mean_square = count > 0.0 ? error_square_sums[layer] / count : 0.0;
        ErrorVarianceRow row;
        row.layer = layer;
        row.width = config.dims[layer];
        row.samples = eval_count;
        row.error_mean = mean;
        row.error_mean_square = mean_square;
        row.error_variance = std::max(0.0, mean_square - mean * mean);
        result.error_variances.push_back(row);
    }
    return result;
}

}  // namespace ltfn
