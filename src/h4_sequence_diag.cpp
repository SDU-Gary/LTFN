#include "ltfn.h"
#include "utils.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ltfn {
namespace h4_detail {

struct Config {
    fs::path checkpoint_path;
    fs::path data_dir;
    fs::path output_dir{"runs/h4_sequence_diag"};
    ComputeBackend backend{ComputeBackend::Cuda};
    std::vector<int> dims{784, 512, 256, 128, 64};
    double tau_r{0.1};
    double lr_w{3.2e-5};
    double dt_r{0.1};
    double dt_w{1.0};
    VisibleLoss visible_loss{VisibleLoss::Bce};
    int steps{200};
    std::size_t sequence_length{200};
    std::vector<int> sequence_labels{0, 1};
    std::vector<int> probe_labels{1, 5, 8};
    std::uint32_t seed{42U};
};

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

std::vector<std::uint8_t> load_mnist_labels(const fs::path& label_file, std::size_t limit) {
    std::ifstream stream(label_file, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open MNIST label file: " + label_file.string());
    }
    const std::uint32_t magic = read_big_endian_u32(stream);
    const std::uint32_t count = read_big_endian_u32(stream);
    if (magic != 2049U) {
        throw std::runtime_error("Invalid MNIST label magic number in " + label_file.string());
    }
    const std::size_t to_read = limit == 0 ? count : std::min<std::size_t>(count, limit);
    std::vector<std::uint8_t> labels(to_read);
    stream.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(to_read));
    if (!stream) {
        throw std::runtime_error("Failed to read MNIST labels from " + label_file.string());
    }
    return labels;
}

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
        throw std::invalid_argument("Expected a non-empty comma-separated list.");
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
            } else if (key == "--sequence-length") {
                require_value(value);
                config.sequence_length = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--sequence-labels") {
                require_value(value);
                config.sequence_labels = parse_int_list(value);
            } else if (key == "--probe-labels") {
                require_value(value);
                config.probe_labels = parse_int_list(value);
            } else if (key == "--seed") {
                require_value(value);
                config.seed = static_cast<std::uint32_t>(std::stoul(value));
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
        if (config.steps <= 0 || config.sequence_length == 0) {
            throw std::invalid_argument("--steps and --sequence-length must be positive.");
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
        << "Usage: " << program_name << " [options]\n"
        << "  --checkpoint PATH\n"
        << "  --data-dir PATH\n"
        << "  --output-dir PATH\n"
        << "  --visible-loss bce\n"
        << "  --steps 200\n"
        << "  --sequence-length 200\n"
        << "  --sequence-labels 0,1\n"
        << "  --probe-labels 1,5,8\n";
    return stream.str();
}

struct SequenceItem {
    std::size_t index{0};
    int label{0};
};

SequenceItem next_item(const std::unordered_map<int, std::vector<std::size_t>>& by_label, std::unordered_map<int, std::size_t>& offsets, int label) {
    auto it = by_label.find(label);
    if (it == by_label.end() || offsets[label] >= it->second.size()) {
        throw std::runtime_error("Not enough samples for label " + std::to_string(label));
    }
    SequenceItem item;
    item.index = it->second[offsets[label]++];
    item.label = label;
    return item;
}

StepDiagnostics process_zero_init(ILTFNModel& model, const Eigen::VectorXd& input, int steps) {
    model.reset_states(input);
    if (steps == 1) {
        return model.step_current(true);
    }
    for (int step = 1; step < steps; ++step) {
        model.advance_current(true);
    }
    return model.step_current(true);
}

StepDiagnostics process_warm_start(ILTFNModel& model, const Eigen::VectorXd& input, int steps, bool first_sample) {
    if (first_sample) {
        return process_zero_init(model, input, steps);
    }
    if (steps == 1) {
        return model.step(input, true);
    }
    model.advance(input, true);
    for (int step = 2; step < steps; ++step) {
        model.advance_current(true);
    }
    return model.step_current(true);
}

double reconstruction_delta(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) {
    return std::sqrt((lhs - rhs).squaredNorm() / static_cast<double>(lhs.size()));
}

void write_reconstruction_pair(
    const fs::path& output_path,
    const Eigen::VectorXd& original,
    const Eigen::VectorXd& reconstruction) {
    write_side_by_side_pgm(output_path, original, reconstruction, 28, 28);
}

}  // namespace h4_detail

int run_h4_sequence(const h4_detail::Config& config) {
    Dataset train_images = load_mnist_images(config.data_dir / "train-images-idx3-ubyte", 0);
    Dataset test_images = load_mnist_images(config.data_dir / "t10k-images-idx3-ubyte", 0);
    std::vector<std::uint8_t> train_labels = h4_detail::load_mnist_labels(config.data_dir / "train-labels-idx1-ubyte", 0);
    std::vector<std::uint8_t> test_labels = h4_detail::load_mnist_labels(config.data_dir / "t10k-labels-idx1-ubyte", 0);
    if (train_images.images.size() != train_labels.size() || test_images.images.size() != test_labels.size()) {
        throw std::runtime_error("Image/label count mismatch.");
    }

    std::unordered_map<int, std::vector<std::size_t>> train_by_label;
    std::unordered_map<int, std::vector<std::size_t>> test_by_label;
    for (std::size_t i = 0; i < train_labels.size(); ++i) {
        train_by_label[static_cast<int>(train_labels[i])].push_back(i);
    }
    for (std::size_t i = 0; i < test_labels.size(); ++i) {
        test_by_label[static_cast<int>(test_labels[i])].push_back(i);
    }

    std::unordered_map<int, std::size_t> train_offsets;
    std::vector<h4_detail::SequenceItem> sequence;
    sequence.reserve(config.sequence_length);
    for (std::size_t i = 0; i < config.sequence_length; ++i) {
        const int label = config.sequence_labels[i % config.sequence_labels.size()];
        sequence.push_back(h4_detail::next_item(train_by_label, train_offsets, label));
    }

    std::vector<h4_detail::SequenceItem> probes;
    probes.reserve(config.probe_labels.size());
    for (int label : config.probe_labels) {
        if (test_by_label[label].empty()) {
            throw std::runtime_error("No test probe found for label " + std::to_string(label));
        }
        probes.push_back({test_by_label[label].front(), label});
    }

    LTFNConfig model_config;
    model_config.dims = config.dims;
    model_config.tau_r = config.tau_r;
    model_config.lr_w = config.lr_w;
    model_config.dt_r = config.dt_r;
    model_config.dt_w = config.dt_w;
    model_config.visible_loss = config.visible_loss;

    ensure_directory(config.output_dir);
    for (const std::string strategy : {"zero_init", "warm_start"}) {
        const fs::path strategy_dir = config.output_dir / strategy;
        ensure_directory(strategy_dir);
        ensure_directory(strategy_dir / "train_recons");
        ensure_directory(strategy_dir / "probe_recons");

        std::unique_ptr<ILTFNModel> train_model = create_model(model_config, config.seed, config.backend);
        std::unique_ptr<ILTFNModel> probe_model = create_model(model_config, config.seed, config.backend);
        CheckpointState checkpoint_state;
        std::string checkpoint_error;
        if (!load_checkpoint(config.checkpoint_path, *train_model, checkpoint_state, checkpoint_error)) {
            throw std::runtime_error("Failed to load checkpoint: " + checkpoint_error);
        }
        probe_model->set_weights(train_model->weights());

        std::ofstream csv(strategy_dir / "sequence_metrics.csv", std::ios::out | std::ios::trunc);
        csv << "strategy,seq_pos,train_label,train_index,train_mse";
        for (std::size_t layer = 0; layer + 1 < config.dims.size(); ++layer) {
            csv << ",error_norm_l" << layer;
        }
        for (std::size_t layer = 0; layer + 1 < config.dims.size(); ++layer) {
            csv << ",weight_norm_l" << layer;
        }
        for (int label : config.probe_labels) {
            csv << ",probe_mse_" << label << ",ghost_" << label;
        }
        csv << "\n";

        std::unordered_map<int, Eigen::VectorXd> previous_probe_reconstructions;

        for (std::size_t seq_pos = 0; seq_pos < sequence.size(); ++seq_pos) {
            const auto& item = sequence[seq_pos];
            const Eigen::VectorXd& input = train_images.images[item.index];
            StepDiagnostics diagnostics = strategy == "zero_init"
                ? h4_detail::process_zero_init(*train_model, input, config.steps)
                : h4_detail::process_warm_start(*train_model, input, config.steps, seq_pos == 0);
            const Eigen::VectorXd train_reconstruction = train_model->current_reconstruction();
            const double train_mse = compute_mse(input, train_reconstruction);

            std::ostringstream recon_name;
            recon_name << "seq_" << std::setw(4) << std::setfill('0') << seq_pos
                       << "_label_" << item.label << ".pgm";
            h4_detail::write_reconstruction_pair(strategy_dir / "train_recons" / recon_name.str(), input, train_reconstruction);

            probe_model->set_weights(train_model->weights());
            std::unordered_map<int, double> probe_mse;
            std::unordered_map<int, double> probe_ghost;
            for (const auto& probe : probes) {
                const Eigen::VectorXd& probe_input = test_images.images[probe.index];
                RelaxationResult probe_result = probe_model->reconstruct(probe_input, config.steps, false);
                probe_mse[probe.label] = probe_result.mse;
                if (previous_probe_reconstructions.count(probe.label)) {
                    probe_ghost[probe.label] = h4_detail::reconstruction_delta(
                        probe_result.reconstruction,
                        previous_probe_reconstructions[probe.label]);
                } else {
                    probe_ghost[probe.label] = 0.0;
                }
                previous_probe_reconstructions[probe.label] = probe_result.reconstruction;

                std::ostringstream probe_name;
                probe_name << "seq_" << std::setw(4) << std::setfill('0') << seq_pos
                           << "_probe_" << probe.label << ".pgm";
                h4_detail::write_reconstruction_pair(
                    strategy_dir / "probe_recons" / probe_name.str(),
                    probe_input,
                    probe_result.reconstruction);
            }

            csv << strategy << "," << seq_pos << "," << item.label << "," << item.index << "," << train_mse;
            for (double value : diagnostics.error_norms) {
                csv << "," << value;
            }
            for (double value : diagnostics.weight_norms) {
                csv << "," << value;
            }
            for (int label : config.probe_labels) {
                csv << "," << probe_mse[label] << "," << probe_ghost[label];
            }
            csv << "\n";

            std::cout << strategy << " processed " << (seq_pos + 1) << "/" << sequence.size() << "\n";
        }
    }

    std::ofstream meta(config.output_dir / "metadata.json", std::ios::out | std::ios::trunc);
    meta << "{\n"
         << "  \"checkpoint\": \"" << config.checkpoint_path.generic_string() << "\",\n"
         << "  \"visible_loss\": \"" << visible_loss_to_string(config.visible_loss) << "\",\n"
         << "  \"steps\": " << config.steps << ",\n"
         << "  \"sequence_length\": " << config.sequence_length << "\n"
         << "}\n";

    return 0;
}

}  // namespace ltfn

int main(int argc, char** argv) {
    try {
        ltfn::h4_detail::Config config;
        std::string error_message;
        if (!ltfn::h4_detail::parse_args(argc, argv, config, error_message)) {
            if (!error_message.empty()) {
                std::cerr << "Argument error: " << error_message << "\n\n";
            }
            std::cout << ltfn::h4_detail::usage_text(argv[0]);
            return error_message.empty() ? 0 : 1;
        }
        return ltfn::run_h4_sequence(config);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << '\n';
        return 1;
    }
}
