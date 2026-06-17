#include "tinylm.h"

#include <Eigen/QR>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ltfn {
namespace tinylm {
namespace {

constexpr double kLog2 = 0.69314718055994530942;

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

std::string now_utc() {
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

bool parse_bool(const std::string& value, bool& parsed) {
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

void append_arg_value(int& index, int argc, char** argv, std::string& value, const std::string& key) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + key);
    }
    value = argv[++index];
}

double uniform_limit(int fan_in, int fan_out) {
    return std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
}

Eigen::MatrixXd random_uniform_matrix(int rows, int cols, double limit, std::mt19937& generator) {
    std::uniform_real_distribution<double> distribution(-limit, limit);
    Eigen::MatrixXd matrix(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            matrix(row, col) = distribution(generator);
        }
    }
    return matrix;
}

Eigen::MatrixXd random_orthogonal_matrix(int dim, std::mt19937& generator) {
    std::normal_distribution<double> distribution(0.0, 1.0);
    Eigen::MatrixXd matrix(dim, dim);
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            matrix(row, col) = distribution(generator);
        }
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(matrix);
    Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(dim, dim);
    return q;
}

Eigen::VectorXd elu_plus_one(const Eigen::VectorXd& values) {
    return values.unaryExpr([](double value) {
        return value >= 0.0 ? value + 1.0 : std::exp(value);
    });
}

Eigen::VectorXd elu_plus_one_derivative(const Eigen::VectorXd& values) {
    return values.unaryExpr([](double value) {
        return value >= 0.0 ? 1.0 : std::exp(value);
    });
}

Eigen::VectorXd softmax(const Eigen::VectorXd& logits) {
    const double max_value = logits.maxCoeff();
    Eigen::VectorXd exps = (logits.array() - max_value).exp().matrix();
    const double sum = exps.sum();
    if (sum <= 0.0 || !std::isfinite(sum)) {
        return Eigen::VectorXd::Ones(logits.size()) / static_cast<double>(logits.size());
    }
    return exps / sum;
}

void normalize_precision_vector(std::vector<double>& values, double min_value, double max_value) {
    if (values.empty()) {
        return;
    }
    double log_sum = 0.0;
    for (double value : values) {
        log_sum += std::log(std::max(value, 1e-12));
    }
    const double geometric_mean = std::exp(log_sum / static_cast<double>(values.size()));
    for (double& value : values) {
        value = std::clamp(value / geometric_mean, min_value, max_value);
    }
}

std::string bool_json(bool value) {
    return value ? "true" : "false";
}

std::string config_json(const TinyLMConfig& config, int argc, char** argv, std::size_t vocab_size) {
    std::ostringstream command;
    for (int i = 0; i < argc; ++i) {
        if (i > 0) {
            command << ' ';
        }
        command << argv[i];
    }

    std::ostringstream stream;
    stream << "{\n"
           << "  \"timestamp_utc\": \"" << now_utc() << "\",\n"
           << "  \"text_file\": \"" << escape_json(config.text_file.generic_string()) << "\",\n"
           << "  \"output_dir\": \"" << escape_json(config.output_dir.generic_string()) << "\",\n"
           << "  \"vocab_size\": " << vocab_size << ",\n"
           << "  \"layers\": " << config.layers << ",\n"
           << "  \"hidden_dim\": " << config.hidden_dim << ",\n"
           << "  \"relax_steps\": " << config.relax_steps << ",\n"
           << "  \"seq_len\": " << config.seq_len << ",\n"
           << "  \"max_epochs\": " << config.max_epochs << ",\n"
           << "  \"max_train_chars\": " << config.max_train_chars << ",\n"
           << "  \"eval_chars\": " << config.eval_chars << ",\n"
           << "  \"eval_interval\": " << config.eval_interval << ",\n"
           << "  \"dt_state\": " << config.dt_state << ",\n"
           << "  \"lr_readout\": " << config.lr_readout << ",\n"
           << "  \"lr_down\": " << config.lr_down << ",\n"
           << "  \"lr_value\": " << config.lr_value << ",\n"
           << "  \"lr_cam_predict\": " << config.lr_cam_predict << ",\n"
           << "  \"lr_query\": " << config.lr_query << ",\n"
           << "  \"lr_key\": " << config.lr_key << ",\n"
           << "  \"weight_decay\": " << config.weight_decay << ",\n"
           << "  \"cam_decay\": " << config.cam_decay << ",\n"
           << "  \"cam_readout_scale\": " << config.cam_readout_scale << ",\n"
           << "  \"adam_slots\": " << config.adam_slots << ",\n"
           << "  \"adaptive_precision\": " << bool_json(config.adaptive_precision) << ",\n"
           << "  \"vertical_only\": " << bool_json(config.vertical_only) << ",\n"
           << "  \"learn_value\": " << bool_json(config.learn_value) << ",\n"
           << "  \"learn_query\": " << bool_json(config.learn_query) << ",\n"
           << "  \"learn_key\": " << bool_json(config.learn_key) << ",\n"
           << "  \"cam_context_top_state\": " << bool_json(config.cam_context_top_state) << ",\n"
           << "  \"seed\": " << config.seed << ",\n"
           << "  \"generate_prompt\": \"" << escape_json(config.generate_prompt) << "\",\n"
           << "  \"command_line\": \"" << escape_json(command.str()) << "\"\n"
           << "}\n";
    return stream.str();
}

StepStats combine_stats(const StepStats& total, const StepStats& add) {
    StepStats result = total;
    result.tokens += add.tokens;
    result.cross_entropy += add.cross_entropy * static_cast<double>(add.tokens);
    result.accuracy += add.accuracy * static_cast<double>(add.tokens);
    result.energy += add.energy * static_cast<double>(add.tokens);
    result.input_energy += add.input_energy * static_cast<double>(add.tokens);
    result.horizontal_energy += add.horizontal_energy * static_cast<double>(add.tokens);
    result.vertical_energy += add.vertical_energy * static_cast<double>(add.tokens);
    result.output_energy += add.output_energy * static_cast<double>(add.tokens);
    result.top_state_norm += add.top_state_norm * static_cast<double>(add.tokens);
    result.scaled_cam_context_norm += add.scaled_cam_context_norm * static_cast<double>(add.tokens);
    result.cam_to_top_ratio += add.cam_to_top_ratio * static_cast<double>(add.tokens);
    result.output_precision += add.output_precision * static_cast<double>(add.tokens);
    result.query_update_norm += add.query_update_norm * static_cast<double>(add.tokens);
    result.main_cross_entropy += add.main_cross_entropy * static_cast<double>(add.tokens);
    result.main_accuracy += add.main_accuracy * static_cast<double>(add.tokens);
    result.cam_cross_entropy += add.cam_cross_entropy * static_cast<double>(add.tokens);
    result.cam_accuracy += add.cam_accuracy * static_cast<double>(add.tokens);
    result.cam_state_mse += add.cam_state_mse * static_cast<double>(add.tokens);
    result.cam_state_cosine += add.cam_state_cosine * static_cast<double>(add.tokens);
    result.cam_predict_update_norm += add.cam_predict_update_norm * static_cast<double>(add.tokens);
    result.value_update_norm += add.value_update_norm * static_cast<double>(add.tokens);
    return result;
}

StepStats finalize_stats(StepStats stats) {
    if (stats.tokens == 0) {
        return stats;
    }
    const double denom = static_cast<double>(stats.tokens);
    stats.cross_entropy /= denom;
    stats.bpc = stats.cross_entropy / kLog2;
    stats.accuracy /= denom;
    stats.energy /= denom;
    stats.input_energy /= denom;
    stats.horizontal_energy /= denom;
    stats.vertical_energy /= denom;
    stats.output_energy /= denom;
    stats.top_state_norm /= denom;
    stats.scaled_cam_context_norm /= denom;
    stats.cam_to_top_ratio /= denom;
    stats.output_precision /= denom;
    stats.query_update_norm /= denom;
    stats.main_cross_entropy /= denom;
    stats.main_bpc = stats.main_cross_entropy / kLog2;
    stats.main_accuracy /= denom;
    stats.cam_cross_entropy /= denom;
    stats.cam_bpc = stats.cam_cross_entropy / kLog2;
    stats.cam_accuracy /= denom;
    stats.cam_state_mse /= denom;
    stats.cam_state_cosine /= denom;
    stats.cam_predict_update_norm /= denom;
    stats.value_update_norm /= denom;
    return stats;
}

std::string printable_char(char value) {
    if (value == '\n') {
        return "\\n";
    }
    if (value == '\r') {
        return "\\r";
    }
    if (value == '\t') {
        return "\\t";
    }
    if (value == '"') {
        return "\\\"";
    }
    if (value == '\\') {
        return "\\\\";
    }
    return std::string(1, value);
}

}  // namespace

LinearMemory::LinearMemory(int dim, int slots, double epsilon)
    : dim_(dim),
      slots_(std::max(1, slots)),
      epsilon_(epsilon),
      keys_(Eigen::MatrixXd::Zero(dim, std::max(1, slots))),
      values_(Eigen::MatrixXd::Zero(dim, std::max(1, slots))),
      usage_(Eigen::VectorXd::Zero(std::max(1, slots))) {
    for (int slot = 0; slot < slots_; ++slot) {
        for (int row = 0; row < dim_; ++row) {
            const double x = static_cast<double>((row + 1) * (slot + 3));
            keys_(row, slot) = std::sin(12.9898 * x) + 0.5 * std::cos(78.233 * x);
        }
        const double norm = std::max(keys_.col(slot).norm(), epsilon_);
        keys_.col(slot) *= std::sqrt(static_cast<double>(dim_)) / norm;
    }
}

void LinearMemory::reset() {
    usage_.setZero();
}

Eigen::VectorXd LinearMemory::read(const Eigen::VectorXd& query) const {
    return values_.col(winner(query));
}

int LinearMemory::winner(const Eigen::VectorXd& query) const {
    if (slots_ <= 1) {
        return 0;
    }
    const double query_norm = std::max(query.norm(), epsilon_);
    int best_slot = 0;
    double best_similarity = -std::numeric_limits<double>::infinity();
    for (int slot = 0; slot < slots_; ++slot) {
        const double key_norm = std::max(keys_.col(slot).norm(), epsilon_);
        const double similarity = keys_.col(slot).dot(query) / (key_norm * query_norm);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_slot = slot;
        }
    }
    return best_slot;
}

void LinearMemory::write(const Eigen::VectorXd& key, const Eigen::VectorXd& value) {
    (void)key;
    (void)value;
    // ADAM slots are learned parameters. Runtime sequence writes would overwrite
    // credit-assigned slot values during evaluation/generation, so this hook is
    // intentionally inert; slot updates happen in update_cam_weights().
}

void LinearMemory::update_winner_value(int winner_slot, const Eigen::VectorXd& gradient, double lr) {
    if (winner_slot < 0 || winner_slot >= slots_ || gradient.size() != dim_) {
        return;
    }
    values_.col(winner_slot) -= lr * gradient;
    usage_(winner_slot) += 1.0;
}

void LinearMemory::update_winner_key(int winner_slot, const Eigen::VectorXd& query, double lr) {
    if (winner_slot < 0 || winner_slot >= slots_ || query.size() != dim_ || lr <= 0.0) {
        return;
    }
    keys_.col(winner_slot) += lr * (query - keys_.col(winner_slot));
    const double norm = std::max(keys_.col(winner_slot).norm(), epsilon_);
    keys_.col(winner_slot) *= std::sqrt(static_cast<double>(dim_)) / norm;
}

Eigen::VectorXd LinearMemory::key_vector(int winner_slot) const {
    if (winner_slot < 0 || winner_slot >= slots_) {
        return Eigen::VectorXd::Zero(dim_);
    }
    return keys_.col(winner_slot);
}

const Eigen::MatrixXd& LinearMemory::keys() const noexcept {
    return keys_;
}

const Eigen::MatrixXd& LinearMemory::values() const noexcept {
    return values_;
}

int LinearMemory::slots() const noexcept {
    return slots_;
}

double LinearMemory::epsilon() const noexcept {
    return epsilon_;
}

LTFNCAMLanguageModel::LTFNCAMLanguageModel(const TinyLMConfig& config, int vocab_size)
    : config_(config),
      vocab_size_(vocab_size) {
    if (config_.layers < 1) {
        throw std::invalid_argument("--layers must be positive.");
    }
    if (config_.hidden_dim < 2) {
        throw std::invalid_argument("--hidden-dim must be at least 2.");
    }
    if (vocab_size_ < 2) {
        throw std::invalid_argument("Vocabulary must contain at least two characters.");
    }

    std::mt19937 generator(config_.seed);
    const int dim = config_.hidden_dim;

    input_embeddings_ = random_uniform_matrix(dim, vocab_size_, uniform_limit(vocab_size_, dim), generator);
    for (int col = 0; col < input_embeddings_.cols(); ++col) {
        Eigen::VectorXd embedding = input_embeddings_.col(col);
        const double rms = std::sqrt(embedding.squaredNorm() / static_cast<double>(dim));
        if (rms > 0.0) {
            input_embeddings_.col(col) = embedding / rms * 0.5;
        }
    }

    memories_.reserve(static_cast<std::size_t>(config_.layers));
    query_weights_.reserve(static_cast<std::size_t>(config_.layers));
    key_weights_.reserve(static_cast<std::size_t>(config_.layers));
    value_weights_.reserve(static_cast<std::size_t>(config_.layers));
    horizontal_precisions_.assign(static_cast<std::size_t>(config_.layers), config_.horizontal_precision);
    horizontal_second_moments_.assign(static_cast<std::size_t>(config_.layers), 1.0);

    for (int layer = 0; layer < config_.layers; ++layer) {
        memories_.emplace_back(dim, config_.adam_slots, config_.cam_epsilon);
        Eigen::MatrixXd shared_projection = random_orthogonal_matrix(dim, generator);
        Eigen::MatrixXd query_projection = Eigen::MatrixXd::Zero(dim, 2 * dim);
        query_projection.leftCols(dim) = 0.5 * shared_projection;
        query_projection.rightCols(dim) = 0.5 * shared_projection;
        query_weights_.push_back(std::move(query_projection));
        key_weights_.push_back(std::move(shared_projection));
        value_weights_.push_back(Eigen::MatrixXd::Identity(dim, dim));
    }

    const int vertical_count = std::max(0, config_.layers - 1);
    down_weights_.reserve(static_cast<std::size_t>(vertical_count));
    vertical_precisions_.assign(static_cast<std::size_t>(vertical_count), config_.vertical_precision);
    vertical_second_moments_.assign(static_cast<std::size_t>(vertical_count), 1.0);
    for (int layer = 0; layer < vertical_count; ++layer) {
        down_weights_.push_back(random_uniform_matrix(dim, dim, uniform_limit(dim, dim), generator));
    }

    cam_predict_weights_ = random_uniform_matrix(vocab_size_, dim, uniform_limit(dim, vocab_size_), generator);
    vocab_weights_ = random_uniform_matrix(vocab_size_, dim, uniform_limit(dim, vocab_size_), generator);
    vocab_bias_ = Eigen::VectorXd::Zero(vocab_size_);
    input_precision_ = config_.input_precision;
    output_precision_ = config_.output_precision;
}

void LTFNCAMLanguageModel::reset_memory() {
    for (LinearMemory& memory : memories_) {
        memory.reset();
    }
}

Eigen::VectorXd LTFNCAMLanguageModel::token_embedding(int token) const {
    return input_embeddings_.col(token);
}

Eigen::VectorXd LTFNCAMLanguageModel::query_input(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& previous_state) const {
    Eigen::VectorXd input(2 * state.size());
    input.head(state.size()) = state;
    input.tail(state.size()) = state - previous_state;
    return input;
}

Eigen::VectorXd LTFNCAMLanguageModel::query_features(
    int layer,
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& previous_state) const {
    return elu_plus_one(query_weights_[static_cast<std::size_t>(layer)] * query_input(state, previous_state));
}

Eigen::VectorXd LTFNCAMLanguageModel::key_features(int layer, const Eigen::VectorXd& state) const {
    return elu_plus_one(key_weights_[static_cast<std::size_t>(layer)] * state);
}

Eigen::VectorXd LTFNCAMLanguageModel::value_vector(int layer, const Eigen::VectorXd& state) const {
    return value_weights_[static_cast<std::size_t>(layer)] * state;
}

void LTFNCAMLanguageModel::normalize_state(Eigen::VectorXd& state) const {
    if (config_.state_rms_limit <= 0.0 || state.size() == 0) {
        return;
    }
    const double rms = std::sqrt(state.squaredNorm() / static_cast<double>(state.size()));
    if (rms > config_.state_rms_limit) {
        state *= config_.state_rms_limit / rms;
    }
}

int LTFNCAMLanguageModel::argmax_token(const Eigen::VectorXd& probabilities) const {
    Eigen::Index index = 0;
    probabilities.maxCoeff(&index);
    return static_cast<int>(index);
}

LTFNCAMLanguageModel::InferenceState LTFNCAMLanguageModel::infer(
    int token,
    int target_token,
    const std::vector<LinearMemory>& prefix_memories,
    const std::vector<Eigen::VectorXd>& previous_states,
    bool use_output_error) const {
    const int dim = config_.hidden_dim;
    const int layers = config_.layers;
    const Eigen::VectorXd input = token_embedding(token);
    std::vector<Eigen::VectorXd> query_previous_states = previous_states;
    if (query_previous_states.size() != static_cast<std::size_t>(layers)) {
        query_previous_states.assign(static_cast<std::size_t>(layers), Eigen::VectorXd::Zero(dim));
    }

    InferenceState state;
    state.states.assign(static_cast<std::size_t>(layers), Eigen::VectorXd::Zero(dim));
    state.horizontal_predictions.assign(static_cast<std::size_t>(layers), Eigen::VectorXd::Zero(dim));
    state.horizontal_errors.assign(static_cast<std::size_t>(layers), Eigen::VectorXd::Zero(dim));
    state.vertical_errors.assign(static_cast<std::size_t>(std::max(0, layers - 1)), Eigen::VectorXd::Zero(dim));
    state.input_error = Eigen::VectorXd::Zero(dim);
    state.logits = Eigen::VectorXd::Zero(vocab_size_);
    state.probabilities = Eigen::VectorXd::Ones(vocab_size_) / static_cast<double>(vocab_size_);
    state.main_logits = Eigen::VectorXd::Zero(vocab_size_);
    state.main_probabilities = Eigen::VectorXd::Ones(vocab_size_) / static_cast<double>(vocab_size_);
    state.cam_logits = Eigen::VectorXd::Zero(vocab_size_);
    state.cam_probabilities = Eigen::VectorXd::Ones(vocab_size_) / static_cast<double>(vocab_size_);
    state.vocab_error = Eigen::VectorXd::Zero(vocab_size_);
    state.readout_error = Eigen::VectorXd::Zero(dim);
    state.cam_error = Eigen::VectorXd::Zero(vocab_size_);
    state.cam_context = Eigen::VectorXd::Zero(dim);
    state.cam_pred_state = Eigen::VectorXd::Zero(vocab_size_);
    state.cam_state_error = Eigen::VectorXd::Zero(dim);

    state.states[0] = input;

    for (int step = 0; step < config_.relax_steps; ++step) {
        std::vector<Eigen::VectorXd> updates(static_cast<std::size_t>(layers), Eigen::VectorXd::Zero(dim));

        state.input_error = state.states[0] - input;
        updates[0] -= input_precision_ * state.input_error;

        if (!config_.vertical_only) {
            for (int layer = 0; layer < layers; ++layer) {
                const std::size_t index = static_cast<std::size_t>(layer);
                const Eigen::VectorXd query =
                    query_features(layer, state.states[index], query_previous_states[index]);
                const LinearMemory& memory = prefix_memories[index];
                const Eigen::VectorXd raw_prediction = memory.read(query);
                state.horizontal_predictions[index] = raw_prediction;
                state.horizontal_errors[index] = state.states[index] - state.horizontal_predictions[index];
            }
        }

        for (int layer = 0; layer + 1 < layers; ++layer) {
            const Eigen::VectorXd prediction =
                down_weights_[static_cast<std::size_t>(layer)] *
                state.states[static_cast<std::size_t>(layer + 1)];
            state.vertical_errors[static_cast<std::size_t>(layer)] =
                state.states[static_cast<std::size_t>(layer)] - prediction;
            updates[static_cast<std::size_t>(layer)] -=
                vertical_precisions_[static_cast<std::size_t>(layer)] *
                state.vertical_errors[static_cast<std::size_t>(layer)];
            updates[static_cast<std::size_t>(layer + 1)] +=
                vertical_precisions_[static_cast<std::size_t>(layer)] *
                down_weights_[static_cast<std::size_t>(layer)].transpose() *
                state.vertical_errors[static_cast<std::size_t>(layer)];
        }

        state.main_logits = vocab_weights_ * state.states.back() + vocab_bias_;
        state.main_probabilities = softmax(state.main_logits);
        if (use_output_error && target_token >= 0) {
            state.vocab_error = state.main_probabilities;
            state.vocab_error(target_token) -= 1.0;
            state.readout_error = vocab_weights_.transpose() * state.vocab_error;
            updates.back() -= output_precision_ * state.readout_error;
        }

        for (int layer = 0; layer < layers; ++layer) {
            state.states[static_cast<std::size_t>(layer)] +=
                config_.dt_state * updates[static_cast<std::size_t>(layer)];
            normalize_state(state.states[static_cast<std::size_t>(layer)]);
        }
    }

    state.input_error = state.states[0] - input;
    state.input_energy = 0.5 * state.input_error.squaredNorm() / static_cast<double>(dim);
    state.horizontal_energy = 0.0;
    state.vertical_energy = 0.0;

    if (!config_.vertical_only) {
        for (int layer = 0; layer < layers; ++layer) {
            const std::size_t index = static_cast<std::size_t>(layer);
            const Eigen::VectorXd query =
                query_features(layer, state.states[index], query_previous_states[index]);
            const LinearMemory& memory = prefix_memories[index];
            const Eigen::VectorXd raw_prediction = memory.read(query);
            state.horizontal_predictions[index] = raw_prediction;
            state.horizontal_errors[index] = state.states[index] - state.horizontal_predictions[index];
            state.horizontal_energy += 0.5 *
                state.horizontal_errors[index].squaredNorm() /
                static_cast<double>(dim);
        }
    }

    for (int layer = 0; layer + 1 < layers; ++layer) {
        const Eigen::VectorXd prediction =
            down_weights_[static_cast<std::size_t>(layer)] *
            state.states[static_cast<std::size_t>(layer + 1)];
        state.vertical_errors[static_cast<std::size_t>(layer)] =
            state.states[static_cast<std::size_t>(layer)] - prediction;
        state.vertical_energy += 0.5 *
            state.vertical_errors[static_cast<std::size_t>(layer)].squaredNorm() /
            static_cast<double>(dim);
    }

    state.top_state_norm = state.states.back().norm();
    state.cam_context = Eigen::VectorXd::Zero(dim);
    if (!config_.vertical_only && config_.cam_context_top_state) {
        state.cam_context = state.states.back();
    } else if (!config_.vertical_only) {
        for (const Eigen::VectorXd& prediction : state.horizontal_predictions) {
            state.cam_context += prediction;
        }
        state.cam_context /= static_cast<double>(state.horizontal_predictions.size());
    }
    state.cam_logits = cam_predict_weights_ * state.cam_context;
    state.cam_pred_state = state.cam_logits;
    state.scaled_cam_context_norm = (config_.cam_readout_scale * state.cam_logits).norm();
    state.cam_to_top_ratio = state.top_state_norm > 0.0
        ? state.scaled_cam_context_norm / state.top_state_norm
        : 0.0;
    state.main_logits = vocab_weights_ * state.states.back() + vocab_bias_;
    state.main_probabilities = softmax(state.main_logits);
    state.cam_probabilities = softmax(state.cam_logits);
    state.logits = state.main_logits + config_.cam_readout_scale * state.cam_logits;
    state.probabilities = softmax(state.logits);
    if (target_token >= 0) {
        state.vocab_error = state.main_probabilities;
        state.vocab_error(target_token) -= 1.0;
        state.readout_error = vocab_weights_.transpose() * state.vocab_error;
        state.output_energy = -std::log(std::max(state.probabilities(target_token), 1e-12));
        state.main_output_energy = -std::log(std::max(state.main_probabilities(target_token), 1e-12));
        state.cam_output_energy = -std::log(std::max(state.cam_probabilities(target_token), 1e-12));
    }
    return state;
}

std::vector<Eigen::VectorXd> LTFNCAMLanguageModel::memory_keys(const InferenceState& state) const {
    std::vector<Eigen::VectorXd> keys;
    keys.reserve(static_cast<std::size_t>(config_.layers));
    for (int layer = 0; layer < config_.layers; ++layer) {
        keys.push_back(key_features(layer, state.states[static_cast<std::size_t>(layer)]));
    }
    return keys;
}

void LTFNCAMLanguageModel::write_transition_to_memory(
    const std::vector<Eigen::VectorXd>& keys,
    const InferenceState& value_state) {
    (void)keys;
    (void)value_state;
    // ADAM uses persistent accountable slots. Per-token writes are intentionally
    // disabled; observed targets update the responsible slot in update_cam_weights().
}

void LTFNCAMLanguageModel::update_precisions(const InferenceState& state) {
    if (!config_.adaptive_precision) {
        return;
    }

    const double beta = config_.precision_beta;
    const int dim = config_.hidden_dim;
    input_second_moment_ = beta * input_second_moment_ +
        (1.0 - beta) * std::max(state.input_error.squaredNorm() / static_cast<double>(dim), 1e-12);
    input_precision_ = 1.0 / std::sqrt(input_second_moment_ + 1e-8);

    output_second_moment_ = beta * output_second_moment_ +
        (1.0 - beta) * std::max(state.vocab_error.squaredNorm() / static_cast<double>(vocab_size_), 1e-12);
    output_precision_ = 1.0 / std::sqrt(output_second_moment_ + 1e-8);

    std::vector<double> all_precisions;
    all_precisions.push_back(input_precision_);
    all_precisions.push_back(output_precision_);

    for (int layer = 0; layer < config_.layers; ++layer) {
        const std::size_t index = static_cast<std::size_t>(layer);
        const double mean_square = config_.vertical_only
            ? 1.0
            : std::max(state.horizontal_errors[index].squaredNorm() / static_cast<double>(dim), 1e-12);
        horizontal_second_moments_[index] =
            beta * horizontal_second_moments_[index] + (1.0 - beta) * mean_square;
        horizontal_precisions_[index] = 1.0 / std::sqrt(horizontal_second_moments_[index] + 1e-8);
        all_precisions.push_back(horizontal_precisions_[index]);
    }

    for (int layer = 0; layer + 1 < config_.layers; ++layer) {
        const std::size_t index = static_cast<std::size_t>(layer);
        const double mean_square =
            std::max(state.vertical_errors[index].squaredNorm() / static_cast<double>(dim), 1e-12);
        vertical_second_moments_[index] =
            beta * vertical_second_moments_[index] + (1.0 - beta) * mean_square;
        vertical_precisions_[index] = 1.0 / std::sqrt(vertical_second_moments_[index] + 1e-8);
        all_precisions.push_back(vertical_precisions_[index]);
    }

    normalize_precision_vector(all_precisions, config_.precision_min, config_.precision_max);
    std::size_t offset = 0;
    input_precision_ = all_precisions[offset++];
    output_precision_ = all_precisions[offset++];
    for (double& precision : horizontal_precisions_) {
        precision = all_precisions[offset++];
    }
    for (double& precision : vertical_precisions_) {
        precision = all_precisions[offset++];
    }
}

void LTFNCAMLanguageModel::update_readout_and_vertical(
    const InferenceState& learning_state,
    int target_token) {
    Eigen::VectorXd vocab_error = learning_state.main_probabilities;
    vocab_error(target_token) -= 1.0;
    vocab_weights_ -= config_.lr_readout * (vocab_error * learning_state.states.back().transpose());
    vocab_bias_ -= config_.lr_readout * vocab_error;
    if (config_.weight_decay > 0.0) {
        vocab_weights_ *= (1.0 - config_.weight_decay);
    }

    for (int layer = 0; layer + 1 < config_.layers; ++layer) {
        const std::size_t index = static_cast<std::size_t>(layer);
        down_weights_[index] += config_.lr_down * vertical_precisions_[index] *
            (learning_state.vertical_errors[index] * learning_state.states[index + 1].transpose());
        if (config_.weight_decay > 0.0) {
            down_weights_[index] *= (1.0 - config_.weight_decay);
        }
    }
}

void LTFNCAMLanguageModel::update_cam_weights(
    const InferenceState& cam_source_state,
    int target_token,
    const std::vector<LinearMemory>& prefix_memories,
    const std::vector<Eigen::VectorXd>& previous_states) {
    last_query_update_norm_ = 0.0;
    last_cam_predict_update_norm_ = 0.0;
    last_value_update_norm_ = 0.0;
    int query_update_count = 0;
    int value_update_count = 0;

    if (config_.vertical_only) {
        return;
    }

    Eigen::VectorXd cam_error = cam_source_state.cam_probabilities;
    cam_error(target_token) -= 1.0;
    const Eigen::VectorXd context_error = cam_predict_weights_.transpose() * cam_error;
    const Eigen::MatrixXd cam_predict_update =
        config_.lr_cam_predict * (cam_error * cam_source_state.cam_context.transpose());
    cam_predict_weights_ -= cam_predict_update;
    last_cam_predict_update_norm_ = cam_predict_update.norm();
    if (config_.weight_decay > 0.0) {
        cam_predict_weights_ *= (1.0 - config_.weight_decay);
    }

    if (config_.cam_context_top_state) {
        return;
    }

    const double reward = std::clamp(1.0 - cam_error.norm(), -1.0, 1.0);
    for (int layer = 0; layer < config_.layers; ++layer) {
        const std::size_t index = static_cast<std::size_t>(layer);
        const Eigen::VectorXd& layer_state = cam_source_state.states[index];
        const Eigen::VectorXd& previous_state = previous_states.size() == static_cast<std::size_t>(config_.layers)
            ? previous_states[index]
            : layer_state;
        const Eigen::VectorXd local_query_input = query_input(layer_state, previous_state);
        const Eigen::VectorXd layer_context_error =
            context_error / static_cast<double>(config_.layers);
        const Eigen::VectorXd query_pre = query_weights_[index] * local_query_input;
        const Eigen::VectorXd query = elu_plus_one(query_pre);
        const int winning_slot = prefix_memories[index].winner(query);

        if (config_.learn_value) {
            memories_[index].update_winner_value(winning_slot, layer_context_error, config_.lr_value);
            last_value_update_norm_ += config_.lr_value * layer_context_error.norm();
            value_update_count += 1;
        }

        if (config_.learn_query) {
            const Eigen::VectorXd key = prefix_memories[index].key_vector(winning_slot);
            const Eigen::VectorXd desired_query_delta = reward * (key - query);
            const Eigen::VectorXd local_delta =
                desired_query_delta.cwiseProduct(elu_plus_one_derivative(query_pre));
            const Eigen::MatrixXd query_update =
                config_.lr_query * (local_delta * local_query_input.transpose());
            query_weights_[index] += query_update;
            last_query_update_norm_ += query_update.norm();
            query_update_count += 1;
            if (config_.weight_decay > 0.0) {
                query_weights_[index] *= (1.0 - config_.weight_decay);
            }
        }

        if (config_.learn_key) {
            memories_[index].update_winner_key(winning_slot, query, config_.lr_key);
        }
    }

    if (query_update_count > 0) {
        last_query_update_norm_ /= static_cast<double>(query_update_count);
    }
    if (value_update_count > 0) {
        last_value_update_norm_ /= static_cast<double>(value_update_count);
    }
}

StepStats LTFNCAMLanguageModel::train_sequence(
    const std::vector<int>& tokens,
    std::size_t begin,
    std::size_t end) {
    if (end <= begin + 1) {
        return {};
    }
    reset_memory();
    StepStats total;
    std::vector<Eigen::VectorXd> previous_prediction_states(
        static_cast<std::size_t>(config_.layers),
        Eigen::VectorXd::Zero(config_.hidden_dim));

    for (std::size_t pos = begin; pos + 1 < end; ++pos) {
        const int token = tokens[pos];
        const int target = tokens[pos + 1];
        const std::vector<LinearMemory> prefix_memories = memories_;

        InferenceState prediction = infer(token, target, prefix_memories, previous_prediction_states, false);
        const double ce = -std::log(std::max(prediction.probabilities(target), 1e-12));
        const bool correct = argmax_token(prediction.probabilities) == target;

        InferenceState learning = infer(token, target, prefix_memories, previous_prediction_states, true);
        update_precisions(learning);
        update_readout_and_vertical(learning, target);
        update_cam_weights(prediction, target, prefix_memories, previous_prediction_states);
        write_transition_to_memory(memory_keys(prediction), prediction);

        StepStats current;
        current.tokens = 1;
        current.cross_entropy = ce;
        current.accuracy = correct ? 1.0 : 0.0;
        current.main_cross_entropy = prediction.main_output_energy;
        current.main_accuracy = argmax_token(prediction.main_probabilities) == target ? 1.0 : 0.0;
        current.cam_cross_entropy = prediction.cam_output_energy;
        current.cam_accuracy = argmax_token(prediction.cam_probabilities) == target ? 1.0 : 0.0;
        current.input_energy = prediction.input_energy;
        current.horizontal_energy = prediction.horizontal_energy;
        current.vertical_energy = prediction.vertical_energy;
        current.output_energy = ce;
        current.top_state_norm = prediction.top_state_norm;
        current.scaled_cam_context_norm = prediction.scaled_cam_context_norm;
        current.cam_to_top_ratio = prediction.cam_to_top_ratio;
        current.output_precision = output_precision_;
        current.query_update_norm = last_query_update_norm_;
        current.cam_predict_update_norm = last_cam_predict_update_norm_;
        current.value_update_norm = last_value_update_norm_;
        current.energy = current.input_energy + current.horizontal_energy +
            current.vertical_energy + current.output_energy;
        total = combine_stats(total, current);
        previous_prediction_states = prediction.states;
    }

    return finalize_stats(total);
}

StepStats LTFNCAMLanguageModel::evaluate_sequence(
    const std::vector<int>& tokens,
    std::size_t begin,
    std::size_t end) {
    if (end <= begin + 1) {
        return {};
    }
    reset_memory();
    StepStats total;
    std::vector<Eigen::VectorXd> previous_prediction_states(
        static_cast<std::size_t>(config_.layers),
        Eigen::VectorXd::Zero(config_.hidden_dim));

    for (std::size_t pos = begin; pos + 1 < end; ++pos) {
        const int token = tokens[pos];
        const int target = tokens[pos + 1];
        const std::vector<LinearMemory> prefix_memories = memories_;
        InferenceState prediction = infer(token, target, prefix_memories, previous_prediction_states, false);
        const double ce = -std::log(std::max(prediction.probabilities(target), 1e-12));
        const bool correct = argmax_token(prediction.probabilities) == target;
        write_transition_to_memory(memory_keys(prediction), prediction);

        StepStats current;
        current.tokens = 1;
        current.cross_entropy = ce;
        current.accuracy = correct ? 1.0 : 0.0;
        current.main_cross_entropy = prediction.main_output_energy;
        current.main_accuracy = argmax_token(prediction.main_probabilities) == target ? 1.0 : 0.0;
        current.cam_cross_entropy = prediction.cam_output_energy;
        current.cam_accuracy = argmax_token(prediction.cam_probabilities) == target ? 1.0 : 0.0;
        current.input_energy = prediction.input_energy;
        current.horizontal_energy = prediction.horizontal_energy;
        current.vertical_energy = prediction.vertical_energy;
        current.output_energy = ce;
        current.top_state_norm = prediction.top_state_norm;
        current.scaled_cam_context_norm = prediction.scaled_cam_context_norm;
        current.cam_to_top_ratio = prediction.cam_to_top_ratio;
        current.output_precision = output_precision_;
        current.energy = current.input_energy + current.horizontal_energy +
            current.vertical_energy + current.output_energy;
        total = combine_stats(total, current);
        previous_prediction_states = prediction.states;
    }

    return finalize_stats(total);
}

std::vector<std::vector<int>> LTFNCAMLanguageModel::trace_slot_winners(
    const std::vector<int>& tokens,
    std::size_t begin,
    std::size_t end) const {
    std::vector<std::vector<int>> traces;
    if (end <= begin + 1 || config_.vertical_only || config_.cam_context_top_state) {
        return traces;
    }
    traces.reserve(end - begin - 1);
    std::vector<Eigen::VectorXd> previous_prediction_states(
        static_cast<std::size_t>(config_.layers),
        Eigen::VectorXd::Zero(config_.hidden_dim));
    for (std::size_t pos = begin; pos + 1 < end; ++pos) {
        const std::vector<LinearMemory> prefix_memories = memories_;
        InferenceState prediction =
            infer(tokens[pos], tokens[pos + 1], prefix_memories, previous_prediction_states, false);
        std::vector<int> layer_winners;
        layer_winners.reserve(static_cast<std::size_t>(config_.layers));
        for (int layer = 0; layer < config_.layers; ++layer) {
            const std::size_t index = static_cast<std::size_t>(layer);
            const Eigen::VectorXd query =
                query_features(layer, prediction.states[index], previous_prediction_states[index]);
            layer_winners.push_back(prefix_memories[index].winner(query));
        }
        traces.push_back(std::move(layer_winners));
        previous_prediction_states = prediction.states;
    }
    return traces;
}

std::string LTFNCAMLanguageModel::generate(
    const CharDataset& dataset,
    const std::string& prompt,
    std::size_t count) {
    reset_memory();
    std::string output = prompt;
    int current = dataset.char_to_id[static_cast<unsigned char>(' ')] >= 0
        ? dataset.char_to_id[static_cast<unsigned char>(' ')]
        : 0;
    std::vector<Eigen::VectorXd> previous_prediction_states(
        static_cast<std::size_t>(config_.layers),
        Eigen::VectorXd::Zero(config_.hidden_dim));

    for (char ch : prompt) {
        const int id = dataset.char_to_id[static_cast<unsigned char>(ch)];
        if (id >= 0) {
            current = id;
            const std::vector<LinearMemory> prefix_memories = memories_;
            InferenceState prediction = infer(current, -1, prefix_memories, previous_prediction_states, false);
            write_transition_to_memory(memory_keys(prediction), prediction);
            previous_prediction_states = prediction.states;
        }
    }

    for (std::size_t i = 0; i < count; ++i) {
        const std::vector<LinearMemory> prefix_memories = memories_;
        InferenceState prediction = infer(current, -1, prefix_memories, previous_prediction_states, false);
        const int next = argmax_token(prediction.probabilities);
        output.push_back(dataset.vocab[static_cast<std::size_t>(next)]);
        write_transition_to_memory(memory_keys(prediction), prediction);
        previous_prediction_states = prediction.states;
        current = next;
    }
    return output;
}

CharDataset load_char_dataset(const fs::path& path, std::size_t max_chars) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open text file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    CharDataset dataset;
    dataset.text = buffer.str();
    if (max_chars > 0 && dataset.text.size() > max_chars) {
        dataset.text.resize(max_chars);
    }
    if (dataset.text.size() < 2) {
        throw std::runtime_error("Text dataset must contain at least two characters.");
    }

    dataset.char_to_id.fill(-1);
    std::array<bool, 256> present{};
    for (unsigned char ch : dataset.text) {
        present[ch] = true;
    }
    for (int i = 0; i < 256; ++i) {
        if (present[static_cast<std::size_t>(i)]) {
            dataset.char_to_id[static_cast<std::size_t>(i)] = static_cast<int>(dataset.vocab.size());
            dataset.vocab.push_back(static_cast<char>(i));
        }
    }
    dataset.tokens.reserve(dataset.text.size());
    for (unsigned char ch : dataset.text) {
        dataset.tokens.push_back(dataset.char_to_id[ch]);
    }
    return dataset;
}

CharDataset make_smoke_dataset(std::size_t max_chars) {
    const std::string base =
        "The king and the queen walk in the garden.\n"
        "The child can see the bright moon.\n"
        "Love shall make the little bird sing.\n"
        "The dog and the cat run home.\n";
    std::string text;
    const std::size_t target = std::max<std::size_t>(max_chars, 512);
    while (text.size() < target) {
        text += base;
    }
    fs::path empty;
    CharDataset dataset;
    dataset.text = text.substr(0, target);
    dataset.char_to_id.fill(-1);
    std::array<bool, 256> present{};
    for (unsigned char ch : dataset.text) {
        present[ch] = true;
    }
    for (int i = 0; i < 256; ++i) {
        if (present[static_cast<std::size_t>(i)]) {
            dataset.char_to_id[static_cast<std::size_t>(i)] = static_cast<int>(dataset.vocab.size());
            dataset.vocab.push_back(static_cast<char>(i));
        }
    }
    for (unsigned char ch : dataset.text) {
        dataset.tokens.push_back(dataset.char_to_id[ch]);
    }
    return dataset;
}

StepStats evaluate_ngram(
    const std::vector<int>& train_tokens,
    const std::vector<int>& eval_tokens,
    int vocab_size,
    int order) {
    if (order < 1 || train_tokens.size() < static_cast<std::size_t>(order) || eval_tokens.size() < 2) {
        return {};
    }

    const int context_order = order - 1;
    std::unordered_map<std::string, std::vector<std::size_t>> counts;
    std::unordered_map<std::string, std::size_t> totals;
    const auto key_for = [&](const std::vector<int>& tokens, std::size_t pos) {
        std::ostringstream key;
        for (int back = context_order; back > 0; --back) {
            if (pos < static_cast<std::size_t>(back)) {
                key << -1;
            } else {
                key << tokens[pos - static_cast<std::size_t>(back)];
            }
            key << ',';
        }
        return key.str();
    };

    for (std::size_t pos = 1; pos < train_tokens.size(); ++pos) {
        const std::string key = key_for(train_tokens, pos);
        auto& row = counts[key];
        if (row.empty()) {
            row.assign(static_cast<std::size_t>(vocab_size), 0);
        }
        row[static_cast<std::size_t>(train_tokens[pos])] += 1;
        totals[key] += 1;
    }

    StepStats stats;
    for (std::size_t pos = 1; pos < eval_tokens.size(); ++pos) {
        const std::string key = key_for(eval_tokens, pos);
        const auto count_it = counts.find(key);
        const auto total_it = totals.find(key);
        const int target = eval_tokens[pos];
        double probability = 1.0 / static_cast<double>(vocab_size);
        int predicted = 0;
        if (count_it != counts.end() && total_it != totals.end()) {
            const std::vector<std::size_t>& row = count_it->second;
            const double denom = static_cast<double>(total_it->second + static_cast<std::size_t>(vocab_size));
            probability = static_cast<double>(row[static_cast<std::size_t>(target)] + 1U) / denom;
            predicted = static_cast<int>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
        }
        stats.tokens += 1;
        stats.cross_entropy += -std::log(std::max(probability, 1e-12));
        stats.accuracy += predicted == target ? 1.0 : 0.0;
    }
    return finalize_stats(stats);
}

bool parse_tinylm_args(int argc, char** argv, TinyLMConfig& config, bool& show_help, std::string& error) {
    show_help = false;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            std::string value;
            if (key == "--help" || key == "-h") {
                show_help = true;
                return true;
            } else if (key == "--text-file") {
                append_arg_value(i, argc, argv, value, key);
                config.text_file = value;
            } else if (key == "--output-dir") {
                append_arg_value(i, argc, argv, value, key);
                config.output_dir = value;
            } else if (key == "--layers") {
                append_arg_value(i, argc, argv, value, key);
                config.layers = std::stoi(value);
            } else if (key == "--hidden-dim") {
                append_arg_value(i, argc, argv, value, key);
                config.hidden_dim = std::stoi(value);
            } else if (key == "--steps") {
                append_arg_value(i, argc, argv, value, key);
                config.relax_steps = std::stoi(value);
            } else if (key == "--seq-len") {
                append_arg_value(i, argc, argv, value, key);
                config.seq_len = std::stoi(value);
            } else if (key == "--max-epochs") {
                append_arg_value(i, argc, argv, value, key);
                config.max_epochs = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--max-train-chars") {
                append_arg_value(i, argc, argv, value, key);
                config.max_train_chars = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--eval-chars") {
                append_arg_value(i, argc, argv, value, key);
                config.eval_chars = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--eval-interval") {
                append_arg_value(i, argc, argv, value, key);
                config.eval_interval = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--generate-chars") {
                append_arg_value(i, argc, argv, value, key);
                config.generate_chars = static_cast<std::size_t>(std::stoull(value));
            } else if (key == "--dt-state") {
                append_arg_value(i, argc, argv, value, key);
                config.dt_state = std::stod(value);
            } else if (key == "--lr-readout") {
                append_arg_value(i, argc, argv, value, key);
                config.lr_readout = std::stod(value);
            } else if (key == "--lr-down") {
                append_arg_value(i, argc, argv, value, key);
                config.lr_down = std::stod(value);
            } else if (key == "--lr-value") {
                append_arg_value(i, argc, argv, value, key);
                config.lr_value = std::stod(value);
            } else if (key == "--lr-cam-predict") {
                append_arg_value(i, argc, argv, value, key);
                config.lr_cam_predict = std::stod(value);
            } else if (key == "--lr-query") {
                append_arg_value(i, argc, argv, value, key);
                config.lr_query = std::stod(value);
            } else if (key == "--lr-key") {
                append_arg_value(i, argc, argv, value, key);
                config.lr_key = std::stod(value);
            } else if (key == "--weight-decay") {
                append_arg_value(i, argc, argv, value, key);
                config.weight_decay = std::stod(value);
            } else if (key == "--cam-decay") {
                append_arg_value(i, argc, argv, value, key);
                config.cam_decay = std::stod(value);
            } else if (key == "--cam-eps") {
                append_arg_value(i, argc, argv, value, key);
                config.cam_epsilon = std::stod(value);
            } else if (key == "--cam-readout-scale") {
                append_arg_value(i, argc, argv, value, key);
                config.cam_readout_scale = std::stod(value);
            } else if (key == "--input-precision") {
                append_arg_value(i, argc, argv, value, key);
                config.input_precision = std::stod(value);
            } else if (key == "--horizontal-precision") {
                append_arg_value(i, argc, argv, value, key);
                config.horizontal_precision = std::stod(value);
            } else if (key == "--vertical-precision") {
                append_arg_value(i, argc, argv, value, key);
                config.vertical_precision = std::stod(value);
            } else if (key == "--output-precision") {
                append_arg_value(i, argc, argv, value, key);
                config.output_precision = std::stod(value);
            } else if (key == "--precision-beta") {
                append_arg_value(i, argc, argv, value, key);
                config.precision_beta = std::stod(value);
            } else if (key == "--precision-min") {
                append_arg_value(i, argc, argv, value, key);
                config.precision_min = std::stod(value);
            } else if (key == "--precision-max") {
                append_arg_value(i, argc, argv, value, key);
                config.precision_max = std::stod(value);
            } else if (key == "--state-rms-limit") {
                append_arg_value(i, argc, argv, value, key);
                config.state_rms_limit = std::stod(value);
            } else if (key == "--adam-slots") {
                append_arg_value(i, argc, argv, value, key);
                config.adam_slots = std::stoi(value);
            } else if (key == "--adaptive-precision") {
                append_arg_value(i, argc, argv, value, key);
                if (!parse_bool(value, config.adaptive_precision)) {
                    throw std::invalid_argument("Expected boolean for --adaptive-precision.");
                }
            } else if (key == "--vertical-only") {
                append_arg_value(i, argc, argv, value, key);
                if (!parse_bool(value, config.vertical_only)) {
                    throw std::invalid_argument("Expected boolean for --vertical-only.");
                }
            } else if (key == "--learn-value") {
                append_arg_value(i, argc, argv, value, key);
                if (!parse_bool(value, config.learn_value)) {
                    throw std::invalid_argument("Expected boolean for --learn-value.");
                }
            } else if (key == "--learn-q") {
                append_arg_value(i, argc, argv, value, key);
                if (!parse_bool(value, config.learn_query)) {
                    throw std::invalid_argument("Expected boolean for --learn-q.");
                }
            } else if (key == "--learn-k") {
                append_arg_value(i, argc, argv, value, key);
                if (!parse_bool(value, config.learn_key)) {
                    throw std::invalid_argument("Expected boolean for --learn-k.");
                }
            } else if (key == "--cam-context-top-state") {
                append_arg_value(i, argc, argv, value, key);
                if (!parse_bool(value, config.cam_context_top_state)) {
                    throw std::invalid_argument("Expected boolean for --cam-context-top-state.");
                }
            } else if (key == "--seed") {
                append_arg_value(i, argc, argv, value, key);
                config.seed = static_cast<std::uint32_t>(std::stoul(value));
            } else if (key == "--generate-prompt") {
                append_arg_value(i, argc, argv, value, key);
                config.generate_prompt = value;
            } else {
                throw std::invalid_argument("Unknown argument: " + key);
            }
        }

        if (config.layers < 1) {
            throw std::invalid_argument("--layers must be positive.");
        }
        if (config.hidden_dim < 2) {
            throw std::invalid_argument("--hidden-dim must be at least 2.");
        }
        if (config.relax_steps < 1) {
            throw std::invalid_argument("--steps must be positive.");
        }
        if (config.seq_len < 2) {
            throw std::invalid_argument("--seq-len must be at least 2.");
        }
        if (config.cam_decay < 0.0 || config.cam_decay > 1.0) {
            throw std::invalid_argument("--cam-decay must be in [0, 1].");
        }
        if (config.cam_epsilon <= 0.0) {
            throw std::invalid_argument("--cam-eps must be positive.");
        }
        if (config.adam_slots < 1) {
            throw std::invalid_argument("--adam-slots must be positive.");
        }
        if (config.precision_beta < 0.0 || config.precision_beta >= 1.0) {
            throw std::invalid_argument("--precision-beta must be in [0, 1).");
        }
        if (config.precision_min <= 0.0 || config.precision_max < config.precision_min) {
            throw std::invalid_argument("Precision bounds are invalid.");
        }
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }
}

std::string tinylm_usage(const char* program_name) {
    std::ostringstream stream;
    stream
        << "Usage: " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --text-file PATH              Text corpus path; omitted uses built-in smoke corpus\n"
        << "  --output-dir PATH             Output directory (default: runs/tinylm_latest)\n"
        << "  --layers INT                  Hidden layers (default: 2)\n"
        << "  --hidden-dim INT              Hidden width (default: 128)\n"
        << "  --steps INT                   Relaxation steps per token (default: 12)\n"
        << "  --seq-len INT                 Memory reset window length (default: 64)\n"
        << "  --max-epochs INT              Training epochs (default: 1)\n"
        << "  --max-train-chars INT         Corpus cap before split (default: 20000)\n"
        << "  --eval-chars INT              Evaluation suffix length (default: 4096)\n"
        << "  --eval-interval INT           Evaluate every N training characters (default: 5000)\n"
        << "  --vertical-only true|false    Disable CAM horizontal memory (default: false)\n"
        << "  --learn-value true|false      Enable value projection local updates (default: true)\n"
        << "  --learn-q true|false          Enable local query updates (default: false)\n"
        << "  --learn-k true|false          Enable local key alignment updates (default: false)\n"
        << "  --adam-slots INT              Hard-addressed ADAM memory slots per layer (default: 16)\n"
        << "  --cam-context-top-state true|false  Use current top state as CAM context (default: false)\n"
        << "  --adaptive-precision true|false  Enable error precision balancing (default: true)\n"
        << "  --lr-readout FLOAT            Local softmax readout lr (default: 2e-3)\n"
        << "  --lr-down FLOAT               Vertical generative lr (default: 5e-4)\n"
        << "  --lr-value FLOAT              Value projection lr (default: 2e-4)\n"
        << "  --lr-cam-predict FLOAT        CAM embedding prediction lr (default: 1e-3)\n"
        << "  --lr-query FLOAT              Query projection lr (default: 1e-4)\n"
        << "  --lr-key FLOAT                Key projection lr (default: 5e-5)\n"
        << "  --horizontal-precision FLOAT  Initial CAM precision (default: 0.25)\n"
        << "  --cam-decay FLOAT             CAM decay in [0,1] (default: 0.98)\n"
        << "  --cam-readout-scale FLOAT     CAM context added to top readout (default: 0.5)\n"
        << "  --generate-prompt TEXT        Prompt for greedy generation (default: \"The \")\n"
        << "  --generate-chars INT          Characters to generate after training (default: 200)\n"
        << "  --seed INT                    Random seed (default: 42)\n"
        << "  --help                        Show this help\n";
    return stream.str();
}

void run_tinylm(const TinyLMConfig& config, int argc, char** argv) {
    fs::create_directories(config.output_dir);
    CharDataset dataset = config.text_file.empty()
        ? make_smoke_dataset(config.max_train_chars + config.eval_chars)
        : load_char_dataset(config.text_file, config.max_train_chars + config.eval_chars);
    const std::size_t eval_len = std::min(config.eval_chars, dataset.tokens.size() / 3);
    const std::size_t train_end = dataset.tokens.size() - eval_len;
    if (train_end <= static_cast<std::size_t>(config.seq_len) || eval_len < 2) {
        throw std::runtime_error("Dataset split is too small for the configured sequence length.");
    }

    std::ofstream config_file(config.output_dir / "config.json", std::ios::out | std::ios::trunc);
    config_file << config_json(config, argc, argv, dataset.vocab.size());

    std::ofstream vocab_file(config.output_dir / "vocab.txt", std::ios::out | std::ios::trunc);
    for (std::size_t i = 0; i < dataset.vocab.size(); ++i) {
        vocab_file << i << "\t" << printable_char(dataset.vocab[i]) << "\n";
    }

    std::ofstream metrics(config.output_dir / "metrics.csv", std::ios::out | std::ios::trunc);
    metrics << "timestamp_utc,epoch,chars_seen,train_ce,train_bpc,train_acc,eval_ce,eval_bpc,eval_acc,"
               "eval_main_ce,eval_main_bpc,eval_main_acc,eval_cam_ce,eval_cam_bpc,eval_cam_acc,"
               "energy,input_energy,horizontal_energy,vertical_energy,output_energy,"
               "top_state_norm,scaled_cam_context_norm,cam_to_top_ratio,output_precision,"
               "cam_state_mse,cam_state_cosine,train_query_update_norm,"
               "train_cam_predict_update_norm,train_value_update_norm\n";

    LTFNCAMLanguageModel model(config, static_cast<int>(dataset.vocab.size()));
    const std::vector<int> train_tokens(dataset.tokens.begin(), dataset.tokens.begin() +
        static_cast<std::ptrdiff_t>(train_end));
    const std::vector<int> eval_tokens(dataset.tokens.begin() + static_cast<std::ptrdiff_t>(train_end),
        dataset.tokens.end());
    const StepStats bigram = evaluate_ngram(train_tokens, eval_tokens, static_cast<int>(dataset.vocab.size()), 2);
    const StepStats trigram = evaluate_ngram(train_tokens, eval_tokens, static_cast<int>(dataset.vocab.size()), 3);

    std::ofstream events(config.output_dir / "events.jsonl", std::ios::out | std::ios::trunc);
    events << "{\"timestamp_utc\":\"" << now_utc()
           << "\",\"type\":\"run_started\",\"payload\":{\"chars\":" << dataset.tokens.size()
           << ",\"train_chars\":" << train_end
           << ",\"eval_chars\":" << eval_len
           << ",\"vocab_size\":" << dataset.vocab.size()
           << ",\"bigram_bpc\":" << bigram.bpc
           << ",\"trigram_bpc\":" << trigram.bpc << "}}\n";

    std::cout << "TinyLM chars=" << dataset.tokens.size()
              << " train=" << train_end
              << " eval=" << eval_len
              << " vocab=" << dataset.vocab.size()
              << " bigram_bpc=" << std::fixed << std::setprecision(4) << bigram.bpc
              << " trigram_bpc=" << trigram.bpc << "\n";

    std::size_t chars_seen = 0;
    StepStats train_window;

    const auto evaluate_and_log = [&](std::size_t epoch, const StepStats& train_stats) {
        StepStats eval_total;
        for (std::size_t begin = 0; begin + 1 < eval_tokens.size(); begin += static_cast<std::size_t>(config.seq_len)) {
            const std::size_t end = std::min(eval_tokens.size(), begin + static_cast<std::size_t>(config.seq_len));
            eval_total = combine_stats(eval_total, model.evaluate_sequence(eval_tokens, begin, end));
        }
        StepStats eval_stats = finalize_stats(eval_total);
        metrics << now_utc() << ','
                << epoch << ','
                << chars_seen << ','
                << train_stats.cross_entropy << ','
                << train_stats.bpc << ','
                << train_stats.accuracy << ','
                << eval_stats.cross_entropy << ','
                << eval_stats.bpc << ','
                << eval_stats.accuracy << ','
                << eval_stats.main_cross_entropy << ','
                << eval_stats.main_bpc << ','
                << eval_stats.main_accuracy << ','
                << eval_stats.cam_cross_entropy << ','
                << eval_stats.cam_bpc << ','
                << eval_stats.cam_accuracy << ','
                << eval_stats.energy << ','
                << eval_stats.input_energy << ','
                << eval_stats.horizontal_energy << ','
                << eval_stats.vertical_energy << ','
                << eval_stats.output_energy << ','
                << eval_stats.top_state_norm << ','
                << eval_stats.scaled_cam_context_norm << ','
                << eval_stats.cam_to_top_ratio << ','
                << eval_stats.output_precision << ','
                << eval_stats.cam_state_mse << ','
                << eval_stats.cam_state_cosine << ','
                << train_stats.query_update_norm << ','
                << train_stats.cam_predict_update_norm << ','
                << train_stats.value_update_norm << '\n';
        metrics.flush();

        std::cout << "[epoch " << epoch
                  << "] chars=" << chars_seen
                  << " train_bpc=" << std::fixed << std::setprecision(4) << train_stats.bpc
                  << " eval_bpc=" << eval_stats.bpc
                  << " main_bpc=" << eval_stats.main_bpc
                  << " cam_bpc=" << eval_stats.cam_bpc
                  << " eval_acc=" << eval_stats.accuracy
                  << " cam_acc=" << eval_stats.cam_accuracy
                  << " cam_cos=" << eval_stats.cam_state_cosine
                  << " energy=" << eval_stats.energy
                  << " h=" << eval_stats.horizontal_energy
                  << " v=" << eval_stats.vertical_energy
                  << " top_norm=" << eval_stats.top_state_norm
                  << " cam_norm=" << eval_stats.scaled_cam_context_norm
                  << " cam/top=" << eval_stats.cam_to_top_ratio
                  << " out_prec=" << eval_stats.output_precision
                  << " q_upd=" << train_stats.query_update_norm
                  << " cam_upd=" << train_stats.cam_predict_update_norm
                  << " v_upd=" << train_stats.value_update_norm << "\n";
    };

    for (std::size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
        for (std::size_t begin = 0; begin + 1 < train_end; begin += static_cast<std::size_t>(config.seq_len)) {
            const std::size_t end = std::min(train_end, begin + static_cast<std::size_t>(config.seq_len));
            StepStats stats = model.train_sequence(dataset.tokens, begin, end);
            train_window = combine_stats(train_window, stats);
            chars_seen += stats.tokens;

            if (config.eval_interval != 0 && chars_seen >= config.eval_interval &&
                chars_seen % config.eval_interval < static_cast<std::size_t>(config.seq_len)) {
                evaluate_and_log(epoch, finalize_stats(train_window));
                train_window = {};
            }
        }
    }

    if (train_window.tokens > 0) {
        evaluate_and_log(config.max_epochs, finalize_stats(train_window));
    }

    const std::vector<std::vector<int>> slot_traces =
        model.trace_slot_winners(eval_tokens, 0, eval_tokens.size());
    std::ofstream slot_trace_file(config.output_dir / "slot_trace.csv", std::ios::out | std::ios::trunc);
    slot_trace_file << "pos,current_token,current_char,target_token,target_char";
    for (int layer = 0; layer < config.layers; ++layer) {
        slot_trace_file << ",layer" << layer << "_slot";
    }
    slot_trace_file << "\n";
    for (std::size_t pos = 0; pos < slot_traces.size(); ++pos) {
        const int current = eval_tokens[pos];
        const int target = eval_tokens[pos + 1];
        slot_trace_file << pos << ','
                        << current << ",\"" << printable_char(dataset.vocab[static_cast<std::size_t>(current)]) << "\","
                        << target << ",\"" << printable_char(dataset.vocab[static_cast<std::size_t>(target)]) << "\"";
        for (int slot : slot_traces[pos]) {
            slot_trace_file << ',' << slot;
        }
        slot_trace_file << "\n";
    }

    const std::string generated = model.generate(dataset, config.generate_prompt, config.generate_chars);
    std::ofstream sample_file(config.output_dir / "generated.txt", std::ios::out | std::ios::trunc);
    sample_file << generated << "\n";
    events << "{\"timestamp_utc\":\"" << now_utc()
           << "\",\"type\":\"run_finished\",\"payload\":{\"generated_file\":\"generated.txt\"}}\n";
    std::cout << "Generated sample written to " << (config.output_dir / "generated.txt").string() << "\n";
}

}  // namespace tinylm
}  // namespace ltfn
