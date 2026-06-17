#pragma once

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ltfn {
namespace tinylm {

namespace fs = std::filesystem;

struct TinyLMConfig {
    fs::path text_file;
    fs::path output_dir{"runs/tinylm_latest"};
    int layers{2};
    int hidden_dim{128};
    int relax_steps{12};
    int seq_len{64};
    std::size_t max_epochs{1};
    std::size_t max_train_chars{20000};
    std::size_t eval_chars{4096};
    std::size_t eval_interval{5000};
    std::size_t generate_chars{200};
    double dt_state{0.12};
    double lr_readout{2e-3};
    double lr_down{5e-4};
    double lr_value{2e-4};
    double lr_cam_predict{1e-3};
    double lr_query{1e-4};
    double lr_key{5e-5};
    double weight_decay{1e-5};
    double cam_decay{0.98};
    double cam_epsilon{1e-6};
    double cam_readout_scale{0.5};
    double input_precision{2.0};
    double horizontal_precision{0.25};
    double vertical_precision{1.0};
    double output_precision{0.5};
    double precision_beta{0.98};
    double precision_min{0.25};
    double precision_max{4.0};
    double state_rms_limit{1.0};
    int adam_slots{16};
    bool adaptive_precision{true};
    bool vertical_only{false};
    bool learn_value{true};
    bool learn_query{false};
    bool learn_key{false};
    bool cam_context_top_state{false};
    std::uint32_t seed{42U};
    std::string generate_prompt{"The "};
};

struct CharDataset {
    std::string text;
    std::vector<int> tokens;
    std::vector<char> vocab;
    std::array<int, 256> char_to_id{};
};

struct StepStats {
    std::size_t tokens{0};
    double cross_entropy{0.0};
    double bpc{0.0};
    double accuracy{0.0};
    double energy{0.0};
    double input_energy{0.0};
    double horizontal_energy{0.0};
    double vertical_energy{0.0};
    double output_energy{0.0};
    double top_state_norm{0.0};
    double scaled_cam_context_norm{0.0};
    double cam_to_top_ratio{0.0};
    double output_precision{0.0};
    double query_update_norm{0.0};
    double main_cross_entropy{0.0};
    double main_bpc{0.0};
    double main_accuracy{0.0};
    double cam_cross_entropy{0.0};
    double cam_bpc{0.0};
    double cam_accuracy{0.0};
    double cam_state_mse{0.0};
    double cam_state_cosine{0.0};
    double cam_predict_update_norm{0.0};
    double value_update_norm{0.0};
};

class LinearMemory {
public:
    LinearMemory() = default;
    LinearMemory(int dim, int slots, double epsilon);

    void reset();
    Eigen::VectorXd read(const Eigen::VectorXd& query) const;
    int winner(const Eigen::VectorXd& query) const;
    void write(const Eigen::VectorXd& key, const Eigen::VectorXd& value);
    void update_winner_value(int winner, const Eigen::VectorXd& gradient, double lr);
    void update_winner_key(int winner, const Eigen::VectorXd& query, double lr);

    Eigen::VectorXd key_vector(int winner) const;
    const Eigen::MatrixXd& keys() const noexcept;
    const Eigen::MatrixXd& values() const noexcept;
    int slots() const noexcept;
    double epsilon() const noexcept;

private:
    int dim_{0};
    int slots_{0};
    double epsilon_{1e-6};
    Eigen::MatrixXd keys_;
    Eigen::MatrixXd values_;
    Eigen::VectorXd usage_;
};

class LTFNCAMLanguageModel {
public:
    explicit LTFNCAMLanguageModel(const TinyLMConfig& config, int vocab_size);

    void reset_memory();
    StepStats train_sequence(const std::vector<int>& tokens, std::size_t begin, std::size_t end);
    StepStats evaluate_sequence(const std::vector<int>& tokens, std::size_t begin, std::size_t end);
    std::vector<std::vector<int>> trace_slot_winners(
        const std::vector<int>& tokens,
        std::size_t begin,
        std::size_t end) const;
    std::string generate(const CharDataset& dataset, const std::string& prompt, std::size_t count);

private:
    struct InferenceState {
        std::vector<Eigen::VectorXd> states;
        std::vector<Eigen::VectorXd> horizontal_predictions;
        std::vector<Eigen::VectorXd> horizontal_errors;
        std::vector<Eigen::VectorXd> vertical_errors;
        Eigen::VectorXd input_error;
        Eigen::VectorXd logits;
        Eigen::VectorXd probabilities;
        Eigen::VectorXd main_logits;
        Eigen::VectorXd main_probabilities;
        Eigen::VectorXd cam_logits;
        Eigen::VectorXd cam_probabilities;
        Eigen::VectorXd vocab_error;
        Eigen::VectorXd readout_error;
        Eigen::VectorXd cam_error;
        Eigen::VectorXd cam_context;
        Eigen::VectorXd cam_pred_state;
        Eigen::VectorXd cam_state_error;
        double input_energy{0.0};
        double horizontal_energy{0.0};
        double vertical_energy{0.0};
        double output_energy{0.0};
        double main_output_energy{0.0};
        double cam_output_energy{0.0};
        double cam_state_mse{0.0};
        double cam_state_cosine{0.0};
        double top_state_norm{0.0};
        double scaled_cam_context_norm{0.0};
        double cam_to_top_ratio{0.0};
    };

    InferenceState infer(
        int token,
        int target_token,
        const std::vector<LinearMemory>& prefix_memories,
        const std::vector<Eigen::VectorXd>& previous_states,
        bool use_output_error) const;
    std::vector<Eigen::VectorXd> memory_keys(const InferenceState& state) const;
    void write_transition_to_memory(
        const std::vector<Eigen::VectorXd>& keys,
        const InferenceState& value_state);
    void update_readout_and_vertical(const InferenceState& learning_state, int target_token);
    void update_cam_weights(
        const InferenceState& cam_source_state,
        int target_token,
        const std::vector<LinearMemory>& prefix_memories,
        const std::vector<Eigen::VectorXd>& previous_states);
    void update_precisions(const InferenceState& state);
    Eigen::VectorXd token_embedding(int token) const;
    Eigen::VectorXd query_input(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& previous_state) const;
    Eigen::VectorXd query_features(
        int layer,
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& previous_state) const;
    Eigen::VectorXd key_features(int layer, const Eigen::VectorXd& state) const;
    Eigen::VectorXd value_vector(int layer, const Eigen::VectorXd& state) const;
    void normalize_state(Eigen::VectorXd& state) const;
    int argmax_token(const Eigen::VectorXd& probabilities) const;

    TinyLMConfig config_;
    int vocab_size_{0};
    std::vector<LinearMemory> memories_;
    Eigen::MatrixXd input_embeddings_;
    std::vector<Eigen::MatrixXd> down_weights_;
    std::vector<Eigen::MatrixXd> query_weights_;
    std::vector<Eigen::MatrixXd> key_weights_;
    std::vector<Eigen::MatrixXd> value_weights_;
    Eigen::MatrixXd cam_predict_weights_;
    Eigen::MatrixXd vocab_weights_;
    Eigen::VectorXd vocab_bias_;
    std::vector<double> horizontal_precisions_;
    std::vector<double> vertical_precisions_;
    std::vector<double> horizontal_second_moments_;
    std::vector<double> vertical_second_moments_;
    double input_second_moment_{0.0};
    double output_second_moment_{0.0};
    double input_precision_{1.0};
    double output_precision_{1.0};
    double last_query_update_norm_{0.0};
    double last_cam_predict_update_norm_{0.0};
    double last_value_update_norm_{0.0};
};

CharDataset load_char_dataset(const fs::path& path, std::size_t max_chars);
CharDataset make_smoke_dataset(std::size_t max_chars);
StepStats evaluate_ngram(
    const std::vector<int>& train_tokens,
    const std::vector<int>& eval_tokens,
    int vocab_size,
    int order);
bool parse_tinylm_args(int argc, char** argv, TinyLMConfig& config, bool& show_help, std::string& error);
std::string tinylm_usage(const char* program_name);
void run_tinylm(const TinyLMConfig& config, int argc, char** argv);

}  // namespace tinylm
}  // namespace ltfn
