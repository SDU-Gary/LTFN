#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ltfn {

enum class ComputeBackend {
    Cpu,
    Cuda
};

enum class VisibleLoss {
    Mse,
    Bce
};

enum class StateInit {
    Zero,
    Tied
};

struct LTFNConfig {
    std::vector<int> dims{784, 256, 64, 32};
    double tau_r{0.1};
    double lr_w{1e-5};
    double dt_r{0.1};
    double dt_w{1.0};
    VisibleLoss visible_loss{VisibleLoss::Mse};
    bool use_biases{false};
    bool visible_unit_precision{false};
    StateInit state_init{StateInit::Zero};
    double error_precision_beta{0.0};
    double error_precision_epsilon{1e-4};
    double error_precision_min{0.25};
    double error_precision_max{4.0};
    double transient_gate_tau{0.0};
    bool sequential_inference{false};
    double layer_adapt_beta{0.0};
    double layer_adapt_epsilon{1e-8};
    double decorrelation_lambda{0.0};
};

struct StepDiagnostics {
    double energy{0.0};
    double mse{0.0};
    std::vector<double> error_norms;
    std::vector<double> state_update_norms;
    std::vector<double> weight_gradient_norms;
    std::vector<double> weight_update_norms;
    std::vector<double> weight_norms;
};

struct RelaxationResult {
    Eigen::VectorXd reconstruction;
    double final_energy{0.0};
    double mse{0.0};
    std::vector<double> final_error_norms;
    std::vector<double> final_weight_gradient_norms;
    std::vector<double> energy_trace;
};

struct LoggedRelaxationStep {
    int relax_step{0};
    StepDiagnostics diagnostics;
};

struct BatchTrainOptions {
    std::vector<int> logged_relax_steps;
    bool update_weights{true};
    bool capture_final_gradients{false};
    bool capture_final_states{false};
};

struct BatchTrainResult {
    std::size_t batch_size{0};
    double average_mse{0.0};
    StepDiagnostics final_diagnostics;
    std::vector<Eigen::MatrixXd> final_weight_gradients;
    std::vector<Eigen::MatrixXd> final_batch_states;
    std::vector<LoggedRelaxationStep> logged_steps;
};

class ILTFNModel {
public:
    virtual ~ILTFNModel() = default;

    virtual const LTFNConfig& config() const noexcept = 0;
    virtual const std::vector<Eigen::MatrixXd>& weights() const = 0;
    virtual const std::vector<Eigen::VectorXd>& biases() const = 0;
    virtual void set_weights(const std::vector<Eigen::MatrixXd>& new_weights) = 0;
    virtual void set_biases(const std::vector<Eigen::VectorXd>& new_biases) = 0;
    virtual void reset_states(const Eigen::VectorXd& input) = 0;
    virtual void advance(const Eigen::VectorXd& input, bool update_weights) = 0;
    virtual void advance_current(bool update_weights) = 0;

    virtual StepDiagnostics step(const Eigen::VectorXd& input, bool update_weights) = 0;
    virtual StepDiagnostics step_current(bool update_weights) = 0;
    virtual StepDiagnostics current_diagnostics() const = 0;
    virtual BatchTrainResult train_batch(
        const std::vector<const Eigen::VectorXd*>& inputs,
        int steps,
        const BatchTrainOptions& options) = 0;
    virtual void set_learning_rate(double learning_rate) = 0;
    virtual void set_weight_momentum(double momentum_beta) = 0;
    virtual RelaxationResult relax(const Eigen::VectorXd& input, int steps, bool capture_trace = false) = 0;
    virtual RelaxationResult reconstruct(const Eigen::VectorXd& input, int steps, bool capture_trace = false) = 0;

    virtual double current_energy() const noexcept = 0;
    virtual Eigen::VectorXd current_reconstruction() const = 0;
    virtual const std::vector<Eigen::VectorXd>& states() const = 0;
};

double compute_mse(const Eigen::VectorXd& expected, const Eigen::VectorXd& actual);
std::string visible_loss_to_string(VisibleLoss visible_loss);
std::string state_init_to_string(StateInit state_init);
std::string backend_to_string(ComputeBackend backend);
bool is_cuda_backend_compiled() noexcept;
std::unique_ptr<ILTFNModel> create_model(const LTFNConfig& config, std::uint32_t seed, ComputeBackend backend);

class LTFN final : public ILTFNModel {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit LTFN(const LTFNConfig& config, std::uint32_t seed = 42U);

    const LTFNConfig& config() const noexcept override;
    const std::vector<Eigen::MatrixXd>& weights() const override;
    const std::vector<Eigen::VectorXd>& biases() const override;
    std::vector<Eigen::MatrixXd>& mutable_weights() noexcept;

    void set_weights(const std::vector<Eigen::MatrixXd>& new_weights) override;
    void set_biases(const std::vector<Eigen::VectorXd>& new_biases) override;
    void reset_states(const Eigen::VectorXd& input) override;
    void advance(const Eigen::VectorXd& input, bool update_weights) override;
    void advance_current(bool update_weights) override;

    StepDiagnostics step(const Eigen::VectorXd& input, bool update_weights) override;
    StepDiagnostics step_current(bool update_weights) override;
    StepDiagnostics current_diagnostics() const override;
    BatchTrainResult train_batch(
        const std::vector<const Eigen::VectorXd*>& inputs,
        int steps,
        const BatchTrainOptions& options) override;
    void set_learning_rate(double learning_rate) override;
    void set_weight_momentum(double momentum_beta) override;
    RelaxationResult relax(const Eigen::VectorXd& input, int steps, bool capture_trace = false) override;
    RelaxationResult reconstruct(const Eigen::VectorXd& input, int steps, bool capture_trace = false) override;

    double current_energy() const noexcept override;
    Eigen::VectorXd current_reconstruction() const override;
    const std::vector<Eigen::VectorXd>& states() const override;

    static double compute_mse(const Eigen::VectorXd& expected, const Eigen::VectorXd& actual);

private:
    void validate_dims() const;
    void initialize_latent_states();
    void compute_predictions_and_errors() const;
    void compute_layer_prediction_error(std::size_t layer) const;
    void ensure_predictions_current() const;
    void ensure_input_shape(const Eigen::VectorXd& input) const;
    void update_error_precisions_from_errors();

    LTFNConfig config_;
    std::vector<Eigen::MatrixXd> weights_;
    std::vector<Eigen::VectorXd> states_;
    mutable std::vector<Eigen::VectorXd> predictions_;
    mutable std::vector<Eigen::VectorXd> errors_;
    mutable std::vector<Eigen::VectorXd> pre_activations_;
    mutable std::vector<Eigen::VectorXd> sigmoid_derivatives_;
    std::vector<Eigen::VectorXd> biases_;
    std::vector<Eigen::MatrixXd> weight_velocities_;
    std::vector<Eigen::VectorXd> bias_velocities_;
    std::vector<double> layer_second_moments_;
    std::vector<double> layer_error_second_moments_;
    std::vector<double> layer_error_precisions_;
    Eigen::VectorXd visible_error_second_moments_;
    Eigen::VectorXd visible_error_precisions_;
    double current_learning_rate_{0.0};
    double momentum_beta_{0.0};
    mutable bool predictions_dirty_{false};
};

}  // namespace ltfn
