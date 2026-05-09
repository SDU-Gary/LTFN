#pragma once

#include "ltfn.h"

#if defined(LTFN_USE_CUDA_BACKEND)

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace ltfn {

class LTFNCuda final : public ILTFNModel {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit LTFNCuda(const LTFNConfig& config, std::uint32_t seed = 42U);
    ~LTFNCuda() override;

    LTFNCuda(const LTFNCuda&) = delete;
    LTFNCuda& operator=(const LTFNCuda&) = delete;

    const LTFNConfig& config() const noexcept override;
    const std::vector<Eigen::MatrixXd>& weights() const override;
    void set_weights(const std::vector<Eigen::MatrixXd>& new_weights) override;
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

private:
    void validate_dims() const;
    void ensure_input_shape(const Eigen::VectorXd& input) const;
    void ensure_predictions_current() const;
    void ensure_batch_capacity(int batch_size);
    void allocate_buffers();
    void allocate_batch_buffers(int batch_size);
    void release_buffers() noexcept;
    void release_batch_buffers() noexcept;
    void compute_predictions_errors_and_deltas() const;
    StepDiagnostics collect_diagnostics(bool update_weights);
    void upload_batch_inputs(const std::vector<const Eigen::VectorXd*>& inputs);
    void compute_batch_errors_and_deltas(int batch_size);
    double compute_batch_effective_gradient(std::size_t layer_index, int batch_size);
    StepDiagnostics collect_batch_diagnostics(
        int batch_size,
        const std::vector<double>* state_update_norms = nullptr,
        const std::vector<double>* gradient_norms = nullptr,
        const std::vector<double>* update_norms = nullptr);
    void upload_all_weights();
    void download_all_weights() const;
    void download_all_states() const;
    void download_reconstruction() const;
    void zero_weight_velocities();
    void zero_latent_states();
    void synchronize() const;

    LTFNConfig config_;
    cublasHandle_t cublas_{nullptr};

    mutable std::vector<Eigen::MatrixXd> host_weights_;
    mutable std::vector<Eigen::VectorXd> host_states_;
    mutable Eigen::VectorXd host_reconstruction_;
    mutable Eigen::MatrixXd host_batch_visible_;
    mutable bool predictions_dirty_{false};

    std::vector<double*> device_weights_;
    std::vector<double*> device_states_;
    std::vector<double*> device_predictions_;
    std::vector<double*> device_errors_;
    std::vector<double*> device_pre_activations_;
    std::vector<double*> device_sigmoid_derivatives_;
    std::vector<double*> device_deltas_;
    std::vector<double*> device_next_states_;
    std::vector<double*> device_back_buffers_;
    std::vector<double*> device_weight_velocities_;

    int batch_capacity_{0};
    std::vector<double*> device_batch_states_;
    std::vector<double*> device_batch_errors_;
    std::vector<double*> device_batch_pre_activations_;
    std::vector<double*> device_batch_deltas_;
    std::vector<double*> device_batch_back_buffers_;
    std::vector<double*> device_batch_gradients_;
    std::vector<double*> device_batch_centered_states_;
    std::vector<double*> device_batch_decor_signals_;
    std::vector<double*> device_batch_covariances_;
    std::vector<double*> device_batch_state_means_;
    std::vector<double> layer_second_moments_;
    double current_learning_rate_{0.0};
    double momentum_beta_{0.0};
};

}  // namespace ltfn

#endif
