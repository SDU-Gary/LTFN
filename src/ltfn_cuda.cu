#include "ltfn_cuda.h"

#if defined(LTFN_USE_CUDA_BACKEND)

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ltfn {

namespace {

constexpr int kThreadsPerBlock = 256;

int blocks_for(int elements) {
    return (elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

double clamp_probability(double value) {
    return std::max(1e-12, std::min(1.0 - 1e-12, value));
}

double binary_cross_entropy_scalar(double target, double prediction) {
    const double p = clamp_probability(prediction);
    return -(target * std::log(p) + (1.0 - target) * std::log(1.0 - p));
}

std::string cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
#if defined(CUBLAS_STATUS_NOT_SUPPORTED)
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

void throw_if_cuda_error(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

void throw_if_cublas_error(cublasStatus_t status, const char* context) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(context) + ": " + cublas_status_to_string(status));
    }
}

double vector_norm(cublasHandle_t handle, const double* device_ptr, int size) {
    double norm = 0.0;
    throw_if_cublas_error(cublasDnrm2(handle, size, device_ptr, 1, &norm), "cublasDnrm2");
    return norm;
}

double matrix_norm(cublasHandle_t handle, const double* device_ptr, int elements) {
    double norm = 0.0;
    throw_if_cublas_error(cublasDnrm2(handle, elements, device_ptr, 1, &norm), "cublasDnrm2(matrix)");
    return norm;
}

bool use_custom_local_rule(const LTFNConfig& config) {
    return config.layer_adapt_beta > 0.0;
}

double blend_second_moment(double previous, double beta, double mean_square) {
    return beta * previous + (1.0 - beta) * mean_square;
}

double matrix_mean_square_from_norm(double norm, int elements) {
    if (elements <= 0) {
        return 0.0;
    }
    return (norm * norm) / static_cast<double>(elements);
}

void normalize_precision_vector(std::vector<double>& precisions, double min_value, double max_value) {
    if (precisions.empty()) {
        return;
    }

    double log_sum = 0.0;
    for (double precision : precisions) {
        log_sum += std::log(std::max(precision, 1e-12));
    }
    const double geometric_mean = std::exp(log_sum / static_cast<double>(precisions.size()));
    for (double& precision : precisions) {
        precision = std::clamp(precision / geometric_mean, min_value, max_value);
    }
}

void normalize_precision_vector(Eigen::VectorXd& precisions, double min_value, double max_value) {
    if (precisions.size() == 0) {
        return;
    }

    double log_sum = 0.0;
    for (Eigen::Index i = 0; i < precisions.size(); ++i) {
        log_sum += std::log(std::max(precisions(i), 1e-12));
    }
    const double geometric_mean = std::exp(log_sum / static_cast<double>(precisions.size()));
    precisions = (precisions.array() / geometric_mean).max(min_value).min(max_value).matrix();
}

__global__ void sigmoid_kernel(const double* pre_activation, double* prediction, double* derivative, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    const double value = fmax(-500.0, fmin(500.0, pre_activation[index]));
    const double sigma = 1.0 / (1.0 + exp(-value));
    prediction[index] = sigma;
    derivative[index] = sigma * (1.0 - sigma);
}

__global__ void error_and_delta_kernel(
    const double* state,
    const double* prediction,
    const double* derivative,
    double* error,
    double* delta,
    int size,
    int use_bce_delta) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    const double diff = state[index] - prediction[index];
    error[index] = diff;
    delta[index] = use_bce_delta != 0 ? diff : diff * derivative[index];
}

__global__ void fused_batch_error_delta_kernel(
    const double* state,
    const double* pre_activation,
    double* error,
    double* delta,
    int elements,
    int use_bce_delta) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }

    const double value = fmax(-500.0, fmin(500.0, pre_activation[index]));
    const double sigma = 1.0 / (1.0 + exp(-value));
    const double diff = state[index] - sigma;
    error[index] = diff;
    delta[index] = use_bce_delta != 0 ? diff : diff * sigma * (1.0 - sigma);
}

__global__ void add_bias_kernel(double* values, const double* bias, int rows, int cols) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * cols;
    if (index >= elements) {
        return;
    }

    const int row = index % rows;
    values[index] += bias[row];
}

__global__ void apply_precision_kernel(double* values, const double* precisions, int rows, int cols) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * cols;
    if (index >= elements) {
        return;
    }

    const int row = index % rows;
    values[index] *= precisions[row];
}

__global__ void update_middle_state_kernel(
    const double* state,
    const double* error,
    double error_scale,
    const double* back,
    double scale,
    double* next_state,
    int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    next_state[index] = state[index] - scale * (error_scale * error[index] - back[index]);
}

__global__ void row_mean_kernel(const double* matrix, double* row_means, int rows, int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (int col = 0; col < cols; ++col) {
        sum += matrix[row + col * rows];
    }
    row_means[row] = sum / static_cast<double>(cols);
}

__global__ void center_rows_kernel(
    const double* matrix,
    const double* row_means,
    double* centered,
    int rows,
    int cols) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = rows * cols;
    if (index >= elements) {
        return;
    }
    const int row = index % rows;
    centered[index] = matrix[index] - row_means[row];
}

__global__ void zero_diagonal_kernel(double* matrix, int dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= dim) {
        return;
    }
    matrix[index + index * dim] = 0.0;
}

}  // namespace

LTFNCuda::LTFNCuda(const LTFNConfig& config, std::uint32_t seed)
    : config_(config) {
    validate_dims();
    current_learning_rate_ = config_.lr_w;

    int device_count = 0;
    throw_if_cuda_error(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA device is available.");
    }

    throw_if_cublas_error(cublasCreate(&cublas_), "cublasCreate");

    const std::size_t layers = config_.dims.size() - 1;
    host_weights_.reserve(layers);
    host_biases_.reserve(layers);
    host_states_.reserve(config_.dims.size());
    host_reconstruction_ = Eigen::VectorXd::Zero(config_.dims.front());
    host_visible_error_precisions_ = Eigen::VectorXd::Ones(config_.dims.front());
    visible_error_second_moments_ = Eigen::VectorXd::Zero(config_.dims.front());
    visible_error_precisions_ = Eigen::VectorXd::Ones(config_.dims.front());

    std::mt19937 generator(seed);
    for (std::size_t l = 0; l < config_.dims.size(); ++l) {
        host_states_.emplace_back(Eigen::VectorXd::Zero(config_.dims[l]));
        if (l == layers) {
            continue;
        }

        const int n_in = config_.dims[l + 1];
        const int n_out = config_.dims[l];
        const double limit = std::sqrt(6.0 / static_cast<double>(n_in + n_out));
        std::uniform_real_distribution<double> distribution(-limit, limit);

        Eigen::MatrixXd weight = Eigen::MatrixXd::Zero(n_out, n_in);
        for (int row = 0; row < n_out; ++row) {
            for (int col = 0; col < n_in; ++col) {
                weight(row, col) = distribution(generator);
            }
        }
        host_weights_.push_back(std::move(weight));
        host_biases_.emplace_back(Eigen::VectorXd::Zero(n_out));
    }

    allocate_buffers();
    upload_all_weights();
    upload_all_biases();
    throw_if_cuda_error(
        cudaMemcpy(
            device_visible_error_precisions_,
            visible_error_precisions_.data(),
            sizeof(double) * static_cast<std::size_t>(visible_error_precisions_.size()),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(visible precision upload)");
    zero_weight_velocities();
    zero_bias_velocities();
    layer_second_moments_.assign(layers, 0.0);
    layer_error_second_moments_.assign(layers, 0.0);
    layer_error_precisions_.assign(layers, 1.0);
    zero_latent_states();
}

LTFNCuda::~LTFNCuda() {
    release_buffers();
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

const LTFNConfig& LTFNCuda::config() const noexcept {
    return config_;
}

const std::vector<Eigen::MatrixXd>& LTFNCuda::weights() const {
    download_all_weights();
    return host_weights_;
}

const std::vector<Eigen::VectorXd>& LTFNCuda::biases() const {
    download_all_biases();
    return host_biases_;
}

void LTFNCuda::set_weights(const std::vector<Eigen::MatrixXd>& new_weights) {
    if (new_weights.size() != host_weights_.size()) {
        throw std::invalid_argument("Weight count does not match the configured architecture.");
    }
    for (std::size_t l = 0; l < new_weights.size(); ++l) {
        if (new_weights[l].rows() != host_weights_[l].rows() || new_weights[l].cols() != host_weights_[l].cols()) {
            throw std::invalid_argument("Checkpoint weight shape does not match the configured architecture.");
        }
    }

    host_weights_ = new_weights;
    upload_all_weights();
    zero_weight_velocities();
    zero_bias_velocities();
    layer_second_moments_.assign(host_weights_.size(), 0.0);
    layer_error_second_moments_.assign(host_weights_.size(), 0.0);
    layer_error_precisions_.assign(host_weights_.size(), 1.0);
    visible_error_second_moments_.setZero();
    visible_error_precisions_.setOnes();
    throw_if_cuda_error(
        cudaMemcpy(
            device_visible_error_precisions_,
            visible_error_precisions_.data(),
            sizeof(double) * static_cast<std::size_t>(visible_error_precisions_.size()),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(reset visible precision)");
    predictions_dirty_ = true;
}

void LTFNCuda::set_biases(const std::vector<Eigen::VectorXd>& new_biases) {
    if (new_biases.size() != host_biases_.size()) {
        throw std::invalid_argument("Bias count does not match the configured architecture.");
    }
    for (std::size_t l = 0; l < new_biases.size(); ++l) {
        if (new_biases[l].size() != host_biases_[l].size()) {
            throw std::invalid_argument("Checkpoint bias shape does not match the configured architecture.");
        }
    }

    host_biases_ = new_biases;
    upload_all_biases();
    zero_bias_velocities();
    predictions_dirty_ = true;
}

void LTFNCuda::set_learning_rate(double learning_rate) {
    if (learning_rate < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative.");
    }
    current_learning_rate_ = learning_rate;
}

void LTFNCuda::set_weight_momentum(double momentum_beta) {
    if (momentum_beta < 0.0 || momentum_beta >= 1.0) {
        throw std::invalid_argument("Momentum beta must be in [0, 1).");
    }
    momentum_beta_ = momentum_beta;
}

void LTFNCuda::reset_states(const Eigen::VectorXd& input) {
    ensure_input_shape(input);
    initialize_host_states_from_input(input);
    throw_if_cuda_error(
        cudaMemcpy(
            device_states_.front(),
            host_states_.front().data(),
            sizeof(double) * static_cast<std::size_t>(host_states_.front().size()),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(input)");
    for (std::size_t l = 1; l < host_states_.size(); ++l) {
        throw_if_cuda_error(
            cudaMemcpy(
                device_states_[l],
                host_states_[l].data(),
                sizeof(double) * static_cast<std::size_t>(host_states_[l].size()),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(initial latent state)");
    }
    compute_predictions_errors_and_deltas();
}

void LTFNCuda::advance(const Eigen::VectorXd& input, bool update_weights) {
    ensure_input_shape(input);

    throw_if_cuda_error(
        cudaMemcpy(
            device_states_.front(),
            input.data(),
            sizeof(double) * static_cast<std::size_t>(input.size()),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(step input)");
    predictions_dirty_ = true;
    advance_current(update_weights);
}

void LTFNCuda::advance_current(bool update_weights) {
    ensure_predictions_current();

    const double state_scale = config_.dt_r / config_.tau_r;
    const std::size_t top_index = config_.dims.size() - 1;

    if (config_.sequential_inference) {
        {
            const std::size_t weight_index = top_index - 1;
            const int rows = config_.dims[weight_index];
            const int cols = config_.dims[top_index];
            const double alpha = layer_error_precisions_[weight_index];
            const double beta = 0.0;
            throw_if_cublas_error(
                cublasDgemv(
                    cublas_,
                    CUBLAS_OP_T,
                    rows,
                    cols,
                    &alpha,
                    device_weights_[weight_index],
                    rows,
                    device_deltas_[weight_index],
                    1,
                    &beta,
                    device_back_buffers_[weight_index],
                    1),
                "cublasDgemv(top state)");
            throw_if_cublas_error(
                cublasDaxpy(
                    cublas_,
                    cols,
                    &state_scale,
                    device_back_buffers_[weight_index],
                    1,
                    device_states_[top_index],
                    1),
                "cublasDaxpy(top state)");
            compute_layer_prediction_error_delta(weight_index);
        }

        for (std::size_t l = top_index - 1; l >= 1; --l) {
            const double alpha = layer_error_precisions_[l - 1];
            const double beta = 0.0;
            const int rows = config_.dims[l - 1];
            const int cols = config_.dims[l];

            throw_if_cublas_error(
                cublasDgemv(
                    cublas_,
                    CUBLAS_OP_T,
                    rows,
                    cols,
                    &alpha,
                    device_weights_[l - 1],
                    rows,
                    device_deltas_[l - 1],
                    1,
                    &beta,
                    device_back_buffers_[l - 1],
                    1),
                "cublasDgemv(backprojection)");

            const double subtract_alpha = -layer_error_precisions_[l];
            throw_if_cublas_error(
                cublasDaxpy(
                    cublas_,
                    cols,
                    &subtract_alpha,
                    device_errors_[l],
                    1,
                    device_back_buffers_[l - 1],
                    1),
                "cublasDaxpy(error subtraction)");

            throw_if_cublas_error(
                cublasDaxpy(
                    cublas_,
                    cols,
                    &state_scale,
                    device_back_buffers_[l - 1],
                    1,
                    device_states_[l],
                    1),
                "cublasDaxpy(middle state)");
            compute_layer_prediction_error_delta(l - 1);

            if (l == 1) {
                break;
            }
        }
    } else {
        {
            const double alpha = state_scale * layer_error_precisions_[top_index - 1];
            const double beta = 1.0;
            const int rows = config_.dims[top_index - 1];
            const int cols = config_.dims[top_index];
            throw_if_cublas_error(
                cublasDgemv(
                    cublas_,
                    CUBLAS_OP_T,
                    rows,
                    cols,
                    &alpha,
                    device_weights_[top_index - 1],
                    rows,
                    device_deltas_[top_index - 1],
                    1,
                    &beta,
                    device_states_[top_index],
                    1),
                "cublasDgemv(top state)");
        }

        for (std::size_t l = top_index - 1; l >= 1; --l) {
            const double alpha = layer_error_precisions_[l - 1];
            const double beta = 0.0;
            const int rows = config_.dims[l - 1];
            const int cols = config_.dims[l];

            throw_if_cublas_error(
                cublasDgemv(
                    cublas_,
                    CUBLAS_OP_T,
                    rows,
                    cols,
                    &alpha,
                    device_weights_[l - 1],
                    rows,
                    device_deltas_[l - 1],
                    1,
                    &beta,
                    device_back_buffers_[l - 1],
                    1),
                "cublasDgemv(backprojection)");

            update_middle_state_kernel<<<blocks_for(cols), kThreadsPerBlock>>>(
                device_states_[l],
                device_errors_[l],
                layer_error_precisions_[l],
                device_back_buffers_[l - 1],
                state_scale,
                device_states_[l],
                cols);
            throw_if_cuda_error(cudaGetLastError(), "update_middle_state_kernel");

            if (l == 1) {
                break;
            }
        }
    }
    predictions_dirty_ = true;

    if (update_weights) {
        compute_predictions_errors_and_deltas();
        update_error_precisions_from_current_errors();
        const double weight_scale = current_learning_rate_ * config_.dt_w;
        const bool custom_rule = use_custom_local_rule(config_);
        for (std::size_t l = 0; l < device_weights_.size(); ++l) {
            const int rows = config_.dims[l];
            const int cols = config_.dims[l + 1];
            const int elements = config_.dims[l] * config_.dims[l + 1];
            double metric_weight_scale = weight_scale;
            double applied_weight_scale = weight_scale * layer_error_precisions_[l];
            if (custom_rule) {
                const double grad_norm =
                    layer_error_precisions_[l] * vector_norm(cublas_, device_deltas_[l], rows) *
                    vector_norm(cublas_, device_states_[l + 1], cols);
                const double next_second_moment = blend_second_moment(
                    layer_second_moments_[l],
                    config_.layer_adapt_beta,
                    matrix_mean_square_from_norm(grad_norm, elements));
                layer_second_moments_[l] = next_second_moment;
                const double denom =
                    std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon;
                metric_weight_scale /= denom;
                applied_weight_scale /= denom;
            }
            if (momentum_beta_ > 0.0) {
                throw_if_cublas_error(
                    cublasDscal(cublas_, elements, &momentum_beta_, device_weight_velocities_[l], 1),
                    "cublasDscal(weight velocity)");
                throw_if_cublas_error(
                    cublasDger(
                        cublas_,
                        rows,
                        cols,
                        &applied_weight_scale,
                        device_deltas_[l],
                        1,
                        device_states_[l + 1],
                        1,
                        device_weight_velocities_[l],
                        rows),
                    "cublasDger(weight velocity update)");
                const double one = 1.0;
                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        elements,
                        &one,
                        device_weight_velocities_[l],
                        1,
                        device_weights_[l],
                        1),
                    "cublasDaxpy(weight momentum update)");
                if (config_.use_biases) {
                    throw_if_cublas_error(
                        cublasDscal(cublas_, rows, &momentum_beta_, device_bias_velocities_[l], 1),
                        "cublasDscal(bias velocity)");
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            rows,
                            &applied_weight_scale,
                            device_deltas_[l],
                            1,
                            device_bias_velocities_[l],
                            1),
                        "cublasDaxpy(bias velocity update)");
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            rows,
                            &one,
                            device_bias_velocities_[l],
                            1,
                            device_biases_[l],
                            1),
                        "cublasDaxpy(bias momentum update)");
                }
            } else {
                throw_if_cublas_error(
                    cublasDger(
                        cublas_,
                        rows,
                        cols,
                        &applied_weight_scale,
                        device_deltas_[l],
                        1,
                        device_states_[l + 1],
                        1,
                        device_weights_[l],
                        rows),
                    "cublasDger(weight update)");
                if (config_.use_biases) {
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            rows,
                            &applied_weight_scale,
                            device_deltas_[l],
                            1,
                            device_biases_[l],
                            1),
                        "cublasDaxpy(bias update)");
                }
            }
        }
        predictions_dirty_ = true;
    }
}

StepDiagnostics LTFNCuda::step(const Eigen::VectorXd& input, bool update_weights) {
    ensure_input_shape(input);

    throw_if_cuda_error(
        cudaMemcpy(
            device_states_.front(),
            input.data(),
            sizeof(double) * static_cast<std::size_t>(input.size()),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(step input)");
    predictions_dirty_ = true;
    return step_current(update_weights);
}

StepDiagnostics LTFNCuda::step_current(bool update_weights) {
    ensure_predictions_current();

    const double state_scale = config_.dt_r / config_.tau_r;
    const std::size_t top_index = config_.dims.size() - 1;
    std::vector<double> state_update_norms(device_weights_.size(), 0.0);

    if (config_.sequential_inference) {
        {
            const std::size_t weight_index = top_index - 1;
            const int rows = config_.dims[weight_index];
            const int cols = config_.dims[top_index];
            const double alpha = layer_error_precisions_[weight_index];
            const double beta = 0.0;
            throw_if_cublas_error(
                cublasDgemv(
                    cublas_,
                    CUBLAS_OP_T,
                    rows,
                    cols,
                    &alpha,
                    device_weights_[weight_index],
                    rows,
                    device_deltas_[weight_index],
                    1,
                    &beta,
                    device_back_buffers_[weight_index],
                    1),
                "cublasDgemv(top state)");
            state_update_norms[weight_index] =
                std::abs(state_scale) * vector_norm(cublas_, device_back_buffers_[weight_index], cols);
            throw_if_cublas_error(
                cublasDaxpy(
                    cublas_,
                    cols,
                    &state_scale,
                    device_back_buffers_[weight_index],
                    1,
                    device_states_[top_index],
                    1),
                "cublasDaxpy(top state)");
            compute_layer_prediction_error_delta(weight_index);
        }

        for (std::size_t l = top_index - 1; l >= 1; --l) {
            const double alpha = layer_error_precisions_[l - 1];
            const double beta = 0.0;
            const int rows = config_.dims[l - 1];
            const int cols = config_.dims[l];

            throw_if_cublas_error(
                cublasDgemv(
                    cublas_,
                    CUBLAS_OP_T,
                    rows,
                    cols,
                    &alpha,
                    device_weights_[l - 1],
                    rows,
                    device_deltas_[l - 1],
                    1,
                    &beta,
                    device_back_buffers_[l - 1],
                    1),
                "cublasDgemv(backprojection)");

            const double subtract_alpha = -layer_error_precisions_[l];
            throw_if_cublas_error(
                cublasDaxpy(
                    cublas_,
                    cols,
                    &subtract_alpha,
                    device_errors_[l],
                    1,
                    device_back_buffers_[l - 1],
                    1),
                "cublasDaxpy(error subtraction)");
            state_update_norms[l - 1] =
                std::abs(state_scale) * vector_norm(cublas_, device_back_buffers_[l - 1], cols);
            throw_if_cublas_error(
                cublasDaxpy(
                    cublas_,
                    cols,
                    &state_scale,
                    device_back_buffers_[l - 1],
                    1,
                    device_states_[l],
                    1),
                "cublasDaxpy(middle state)");
            compute_layer_prediction_error_delta(l - 1);

            if (l == 1) {
                break;
            }
        }
    } else {
        throw_if_cuda_error(
            cudaMemcpy(
                device_next_states_[top_index],
                device_states_[top_index],
                sizeof(double) * static_cast<std::size_t>(config_.dims[top_index]),
                cudaMemcpyDeviceToDevice),
            "cudaMemcpy(top state)");

        {
            const double alpha = state_scale * layer_error_precisions_[top_index - 1];
            const double beta = 1.0;
            const int rows = config_.dims[top_index - 1];
            const int cols = config_.dims[top_index];
            throw_if_cublas_error(
                cublasDgemv(
                    cublas_,
                    CUBLAS_OP_T,
                    rows,
                    cols,
                    &alpha,
                    device_weights_[top_index - 1],
                    rows,
                    device_deltas_[top_index - 1],
                    1,
                    &beta,
                    device_next_states_[top_index],
                    1),
                "cublasDgemv(top state)");
        }

        for (std::size_t l = top_index - 1; l >= 1; --l) {
            const double alpha = layer_error_precisions_[l - 1];
            const double beta = 0.0;
            const int rows = config_.dims[l - 1];
            const int cols = config_.dims[l];

            throw_if_cublas_error(
                cublasDgemv(
                    cublas_,
                    CUBLAS_OP_T,
                    rows,
                    cols,
                    &alpha,
                    device_weights_[l - 1],
                    rows,
                    device_deltas_[l - 1],
                    1,
                    &beta,
                    device_back_buffers_[l - 1],
                    1),
                "cublasDgemv(backprojection)");

            update_middle_state_kernel<<<blocks_for(cols), kThreadsPerBlock>>>(
                device_states_[l],
                device_errors_[l],
                layer_error_precisions_[l],
                device_back_buffers_[l - 1],
                state_scale,
                device_next_states_[l],
                cols);
            throw_if_cuda_error(cudaGetLastError(), "update_middle_state_kernel");

            if (l == 1) {
                break;
            }
        }

        for (std::size_t l = 1; l < device_states_.size(); ++l) {
            throw_if_cuda_error(
                cudaMemcpy(
                    device_back_buffers_[l - 1],
                    device_next_states_[l],
                    sizeof(double) * static_cast<std::size_t>(config_.dims[l]),
                    cudaMemcpyDeviceToDevice),
                "cudaMemcpy(state delta seed)");
            {
                const double alpha = -1.0;
                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        config_.dims[l],
                        &alpha,
                        device_states_[l],
                        1,
                        device_back_buffers_[l - 1],
                        1),
                    "cublasDaxpy(state delta)");
                state_update_norms[l - 1] = vector_norm(cublas_, device_back_buffers_[l - 1], config_.dims[l]);
            }
            throw_if_cuda_error(
                cudaMemcpy(
                    device_states_[l],
                    device_next_states_[l],
                    sizeof(double) * static_cast<std::size_t>(config_.dims[l]),
                    cudaMemcpyDeviceToDevice),
                "cudaMemcpy(updated state)");
        }
    }

    compute_predictions_errors_and_deltas();
    if (update_weights) {
        update_error_precisions_from_current_errors();
    }
    StepDiagnostics diagnostics = collect_diagnostics(update_weights);
    diagnostics.state_update_norms = std::move(state_update_norms);

    if (update_weights) {
        compute_predictions_errors_and_deltas();
    }

    diagnostics.energy = current_energy();
    diagnostics.mse = static_cast<double>(config_.dims.front()) > 0
        ? (std::pow(vector_norm(cublas_, device_errors_.front(), config_.dims.front()), 2.0) /
              static_cast<double>(config_.dims.front()))
        : 0.0;
    return diagnostics;
}

StepDiagnostics LTFNCuda::current_diagnostics() const {
    ensure_predictions_current();
    StepDiagnostics diagnostics;
    diagnostics.error_norms.reserve(device_errors_.size());
    diagnostics.state_update_norms.assign(device_weights_.size(), 0.0);
    diagnostics.weight_gradient_norms.reserve(device_weights_.size());
    diagnostics.weight_update_norms.reserve(device_weights_.size());
    diagnostics.weight_norms.reserve(device_weights_.size());

    double energy = 0.0;
    const double weight_scale = current_learning_rate_ * config_.dt_w;
    const bool custom_rule = use_custom_local_rule(config_);
    for (std::size_t l = 0; l < device_errors_.size(); ++l) {
        const int rows = config_.dims[l];
        const int cols = config_.dims[l + 1];
        const int elements = rows * cols;
        const double error_norm = vector_norm(cublas_, device_errors_[l], config_.dims[l]);
        const double delta_norm = vector_norm(cublas_, device_deltas_[l], config_.dims[l]);
        const double state_norm = vector_norm(cublas_, device_states_[l + 1], config_.dims[l + 1]);
        const double grad_norm = layer_error_precisions_[l] * delta_norm * state_norm;
        double update_norm = weight_scale * grad_norm;

        if (custom_rule) {
            double metric_weight_scale = weight_scale;
            const double next_second_moment = blend_second_moment(
                layer_second_moments_[l],
                config_.layer_adapt_beta,
                matrix_mean_square_from_norm(grad_norm, elements));
            metric_weight_scale /=
                (std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon);
            if (momentum_beta_ > 0.0) {
                update_norm =
                    momentum_beta_ * matrix_norm(cublas_, device_weight_velocities_[l], elements) +
                    metric_weight_scale * grad_norm;
            } else {
                update_norm = metric_weight_scale * grad_norm;
            }
        }

        diagnostics.error_norms.push_back(error_norm);
        diagnostics.weight_gradient_norms.push_back(grad_norm);
        diagnostics.weight_update_norms.push_back(update_norm);
        diagnostics.weight_norms.push_back(matrix_norm(cublas_, device_weights_[l], elements));
        if (config_.visible_loss == VisibleLoss::Bce && l == 0) {
            Eigen::VectorXd host_state(config_.dims.front());
            Eigen::VectorXd host_prediction(config_.dims.front());
            const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims.front());
            throw_if_cuda_error(
                cudaMemcpy(host_state.data(), device_states_[0], bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(visible state for BCE energy)");
            throw_if_cuda_error(
                cudaMemcpy(host_prediction.data(), device_predictions_[0], bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(visible prediction for BCE energy)");
            for (Eigen::Index i = 0; i < host_state.size(); ++i) {
                const double precision = config_.visible_unit_precision
                    ? visible_error_precisions_(i)
                    : layer_error_precisions_[l];
                energy += precision * binary_cross_entropy_scalar(host_state(i), host_prediction(i));
            }
        } else if (config_.visible_unit_precision && l == 0) {
            Eigen::VectorXd host_error(config_.dims.front());
            const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims.front());
            throw_if_cuda_error(
                cudaMemcpy(host_error.data(), device_errors_[0], bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(visible error for weighted current_diagnostics)");
            energy += 0.5 * (visible_error_precisions_.array() * host_error.array().square()).sum();
        } else {
            energy += layer_error_precisions_[l] * 0.5 * error_norm * error_norm;
        }
    }

    diagnostics.energy = energy;
    diagnostics.mse = static_cast<double>(config_.dims.front()) > 0
        ? (diagnostics.error_norms.front() * diagnostics.error_norms.front() /
              static_cast<double>(config_.dims.front()))
        : 0.0;
    return diagnostics;
}

BatchTrainResult LTFNCuda::train_batch(
    const std::vector<const Eigen::VectorXd*>& inputs,
    int steps,
    const BatchTrainOptions& options) {
    if (inputs.empty()) {
        throw std::invalid_argument("train_batch requires at least one input sample.");
    }
    if (steps < 0) {
        throw std::invalid_argument("train_batch steps must be non-negative.");
    }

    const int batch_size = static_cast<int>(inputs.size());
    const std::size_t layer_count = device_weights_.size();
    const std::size_t state_count = config_.dims.size();
    const double state_scale = config_.dt_r / config_.tau_r;
    const double update_scale = current_learning_rate_ * config_.dt_w;
    const double batch_norm_scale = 1.0 / std::sqrt(static_cast<double>(batch_size));
    const bool custom_rule = use_custom_local_rule(config_);

    std::vector<int> logged_steps = options.logged_relax_steps;
    logged_steps.erase(
        std::remove_if(logged_steps.begin(), logged_steps.end(), [steps](int step) { return step < 0 || step > steps; }),
        logged_steps.end());
    std::sort(logged_steps.begin(), logged_steps.end());
    logged_steps.erase(std::unique(logged_steps.begin(), logged_steps.end()), logged_steps.end());

    upload_batch_inputs(inputs);
    compute_batch_errors_and_deltas(batch_size);
    if (options.update_weights) {
        update_error_precisions_from_batch_errors(batch_size);
    }

    BatchTrainResult result;
    result.batch_size = inputs.size();
    if (options.capture_final_gradients) {
        result.final_weight_gradients.resize(layer_count);
    }
    if (options.capture_final_states) {
        result.final_batch_states.resize(state_count);
    }

    auto compute_batch_gradients = [&]() -> std::vector<double> {
        std::vector<double> gradient_norms(layer_count, 0.0);
        for (std::size_t l = 0; l < layer_count; ++l) {
            gradient_norms[l] = compute_batch_effective_gradient(l, batch_size);
        }
        return gradient_norms;
    };

    auto maybe_log_step =
        [&](int relax_step,
            const std::vector<double>& state_update_norms,
            const std::vector<double>* gradient_norms,
            const std::vector<double>* update_norms) {
        if (!std::binary_search(logged_steps.begin(), logged_steps.end(), relax_step)) {
            return;
        }

        LoggedRelaxationStep entry;
        entry.relax_step = relax_step;
        entry.diagnostics = collect_batch_diagnostics(batch_size, &state_update_norms, gradient_norms, update_norms);
        result.logged_steps.push_back(std::move(entry));
    };

    maybe_log_step(0, std::vector<double>(layer_count, 0.0), nullptr, nullptr);

    const std::size_t top_index = state_count - 1;
    std::vector<double> last_state_update_norms(layer_count, 0.0);
    std::vector<double> last_gradient_norms(layer_count, 0.0);
    std::vector<double> last_update_norms(layer_count, 0.0);
    std::vector<Eigen::MatrixXd> last_gradients;
    if (options.capture_final_gradients) {
        last_gradients.resize(layer_count);
    }
    for (int relax_step = 1; relax_step <= steps; ++relax_step) {
        std::vector<double> state_update_norms(layer_count, 0.0);
        const double step_gate = config_.transient_gate_tau > 0.0
            ? (1.0 - std::exp(-static_cast<double>(relax_step) / config_.transient_gate_tau))
            : 1.0;

        if (config_.sequential_inference) {
            {
                const std::size_t weight_index = top_index - 1;
                const int rows = config_.dims[weight_index];
                const int cols = config_.dims[top_index];
                const double alpha = layer_error_precisions_[weight_index];
                const double beta = 0.0;
                throw_if_cublas_error(
                    cublasDgemm(
                        cublas_,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        cols,
                        batch_size,
                        rows,
                        &alpha,
                        device_weights_[weight_index],
                        rows,
                        device_batch_deltas_[weight_index],
                        rows,
                        &beta,
                        device_batch_back_buffers_[weight_index],
                        cols),
                    "cublasDgemm(batch top backprojection)");

                state_update_norms[weight_index] = std::abs(state_scale) *
                    matrix_norm(cublas_, device_batch_back_buffers_[weight_index], cols * batch_size) *
                    batch_norm_scale;

                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        cols * batch_size,
                        &state_scale,
                        device_batch_back_buffers_[weight_index],
                        1,
                        device_batch_states_[top_index],
                        1),
                    "cublasDaxpy(batch top state)");
                compute_batch_layer_error_delta(weight_index, batch_size);
            }

            for (std::size_t l = top_index - 1; l >= 1; --l) {
                const int rows = config_.dims[l - 1];
                const int cols = config_.dims[l];
                const double alpha = layer_error_precisions_[l - 1];
                const double beta = 0.0;
                throw_if_cublas_error(
                    cublasDgemm(
                        cublas_,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        cols,
                        batch_size,
                        rows,
                        &alpha,
                        device_weights_[l - 1],
                        rows,
                        device_batch_deltas_[l - 1],
                        rows,
                        &beta,
                        device_batch_back_buffers_[l - 1],
                        cols),
                    "cublasDgemm(batch middle backprojection)");

                const double subtract_alpha = -layer_error_precisions_[l];
                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        cols * batch_size,
                        &subtract_alpha,
                        device_batch_errors_[l],
                        1,
                        device_batch_back_buffers_[l - 1],
                        1),
                    "cublasDaxpy(batch error subtraction)");

                state_update_norms[l - 1] = std::abs(state_scale) *
                    matrix_norm(cublas_, device_batch_back_buffers_[l - 1], cols * batch_size) *
                    batch_norm_scale;

                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        cols * batch_size,
                        &state_scale,
                        device_batch_back_buffers_[l - 1],
                        1,
                        device_batch_states_[l],
                        1),
                    "cublasDaxpy(batch middle state)");
                compute_batch_layer_error_delta(l - 1, batch_size);

                if (l == 1) {
                    break;
                }
            }
        } else {
            {
                const std::size_t weight_index = top_index - 1;
                const int rows = config_.dims[weight_index];
                const int cols = config_.dims[top_index];
                const double alpha = layer_error_precisions_[weight_index];
                const double beta = 0.0;
                throw_if_cublas_error(
                    cublasDgemm(
                        cublas_,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        cols,
                        batch_size,
                        rows,
                        &alpha,
                        device_weights_[weight_index],
                        rows,
                        device_batch_deltas_[weight_index],
                        rows,
                        &beta,
                        device_batch_back_buffers_[weight_index],
                        cols),
                    "cublasDgemm(batch top backprojection)");

                state_update_norms[weight_index] = std::abs(state_scale) *
                    matrix_norm(cublas_, device_batch_back_buffers_[weight_index], cols * batch_size) *
                    batch_norm_scale;

                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        cols * batch_size,
                        &state_scale,
                        device_batch_back_buffers_[weight_index],
                        1,
                        device_batch_states_[top_index],
                        1),
                    "cublasDaxpy(batch top state)");
            }

            for (std::size_t l = top_index - 1; l >= 1; --l) {
                const int rows = config_.dims[l - 1];
                const int cols = config_.dims[l];
                const double alpha = layer_error_precisions_[l - 1];
                const double beta = 0.0;
                throw_if_cublas_error(
                    cublasDgemm(
                        cublas_,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        cols,
                        batch_size,
                        rows,
                        &alpha,
                        device_weights_[l - 1],
                        rows,
                        device_batch_deltas_[l - 1],
                        rows,
                        &beta,
                        device_batch_back_buffers_[l - 1],
                        cols),
                    "cublasDgemm(batch middle backprojection)");

                const double subtract_alpha = -layer_error_precisions_[l];
                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        cols * batch_size,
                        &subtract_alpha,
                        device_batch_errors_[l],
                        1,
                        device_batch_back_buffers_[l - 1],
                        1),
                    "cublasDaxpy(batch error subtraction)");

                state_update_norms[l - 1] = std::abs(state_scale) *
                    matrix_norm(cublas_, device_batch_back_buffers_[l - 1], cols * batch_size) *
                    batch_norm_scale;

                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        cols * batch_size,
                        &state_scale,
                        device_batch_back_buffers_[l - 1],
                        1,
                        device_batch_states_[l],
                        1),
                    "cublasDaxpy(batch middle state)");

                if (l == 1) {
                    break;
                }
            }
        }

        compute_batch_errors_and_deltas(batch_size);
        if (options.update_weights) {
            update_error_precisions_from_batch_errors(batch_size);
        }
        std::vector<double> gradient_norms = compute_batch_gradients();
        std::vector<double> update_norms(layer_count, 0.0);
        for (std::size_t l = 0; l < layer_count; ++l) {
            const int rows = config_.dims[l];
            const int cols = config_.dims[l + 1];
            const int elements = rows * cols;
            if (options.capture_final_gradients) {
                Eigen::MatrixXd host_gradient(config_.dims[l], config_.dims[l + 1]);
                const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(elements);
                throw_if_cuda_error(
                    cudaMemcpy(host_gradient.data(), device_batch_gradients_[l], bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy(batch gradient download)");
                last_gradients[l] = std::move(host_gradient);
            }
            double scaled_update_scale = update_scale;
            double effective_gradient_norm = gradient_norms[l];
            if (config_.use_biases) {
                const double alpha = (1.0 / static_cast<double>(batch_size)) * layer_error_precisions_[l];
                const double beta = 0.0;
                throw_if_cublas_error(
                    cublasDgemv(
                        cublas_,
                        CUBLAS_OP_N,
                        rows,
                        batch_size,
                        &alpha,
                        device_batch_deltas_[l],
                        rows,
                        device_batch_ones_,
                        1,
                        &beta,
                        device_batch_bias_gradients_[l],
                        1),
                    "cublasDgemv(batch bias gradient)");
            }
            if (custom_rule) {
                if (config_.layer_adapt_beta > 0.0) {
                    const double next_second_moment = blend_second_moment(
                        layer_second_moments_[l],
                        config_.layer_adapt_beta,
                        matrix_mean_square_from_norm(effective_gradient_norm, elements));
                    if (options.update_weights) {
                        layer_second_moments_[l] = next_second_moment;
                    }
                    scaled_update_scale /=
                        (std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon);
                }
            }
            scaled_update_scale *= step_gate;
            if (options.update_weights) {
                if (momentum_beta_ > 0.0) {
                    throw_if_cublas_error(
                        cublasDscal(cublas_, elements, &momentum_beta_, device_weight_velocities_[l], 1),
                        "cublasDscal(batch weight velocity)");
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            elements,
                            &scaled_update_scale,
                            device_batch_gradients_[l],
                            1,
                            device_weight_velocities_[l],
                            1),
                        "cublasDaxpy(batch weight velocity update)");
                    update_norms[l] = matrix_norm(cublas_, device_weight_velocities_[l], elements);
                    const double one = 1.0;
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            elements,
                            &one,
                            device_weight_velocities_[l],
                            1,
                            device_weights_[l],
                            1),
                        "cublasDaxpy(batch momentum weight update)");
                    if (config_.use_biases) {
                        throw_if_cublas_error(
                            cublasDscal(cublas_, rows, &momentum_beta_, device_bias_velocities_[l], 1),
                            "cublasDscal(batch bias velocity)");
                        throw_if_cublas_error(
                            cublasDaxpy(
                                cublas_,
                                rows,
                                &scaled_update_scale,
                                device_batch_bias_gradients_[l],
                                1,
                                device_bias_velocities_[l],
                                1),
                            "cublasDaxpy(batch bias velocity update)");
                        const double one = 1.0;
                        throw_if_cublas_error(
                            cublasDaxpy(
                                cublas_,
                                rows,
                                &one,
                                device_bias_velocities_[l],
                                1,
                                device_biases_[l],
                                1),
                            "cublasDaxpy(batch bias momentum update)");
                    }
                } else {
                    update_norms[l] = scaled_update_scale * effective_gradient_norm;
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            elements,
                            &scaled_update_scale,
                            device_batch_gradients_[l],
                            1,
                            device_weights_[l],
                            1),
                        "cublasDaxpy(batch weight update)");
                    if (config_.use_biases) {
                        throw_if_cublas_error(
                            cublasDaxpy(
                                cublas_,
                                rows,
                                &scaled_update_scale,
                                device_batch_bias_gradients_[l],
                                1,
                                device_biases_[l],
                                1),
                            "cublasDaxpy(batch bias update)");
                    }
                }
            } else if (momentum_beta_ > 0.0) {
                update_norms[l] =
                    matrix_norm(cublas_, device_weight_velocities_[l], elements) * momentum_beta_ +
                    scaled_update_scale * effective_gradient_norm;
            } else {
                update_norms[l] = scaled_update_scale * effective_gradient_norm;
            }
        }
        compute_batch_errors_and_deltas(batch_size);
        last_state_update_norms = state_update_norms;
        last_gradient_norms = gradient_norms;
        last_update_norms = update_norms;
        maybe_log_step(relax_step, state_update_norms, &gradient_norms, &update_norms);
    }

    result.final_diagnostics = collect_batch_diagnostics(
        batch_size,
        &last_state_update_norms,
        steps > 0 ? &last_gradient_norms : nullptr,
        steps > 0 ? &last_update_norms : nullptr);
    if (options.capture_final_gradients) {
        result.final_weight_gradients = std::move(last_gradients);
    }
    if (options.capture_final_states) {
        for (std::size_t l = 0; l < state_count; ++l) {
            result.final_batch_states[l].resize(config_.dims[l], batch_size);
            const std::size_t bytes =
                sizeof(double) * static_cast<std::size_t>(config_.dims[l]) * static_cast<std::size_t>(batch_size);
            throw_if_cuda_error(
                cudaMemcpy(
                    result.final_batch_states[l].data(),
                    device_batch_states_[l],
                    bytes,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(batch states download)");
        }
    }
    result.average_mse = result.final_diagnostics.mse;
    predictions_dirty_ = true;
    return result;
}

RelaxationResult LTFNCuda::relax(const Eigen::VectorXd& input, int steps, bool capture_trace) {
    reset_states(input);

    RelaxationResult result;
    if (capture_trace) {
        result.energy_trace.reserve(static_cast<std::size_t>(std::max(steps, 0)));
    }

    StepDiagnostics diagnostics;
    if (capture_trace || steps == 0) {
        diagnostics = current_diagnostics();
    }
    if (capture_trace) {
        for (int t = 0; t < steps; ++t) {
            diagnostics = step_current(true);
            result.energy_trace.push_back(diagnostics.energy);
        }
    } else if (steps > 0) {
        for (int t = 1; t < steps; ++t) {
            advance_current(true);
        }
        diagnostics = step_current(true);
    }

    result.reconstruction = current_reconstruction();
    result.final_energy = diagnostics.energy;
    result.mse = compute_mse(input, result.reconstruction);
    result.final_error_norms = diagnostics.error_norms;
    result.final_weight_gradient_norms = diagnostics.weight_gradient_norms;
    return result;
}

RelaxationResult LTFNCuda::reconstruct(const Eigen::VectorXd& input, int steps, bool capture_trace) {
    reset_states(input);

    const bool adapt_inference_precisions = config_.error_precision_beta > 0.0;
    const std::vector<double> saved_layer_error_second_moments = layer_error_second_moments_;
    const std::vector<double> saved_layer_error_precisions = layer_error_precisions_;
    const Eigen::VectorXd saved_visible_error_second_moments = visible_error_second_moments_;
    const Eigen::VectorXd saved_visible_error_precisions = visible_error_precisions_;
    const auto restore_precision_state = [&]() {
        layer_error_second_moments_ = saved_layer_error_second_moments;
        layer_error_precisions_ = saved_layer_error_precisions;
        visible_error_second_moments_ = saved_visible_error_second_moments;
        visible_error_precisions_ = saved_visible_error_precisions;
        if (config_.visible_unit_precision) {
            const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims.front());
            throw_if_cuda_error(
                cudaMemcpy(
                    device_visible_error_precisions_,
                    visible_error_precisions_.data(),
                    bytes,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(visible precision restore)");
        }
    };

    RelaxationResult result;
    if (capture_trace) {
        result.energy_trace.reserve(static_cast<std::size_t>(std::max(steps, 0)));
    }

    if (adapt_inference_precisions) {
        update_error_precisions_from_current_errors();
    }

    StepDiagnostics diagnostics;
    if (capture_trace || steps == 0) {
        diagnostics = current_diagnostics();
    }
    if (capture_trace) {
        for (int t = 0; t < steps; ++t) {
            diagnostics = step_current(false);
            if (adapt_inference_precisions) {
                update_error_precisions_from_current_errors();
                diagnostics = current_diagnostics();
            }
            result.energy_trace.push_back(diagnostics.energy);
        }
    } else if (steps > 0) {
        for (int t = 1; t < steps; ++t) {
            advance_current(false);
            if (adapt_inference_precisions) {
                ensure_predictions_current();
                update_error_precisions_from_current_errors();
            }
        }
        diagnostics = step_current(false);
        if (adapt_inference_precisions) {
            update_error_precisions_from_current_errors();
            diagnostics = current_diagnostics();
        }
    }

    result.reconstruction = current_reconstruction();
    result.final_energy = diagnostics.energy;
    result.mse = compute_mse(input, result.reconstruction);
    result.final_error_norms = diagnostics.error_norms;
    result.final_weight_gradient_norms = diagnostics.weight_gradient_norms;
    restore_precision_state();
    return result;
}

double LTFNCuda::current_energy() const noexcept {
    try {
        ensure_predictions_current();
        double energy = 0.0;
        for (std::size_t l = 0; l < device_errors_.size(); ++l) {
            if (config_.visible_loss == VisibleLoss::Bce && l == 0) {
                Eigen::VectorXd host_state(config_.dims.front());
                Eigen::VectorXd host_prediction(config_.dims.front());
                const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims.front());
                throw_if_cuda_error(
                    cudaMemcpy(host_state.data(), device_states_[0], bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy(visible state for BCE current_energy)");
                throw_if_cuda_error(
                    cudaMemcpy(host_prediction.data(), device_predictions_[0], bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy(visible prediction for BCE current_energy)");
                for (Eigen::Index i = 0; i < host_state.size(); ++i) {
                    const double precision = config_.visible_unit_precision
                        ? visible_error_precisions_(i)
                        : layer_error_precisions_[l];
                    energy += precision * binary_cross_entropy_scalar(host_state(i), host_prediction(i));
                }
            } else if (config_.visible_unit_precision && l == 0) {
                Eigen::VectorXd host_error(config_.dims.front());
                const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims.front());
                throw_if_cuda_error(
                    cudaMemcpy(host_error.data(), device_errors_[0], bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy(visible error for weighted current_energy)");
                energy += 0.5 * (visible_error_precisions_.array() * host_error.array().square()).sum();
            } else {
                const double error_norm = vector_norm(cublas_, device_errors_[l], config_.dims[l]);
                energy += layer_error_precisions_[l] * 0.5 * error_norm * error_norm;
            }
        }
        return energy;
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

Eigen::VectorXd LTFNCuda::current_reconstruction() const {
    ensure_predictions_current();
    download_reconstruction();
    return host_reconstruction_;
}

const std::vector<Eigen::VectorXd>& LTFNCuda::states() const {
    download_all_states();
    return host_states_;
}

void LTFNCuda::validate_dims() const {
    if (config_.dims.size() < 2) {
        throw std::invalid_argument("LTFN requires at least an input layer and one latent layer.");
    }
    for (int dim : config_.dims) {
        if (dim <= 0) {
            throw std::invalid_argument("All layer dimensions must be positive.");
        }
    }
    if (config_.tau_r <= 0.0) {
        throw std::invalid_argument("tau_r must be positive.");
    }
    if (config_.error_precision_beta < 0.0 || config_.error_precision_beta >= 1.0) {
        throw std::invalid_argument("error_precision_beta must be in [0, 1).");
    }
    if (config_.error_precision_epsilon <= 0.0) {
        throw std::invalid_argument("error_precision_epsilon must be positive.");
    }
    if (config_.error_precision_min <= 0.0 || config_.error_precision_max < config_.error_precision_min) {
        throw std::invalid_argument("Error precision clamp bounds are invalid.");
    }
}

void LTFNCuda::ensure_input_shape(const Eigen::VectorXd& input) const {
    if (input.size() != config_.dims.front()) {
        throw std::invalid_argument("Input dimension does not match the configured visible layer.");
    }
}

void LTFNCuda::initialize_host_states_from_input(const Eigen::VectorXd& input) {
    host_states_.front() = input;
    if (config_.state_init == StateInit::Zero) {
        for (std::size_t l = 1; l < host_states_.size(); ++l) {
            host_states_[l].setZero();
        }
        return;
    }

    download_all_weights();
    if (config_.use_biases) {
        download_all_biases();
    }
    for (std::size_t l = 1; l < host_states_.size(); ++l) {
        const Eigen::VectorXd upward = host_weights_[l - 1].transpose() * host_states_[l - 1];
        const Eigen::ArrayXd clipped = upward.array().max(-500.0).min(500.0);
        host_states_[l] = (1.0 / (1.0 + (-clipped).exp())).matrix();
    }
}

void LTFNCuda::update_error_precisions_from_current_errors() {
    if (config_.error_precision_beta <= 0.0) {
        return;
    }

    std::size_t start_layer = 0;
    if (config_.visible_unit_precision) {
        Eigen::VectorXd host_visible_errors(config_.dims.front());
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims.front());
        throw_if_cuda_error(
            cudaMemcpy(host_visible_errors.data(), device_errors_[0], bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(visible errors for precision)");
        visible_error_second_moments_ =
            (config_.error_precision_beta * visible_error_second_moments_.array() +
                (1.0 - config_.error_precision_beta) * host_visible_errors.array().square())
                .matrix();
        visible_error_precisions_ =
            (1.0 / (visible_error_second_moments_.array().max(0.0).sqrt() + config_.error_precision_epsilon)).matrix();
        normalize_precision_vector(
            visible_error_precisions_,
            config_.error_precision_min,
            config_.error_precision_max);
        throw_if_cuda_error(
            cudaMemcpy(
                device_visible_error_precisions_,
                visible_error_precisions_.data(),
                bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy(visible precision upload)");
        layer_error_precisions_[0] = 1.0;
        start_layer = 1;
    }

    for (std::size_t l = start_layer; l < device_errors_.size(); ++l) {
        const double error_norm = vector_norm(cublas_, device_errors_[l], config_.dims[l]);
        const double mean_square = matrix_mean_square_from_norm(error_norm, config_.dims[l]);
        const double next_second_moment = blend_second_moment(
            layer_error_second_moments_[l],
            config_.error_precision_beta,
            mean_square);
        layer_error_second_moments_[l] = next_second_moment;
        layer_error_precisions_[l] =
            1.0 / (std::sqrt(std::max(0.0, next_second_moment)) + config_.error_precision_epsilon);
    }

    if (start_layer == 0) {
        normalize_precision_vector(
            layer_error_precisions_,
            config_.error_precision_min,
            config_.error_precision_max);
    } else if (layer_error_precisions_.size() > 1) {
        std::vector<double> hidden_precisions(
            layer_error_precisions_.begin() + 1,
            layer_error_precisions_.end());
        normalize_precision_vector(
            hidden_precisions,
            config_.error_precision_min,
            config_.error_precision_max);
        std::copy(hidden_precisions.begin(), hidden_precisions.end(), layer_error_precisions_.begin() + 1);
    }
}

void LTFNCuda::update_error_precisions_from_batch_errors(int batch_size) {
    if (config_.error_precision_beta <= 0.0) {
        return;
    }

    std::size_t start_layer = 0;
    if (config_.visible_unit_precision) {
        Eigen::MatrixXd host_visible_errors(config_.dims.front(), batch_size);
        const std::size_t bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims.front()) * static_cast<std::size_t>(batch_size);
        throw_if_cuda_error(
            cudaMemcpy(host_visible_errors.data(), device_batch_errors_[0], bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(batch visible errors for precision)");
        const Eigen::ArrayXd mean_square = host_visible_errors.array().square().rowwise().mean();
        visible_error_second_moments_ =
            (config_.error_precision_beta * visible_error_second_moments_.array() +
                (1.0 - config_.error_precision_beta) * mean_square)
                .matrix();
        visible_error_precisions_ =
            (1.0 / (visible_error_second_moments_.array().max(0.0).sqrt() + config_.error_precision_epsilon)).matrix();
        normalize_precision_vector(
            visible_error_precisions_,
            config_.error_precision_min,
            config_.error_precision_max);
        throw_if_cuda_error(
            cudaMemcpy(
                device_visible_error_precisions_,
                visible_error_precisions_.data(),
                sizeof(double) * static_cast<std::size_t>(config_.dims.front()),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(batch visible precision upload)");
        layer_error_precisions_[0] = 1.0;
        start_layer = 1;
    }

    for (std::size_t l = start_layer; l < device_batch_errors_.size(); ++l) {
        const double error_norm = matrix_norm(cublas_, device_batch_errors_[l], config_.dims[l] * batch_size);
        const double mean_square = matrix_mean_square_from_norm(error_norm, config_.dims[l] * batch_size);
        const double next_second_moment = blend_second_moment(
            layer_error_second_moments_[l],
            config_.error_precision_beta,
            mean_square);
        layer_error_second_moments_[l] = next_second_moment;
        layer_error_precisions_[l] =
            1.0 / (std::sqrt(std::max(0.0, next_second_moment)) + config_.error_precision_epsilon);
    }

    if (start_layer == 0) {
        normalize_precision_vector(
            layer_error_precisions_,
            config_.error_precision_min,
            config_.error_precision_max);
    } else if (layer_error_precisions_.size() > 1) {
        std::vector<double> hidden_precisions(
            layer_error_precisions_.begin() + 1,
            layer_error_precisions_.end());
        normalize_precision_vector(
            hidden_precisions,
            config_.error_precision_min,
            config_.error_precision_max);
        std::copy(hidden_precisions.begin(), hidden_precisions.end(), layer_error_precisions_.begin() + 1);
    }
}

void LTFNCuda::ensure_predictions_current() const {
    if (predictions_dirty_) {
        compute_predictions_errors_and_deltas();
    }
}

void LTFNCuda::allocate_buffers() {
    const std::size_t layers = config_.dims.size() - 1;
    const std::size_t visible_bytes = sizeof(double) * static_cast<std::size_t>(config_.dims.front());

    device_weights_.resize(layers, nullptr);
    device_biases_.resize(layers, nullptr);
    device_weight_velocities_.resize(layers, nullptr);
    device_bias_velocities_.resize(layers, nullptr);
    device_predictions_.resize(layers, nullptr);
    device_errors_.resize(layers, nullptr);
    device_pre_activations_.resize(layers, nullptr);
    device_sigmoid_derivatives_.resize(layers, nullptr);
    device_deltas_.resize(layers, nullptr);
    device_back_buffers_.resize(layers, nullptr);
    device_states_.resize(config_.dims.size(), nullptr);
    device_next_states_.resize(config_.dims.size(), nullptr);
    throw_if_cuda_error(
        cudaMalloc(&device_visible_error_precisions_, visible_bytes),
        "cudaMalloc(visible precision)");
    throw_if_cuda_error(
        cudaMemset(device_visible_error_precisions_, 0, visible_bytes),
        "cudaMemset(visible precision)");

    for (std::size_t l = 0; l < config_.dims.size(); ++l) {
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims[l]);
        throw_if_cuda_error(cudaMalloc(&device_states_[l], bytes), "cudaMalloc(state)");
        throw_if_cuda_error(cudaMalloc(&device_next_states_[l], bytes), "cudaMalloc(next_state)");
        throw_if_cuda_error(cudaMemset(device_states_[l], 0, bytes), "cudaMemset(state)");
        throw_if_cuda_error(cudaMemset(device_next_states_[l], 0, bytes), "cudaMemset(next_state)");

        if (l == layers) {
            continue;
        }

        const std::size_t matrix_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l]) * static_cast<std::size_t>(config_.dims[l + 1]);
        throw_if_cuda_error(cudaMalloc(&device_weights_[l], matrix_bytes), "cudaMalloc(weight)");
        throw_if_cuda_error(cudaMalloc(&device_biases_[l], bytes), "cudaMalloc(bias)");
        throw_if_cuda_error(cudaMalloc(&device_weight_velocities_[l], matrix_bytes), "cudaMalloc(weight_velocity)");
        throw_if_cuda_error(cudaMalloc(&device_bias_velocities_[l], bytes), "cudaMalloc(bias_velocity)");
        throw_if_cuda_error(cudaMalloc(&device_predictions_[l], bytes), "cudaMalloc(prediction)");
        throw_if_cuda_error(cudaMalloc(&device_errors_[l], bytes), "cudaMalloc(error)");
        throw_if_cuda_error(cudaMalloc(&device_pre_activations_[l], bytes), "cudaMalloc(pre_activation)");
        throw_if_cuda_error(cudaMalloc(&device_sigmoid_derivatives_[l], bytes), "cudaMalloc(sigmoid_derivative)");
        throw_if_cuda_error(cudaMalloc(&device_deltas_[l], bytes), "cudaMalloc(delta)");
        throw_if_cuda_error(
            cudaMalloc(&device_back_buffers_[l], sizeof(double) * static_cast<std::size_t>(config_.dims[l + 1])),
            "cudaMalloc(back_buffer)");

        throw_if_cuda_error(cudaMemset(device_predictions_[l], 0, bytes), "cudaMemset(prediction)");
        throw_if_cuda_error(cudaMemset(device_biases_[l], 0, bytes), "cudaMemset(bias)");
        throw_if_cuda_error(cudaMemset(device_weight_velocities_[l], 0, matrix_bytes), "cudaMemset(weight_velocity)");
        throw_if_cuda_error(cudaMemset(device_bias_velocities_[l], 0, bytes), "cudaMemset(bias_velocity)");
        throw_if_cuda_error(cudaMemset(device_errors_[l], 0, bytes), "cudaMemset(error)");
        throw_if_cuda_error(cudaMemset(device_pre_activations_[l], 0, bytes), "cudaMemset(pre_activation)");
        throw_if_cuda_error(cudaMemset(device_sigmoid_derivatives_[l], 0, bytes), "cudaMemset(sigmoid_derivative)");
        throw_if_cuda_error(cudaMemset(device_deltas_[l], 0, bytes), "cudaMemset(delta)");
        throw_if_cuda_error(
            cudaMemset(device_back_buffers_[l], 0, sizeof(double) * static_cast<std::size_t>(config_.dims[l + 1])),
            "cudaMemset(back_buffer)");
    }
}

void LTFNCuda::ensure_batch_capacity(int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be positive.");
    }
    if (batch_capacity_ >= batch_size) {
        return;
    }
    release_batch_buffers();
    allocate_batch_buffers(batch_size);
    batch_capacity_ = batch_size;
    host_batch_visible_.resize(config_.dims.front(), batch_size);
}

void LTFNCuda::allocate_batch_buffers(int batch_size) {
    const std::size_t layers = config_.dims.size() - 1;
    device_batch_states_.resize(config_.dims.size(), nullptr);
    device_batch_errors_.resize(layers, nullptr);
    device_batch_pre_activations_.resize(layers, nullptr);
    device_batch_deltas_.resize(layers, nullptr);
    device_batch_back_buffers_.resize(layers, nullptr);
    device_batch_gradients_.resize(layers, nullptr);
    device_batch_bias_gradients_.resize(layers, nullptr);
    device_batch_centered_states_.resize(layers, nullptr);
    device_batch_decor_signals_.resize(layers, nullptr);
    device_batch_covariances_.resize(layers, nullptr);
    device_batch_state_means_.resize(layers, nullptr);

    for (std::size_t l = 0; l < config_.dims.size(); ++l) {
        const std::size_t state_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l]) * static_cast<std::size_t>(batch_size);
        throw_if_cuda_error(cudaMalloc(&device_batch_states_[l], state_bytes), "cudaMalloc(batch state)");
        throw_if_cuda_error(cudaMemset(device_batch_states_[l], 0, state_bytes), "cudaMemset(batch state)");

        if (l == layers) {
            continue;
        }

        const std::size_t layer_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l]) * static_cast<std::size_t>(batch_size);
        const std::size_t back_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l + 1]) * static_cast<std::size_t>(batch_size);
        const std::size_t gradient_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l]) * static_cast<std::size_t>(config_.dims[l + 1]);
        const std::size_t centered_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l + 1]) * static_cast<std::size_t>(batch_size);
        const std::size_t covariance_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l + 1]) * static_cast<std::size_t>(config_.dims[l + 1]);
        const std::size_t mean_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l + 1]);
        const std::size_t bias_gradient_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l]);
        throw_if_cuda_error(cudaMalloc(&device_batch_errors_[l], layer_bytes), "cudaMalloc(batch error)");
        throw_if_cuda_error(cudaMalloc(&device_batch_pre_activations_[l], layer_bytes), "cudaMalloc(batch pre_activation)");
        throw_if_cuda_error(cudaMalloc(&device_batch_deltas_[l], layer_bytes), "cudaMalloc(batch delta)");
        throw_if_cuda_error(cudaMalloc(&device_batch_back_buffers_[l], back_bytes), "cudaMalloc(batch back buffer)");
        throw_if_cuda_error(cudaMalloc(&device_batch_gradients_[l], gradient_bytes), "cudaMalloc(batch gradient)");
        throw_if_cuda_error(
            cudaMalloc(&device_batch_bias_gradients_[l], bias_gradient_bytes),
            "cudaMalloc(batch bias gradient)");
        throw_if_cuda_error(cudaMalloc(&device_batch_centered_states_[l], centered_bytes), "cudaMalloc(batch centered state)");
        throw_if_cuda_error(cudaMalloc(&device_batch_decor_signals_[l], centered_bytes), "cudaMalloc(batch decor signal)");
        throw_if_cuda_error(cudaMalloc(&device_batch_covariances_[l], covariance_bytes), "cudaMalloc(batch covariance)");
        throw_if_cuda_error(cudaMalloc(&device_batch_state_means_[l], mean_bytes), "cudaMalloc(batch state means)");
        throw_if_cuda_error(cudaMemset(device_batch_errors_[l], 0, layer_bytes), "cudaMemset(batch error)");
        throw_if_cuda_error(cudaMemset(device_batch_pre_activations_[l], 0, layer_bytes), "cudaMemset(batch pre_activation)");
        throw_if_cuda_error(cudaMemset(device_batch_deltas_[l], 0, layer_bytes), "cudaMemset(batch delta)");
        throw_if_cuda_error(cudaMemset(device_batch_back_buffers_[l], 0, back_bytes), "cudaMemset(batch back buffer)");
        throw_if_cuda_error(cudaMemset(device_batch_gradients_[l], 0, gradient_bytes), "cudaMemset(batch gradient)");
        throw_if_cuda_error(
            cudaMemset(device_batch_bias_gradients_[l], 0, bias_gradient_bytes),
            "cudaMemset(batch bias gradient)");
        throw_if_cuda_error(cudaMemset(device_batch_centered_states_[l], 0, centered_bytes), "cudaMemset(batch centered state)");
        throw_if_cuda_error(cudaMemset(device_batch_decor_signals_[l], 0, centered_bytes), "cudaMemset(batch decor signal)");
        throw_if_cuda_error(cudaMemset(device_batch_covariances_[l], 0, covariance_bytes), "cudaMemset(batch covariance)");
        throw_if_cuda_error(cudaMemset(device_batch_state_means_[l], 0, mean_bytes), "cudaMemset(batch state means)");
    }

    const std::size_t ones_bytes = sizeof(double) * static_cast<std::size_t>(batch_size);
    throw_if_cuda_error(cudaMalloc(&device_batch_ones_, ones_bytes), "cudaMalloc(batch ones)");
    std::vector<double> host_ones(static_cast<std::size_t>(batch_size), 1.0);
    throw_if_cuda_error(
        cudaMemcpy(device_batch_ones_, host_ones.data(), ones_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(batch ones)");
}

void LTFNCuda::release_buffers() noexcept {
    release_batch_buffers();
    if (device_visible_error_precisions_ != nullptr) {
        cudaFree(device_visible_error_precisions_);
        device_visible_error_precisions_ = nullptr;
    }
    for (double*& ptr : device_bias_velocities_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_weight_velocities_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_back_buffers_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_deltas_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_sigmoid_derivatives_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_pre_activations_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_errors_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_predictions_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_weights_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_biases_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_next_states_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_states_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
}

void LTFNCuda::release_batch_buffers() noexcept {
    if (device_batch_ones_ != nullptr) {
        cudaFree(device_batch_ones_);
        device_batch_ones_ = nullptr;
    }
    for (double*& ptr : device_batch_bias_gradients_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_gradients_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_state_means_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_covariances_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_decor_signals_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_centered_states_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_back_buffers_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_deltas_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_pre_activations_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_errors_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (double*& ptr : device_batch_states_) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    batch_capacity_ = 0;
}

void LTFNCuda::compute_layer_prediction_error_delta(std::size_t layer) const {
    const double alpha = 1.0;
    const double beta = 0.0;
    const int rows = config_.dims[layer];
    const int cols = config_.dims[layer + 1];

    throw_if_cublas_error(
        cublasDgemv(
            cublas_,
            CUBLAS_OP_N,
            rows,
            cols,
            &alpha,
            device_weights_[layer],
            rows,
            device_states_[layer + 1],
            1,
            &beta,
            device_pre_activations_[layer],
            1),
        "cublasDgemv(forward)");

    if (config_.use_biases) {
        add_bias_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(
            device_pre_activations_[layer],
            device_biases_[layer],
            rows,
            1);
        throw_if_cuda_error(cudaGetLastError(), "add_bias_kernel");
    }

    sigmoid_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(
        device_pre_activations_[layer],
        device_predictions_[layer],
        device_sigmoid_derivatives_[layer],
        rows);
    throw_if_cuda_error(cudaGetLastError(), "sigmoid_kernel");

    error_and_delta_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(
        device_states_[layer],
        device_predictions_[layer],
        device_sigmoid_derivatives_[layer],
        device_errors_[layer],
        device_deltas_[layer],
        rows,
        (config_.visible_loss == VisibleLoss::Bce && layer == 0) ? 1 : 0);
    throw_if_cuda_error(cudaGetLastError(), "error_and_delta_kernel");
    if (config_.visible_unit_precision && layer == 0) {
        apply_precision_kernel<<<blocks_for(rows), kThreadsPerBlock>>>(
            device_deltas_[0],
            device_visible_error_precisions_,
            rows,
            1);
        throw_if_cuda_error(cudaGetLastError(), "apply_precision_kernel");
    }
}

void LTFNCuda::compute_predictions_errors_and_deltas() const {
    for (std::size_t l = 0; l < device_weights_.size(); ++l) {
        compute_layer_prediction_error_delta(l);
    }
    predictions_dirty_ = false;
}

void LTFNCuda::compute_batch_layer_error_delta(std::size_t layer, int batch_size) {
    const double alpha = 1.0;
    const double beta = 0.0;
    const int rows = config_.dims[layer];
    const int cols = config_.dims[layer + 1];

    throw_if_cublas_error(
        cublasDgemm(
            cublas_,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            rows,
            batch_size,
            cols,
            &alpha,
            device_weights_[layer],
            rows,
            device_batch_states_[layer + 1],
            cols,
            &beta,
            device_batch_pre_activations_[layer],
            rows),
        "cublasDgemm(batch forward)");

    if (config_.use_biases) {
        const int elements = rows * batch_size;
        add_bias_kernel<<<blocks_for(elements), kThreadsPerBlock>>>(
            device_batch_pre_activations_[layer],
            device_biases_[layer],
            rows,
            batch_size);
        throw_if_cuda_error(cudaGetLastError(), "add_bias_kernel(batch)");
    }

    const int elements = rows * batch_size;
    fused_batch_error_delta_kernel<<<blocks_for(elements), kThreadsPerBlock>>>(
        device_batch_states_[layer],
        device_batch_pre_activations_[layer],
        device_batch_errors_[layer],
        device_batch_deltas_[layer],
        elements,
        (config_.visible_loss == VisibleLoss::Bce && layer == 0) ? 1 : 0);
    throw_if_cuda_error(cudaGetLastError(), "fused_batch_error_delta_kernel");
    if (config_.visible_unit_precision && layer == 0) {
        apply_precision_kernel<<<blocks_for(elements), kThreadsPerBlock>>>(
            device_batch_deltas_[0],
            device_visible_error_precisions_,
            rows,
            batch_size);
        throw_if_cuda_error(cudaGetLastError(), "apply_precision_kernel(batch)");
    }
}

StepDiagnostics LTFNCuda::collect_diagnostics(bool update_weights) {
    StepDiagnostics diagnostics;
    diagnostics.error_norms.reserve(device_errors_.size());
    diagnostics.weight_gradient_norms.reserve(device_weights_.size());
    diagnostics.weight_update_norms.reserve(device_weights_.size());
    diagnostics.weight_norms.reserve(device_weights_.size());

    const double weight_scale = current_learning_rate_ * config_.dt_w;
    const bool custom_rule = use_custom_local_rule(config_);
    for (std::size_t l = 0; l < device_weights_.size(); ++l) {
        const int rows = config_.dims[l];
        const int cols = config_.dims[l + 1];
        const int elements = rows * cols;
        const double error_norm = vector_norm(cublas_, device_errors_[l], config_.dims[l]);
        const double delta_norm = vector_norm(cublas_, device_deltas_[l], config_.dims[l]);
        const double state_norm = vector_norm(cublas_, device_states_[l + 1], config_.dims[l + 1]);
        const double grad_norm = layer_error_precisions_[l] * delta_norm * state_norm;
        double update_norm = weight_scale * grad_norm;

        diagnostics.error_norms.push_back(error_norm);
        diagnostics.weight_gradient_norms.push_back(grad_norm);

        double metric_weight_scale = weight_scale;
        double applied_weight_scale = weight_scale * layer_error_precisions_[l];
        if (custom_rule) {
            const double next_second_moment = blend_second_moment(
                layer_second_moments_[l],
                config_.layer_adapt_beta,
                matrix_mean_square_from_norm(grad_norm, elements));
            if (update_weights) {
                layer_second_moments_[l] = next_second_moment;
            }
            const double denom =
                std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon;
            metric_weight_scale /= denom;
            applied_weight_scale /= denom;
        }

        if (update_weights) {
            if (momentum_beta_ > 0.0) {
                throw_if_cublas_error(
                    cublasDscal(cublas_, elements, &momentum_beta_, device_weight_velocities_[l], 1),
                    "cublasDscal(step weight velocity)");
                throw_if_cublas_error(
                    cublasDger(
                        cublas_,
                        rows,
                        cols,
                        &applied_weight_scale,
                        device_deltas_[l],
                        1,
                        device_states_[l + 1],
                        1,
                        device_weight_velocities_[l],
                        rows),
                    "cublasDger(step weight velocity update)");
                update_norm = matrix_norm(cublas_, device_weight_velocities_[l], elements);
                const double one = 1.0;
                throw_if_cublas_error(
                    cublasDaxpy(
                        cublas_,
                        elements,
                        &one,
                        device_weight_velocities_[l],
                        1,
                        device_weights_[l],
                        1),
                    "cublasDaxpy(step momentum weight update)");
                if (config_.use_biases) {
                    throw_if_cublas_error(
                        cublasDscal(cublas_, rows, &momentum_beta_, device_bias_velocities_[l], 1),
                        "cublasDscal(step bias velocity)");
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            rows,
                            &applied_weight_scale,
                            device_deltas_[l],
                            1,
                            device_bias_velocities_[l],
                            1),
                        "cublasDaxpy(step bias velocity update)");
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            rows,
                            &one,
                            device_bias_velocities_[l],
                            1,
                            device_biases_[l],
                            1),
                        "cublasDaxpy(step bias momentum update)");
                }
            } else {
                update_norm = metric_weight_scale * grad_norm;
                throw_if_cublas_error(
                    cublasDger(
                        cublas_,
                        rows,
                        cols,
                        &applied_weight_scale,
                        device_deltas_[l],
                        1,
                        device_states_[l + 1],
                        1,
                        device_weights_[l],
                        rows),
                    "cublasDger(weight update)");
                if (config_.use_biases) {
                    throw_if_cublas_error(
                        cublasDaxpy(
                            cublas_,
                            rows,
                            &applied_weight_scale,
                            device_deltas_[l],
                            1,
                            device_biases_[l],
                            1),
                        "cublasDaxpy(step bias update)");
                }
            }
        } else if (momentum_beta_ > 0.0) {
            update_norm =
                momentum_beta_ * matrix_norm(cublas_, device_weight_velocities_[l], elements) +
                metric_weight_scale * grad_norm;
        } else {
            update_norm = metric_weight_scale * grad_norm;
        }
        diagnostics.weight_update_norms.push_back(update_norm);

        diagnostics.weight_norms.push_back(matrix_norm(cublas_, device_weights_[l], elements));
    }

    return diagnostics;
}

void LTFNCuda::upload_batch_inputs(const std::vector<const Eigen::VectorXd*>& inputs) {
    const int batch_size = static_cast<int>(inputs.size());
    ensure_batch_capacity(batch_size);
    host_batch_visible_.setZero(config_.dims.front(), batch_size);
    for (int column = 0; column < batch_size; ++column) {
        ensure_input_shape(*inputs[static_cast<std::size_t>(column)]);
        host_batch_visible_.col(column) = *inputs[static_cast<std::size_t>(column)];
    }

    std::vector<Eigen::MatrixXd> host_batch_states;
    host_batch_states.reserve(config_.dims.size());
    host_batch_states.push_back(host_batch_visible_);
    if (config_.state_init == StateInit::Tied) {
        download_all_weights();
        if (config_.use_biases) {
            download_all_biases();
        }
    }
    for (std::size_t l = 1; l < config_.dims.size(); ++l) {
        if (config_.state_init == StateInit::Zero) {
            host_batch_states.emplace_back(Eigen::MatrixXd::Zero(config_.dims[l], batch_size));
        } else {
            Eigen::MatrixXd upward = host_weights_[l - 1].transpose() * host_batch_states[l - 1];
            const Eigen::ArrayXXd clipped = upward.array().max(-500.0).min(500.0);
            host_batch_states.emplace_back((1.0 / (1.0 + (-clipped).exp())).matrix());
        }
    }

    for (std::size_t l = 0; l < config_.dims.size(); ++l) {
        const std::size_t state_bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l]) * static_cast<std::size_t>(batch_size);
        throw_if_cuda_error(
            cudaMemcpy(device_batch_states_[l], host_batch_states[l].data(), state_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(batch state upload)");
    }
}

void LTFNCuda::compute_batch_errors_and_deltas(int batch_size) {
    for (std::size_t l = 0; l < device_weights_.size(); ++l) {
        compute_batch_layer_error_delta(l, batch_size);
    }
}

double LTFNCuda::compute_batch_effective_gradient(std::size_t layer_index, int batch_size) {
    const int rows = config_.dims[layer_index];
    const int cols = config_.dims[layer_index + 1];
    const double batch_scale = 1.0 / static_cast<double>(batch_size);
    const double alpha = batch_scale * layer_error_precisions_[layer_index];
    const double beta = 0.0;
    throw_if_cublas_error(
        cublasDgemm(
            cublas_,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            rows,
            cols,
            batch_size,
            &alpha,
            device_batch_deltas_[layer_index],
            rows,
            device_batch_states_[layer_index + 1],
            cols,
            &beta,
            device_batch_gradients_[layer_index],
            rows),
        "cublasDgemm(batch gradient)");

    if (config_.decorrelation_lambda > 0.0 && batch_size > 1) {
        row_mean_kernel<<<blocks_for(cols), kThreadsPerBlock>>>(
            device_batch_states_[layer_index + 1],
            device_batch_state_means_[layer_index],
            cols,
            batch_size);
        throw_if_cuda_error(cudaGetLastError(), "row_mean_kernel");

        center_rows_kernel<<<blocks_for(cols * batch_size), kThreadsPerBlock>>>(
            device_batch_states_[layer_index + 1],
            device_batch_state_means_[layer_index],
            device_batch_centered_states_[layer_index],
            cols,
            batch_size);
        throw_if_cuda_error(cudaGetLastError(), "center_rows_kernel");

        throw_if_cublas_error(
            cublasDgemm(
                cublas_,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                cols,
                cols,
                batch_size,
                &alpha,
                device_batch_centered_states_[layer_index],
                cols,
                device_batch_centered_states_[layer_index],
                cols,
                &beta,
                device_batch_covariances_[layer_index],
                cols),
            "cublasDgemm(batch covariance)");

        zero_diagonal_kernel<<<blocks_for(cols), kThreadsPerBlock>>>(
            device_batch_covariances_[layer_index],
            cols);
        throw_if_cuda_error(cudaGetLastError(), "zero_diagonal_kernel");

        const double one = 1.0;
        throw_if_cublas_error(
            cublasDgemm(
                cublas_,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                cols,
                batch_size,
                cols,
                &one,
                device_batch_covariances_[layer_index],
                cols,
                device_batch_centered_states_[layer_index],
                cols,
                &beta,
                device_batch_decor_signals_[layer_index],
                cols),
            "cublasDgemm(batch decor signal)");

        const double correction_alpha =
            -config_.decorrelation_lambda * batch_scale * layer_error_precisions_[layer_index];
        const double correction_beta = 1.0;
        throw_if_cublas_error(
            cublasDgemm(
                cublas_,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                rows,
                cols,
                batch_size,
                &correction_alpha,
                device_batch_deltas_[layer_index],
                rows,
                device_batch_decor_signals_[layer_index],
                cols,
                &correction_beta,
                device_batch_gradients_[layer_index],
                rows),
            "cublasDgemm(batch decor correction)");
    }

    return matrix_norm(cublas_, device_batch_gradients_[layer_index], rows * cols);
}

StepDiagnostics LTFNCuda::collect_batch_diagnostics(
    int batch_size,
    const std::vector<double>* state_update_norms,
    const std::vector<double>* gradient_norms,
    const std::vector<double>* update_norms) {
    StepDiagnostics diagnostics;
    diagnostics.state_update_norms = state_update_norms != nullptr
        ? *state_update_norms
        : std::vector<double>(device_weights_.size(), 0.0);
    diagnostics.error_norms.reserve(device_weights_.size());
    diagnostics.weight_gradient_norms.reserve(device_weights_.size());
    diagnostics.weight_update_norms.reserve(device_weights_.size());
    diagnostics.weight_norms.reserve(device_weights_.size());

    const double batch_scale = 1.0 / static_cast<double>(batch_size);
    const double batch_norm_scale = 1.0 / std::sqrt(static_cast<double>(batch_size));
    const double weight_scale = current_learning_rate_ * config_.dt_w;
    const bool custom_rule = use_custom_local_rule(config_);
    double visible_error_norm = 0.0;
    double visible_bce_energy = 0.0;

    if (config_.visible_loss == VisibleLoss::Bce) {
        Eigen::MatrixXd host_state(config_.dims.front(), batch_size);
        Eigen::MatrixXd host_error(config_.dims.front(), batch_size);
        const std::size_t bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims.front()) * static_cast<std::size_t>(batch_size);
        throw_if_cuda_error(
            cudaMemcpy(host_state.data(), device_batch_states_[0], bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(batch visible state for BCE energy)");
        throw_if_cuda_error(
            cudaMemcpy(host_error.data(), device_batch_errors_[0], bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(batch visible error for BCE energy)");
        for (Eigen::Index col = 0; col < host_state.cols(); ++col) {
            for (Eigen::Index row = 0; row < host_state.rows(); ++row) {
                const double precision = config_.visible_unit_precision
                    ? visible_error_precisions_(row)
                    : layer_error_precisions_[0];
                visible_bce_energy += precision *
                    binary_cross_entropy_scalar(host_state(row, col), host_state(row, col) - host_error(row, col));
            }
        }
    }

    for (std::size_t l = 0; l < device_weights_.size(); ++l) {
        const int rows = config_.dims[l];
        const int cols = config_.dims[l + 1];
        const double error_norm_total =
            matrix_norm(cublas_, device_batch_errors_[l], rows * batch_size);
        const double error_norm = error_norm_total * batch_norm_scale;
        if (l == 0) {
            visible_error_norm = error_norm_total;
        }

        diagnostics.error_norms.push_back(error_norm);
        if (config_.visible_loss == VisibleLoss::Bce && l == 0) {
            diagnostics.energy += visible_bce_energy * batch_scale;
        } else if (config_.visible_unit_precision && l == 0) {
            Eigen::MatrixXd host_error(config_.dims.front(), batch_size);
            const std::size_t bytes =
                sizeof(double) * static_cast<std::size_t>(config_.dims.front()) * static_cast<std::size_t>(batch_size);
            throw_if_cuda_error(
                cudaMemcpy(host_error.data(), device_batch_errors_[0], bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(batch visible error for weighted diagnostics)");
            diagnostics.energy +=
                0.5 * (host_error.array().square().colwise() * visible_error_precisions_.array()).sum() * batch_scale;
        } else {
            diagnostics.energy +=
                layer_error_precisions_[l] * 0.5 * error_norm_total * error_norm_total * batch_scale;
        }

        double gradient_norm = 0.0;
        if (gradient_norms != nullptr) {
            gradient_norm = (*gradient_norms)[l];
        } else {
            gradient_norm = compute_batch_effective_gradient(l, batch_size);
        }
        diagnostics.weight_gradient_norms.push_back(gradient_norm);
        if (update_norms != nullptr) {
            diagnostics.weight_update_norms.push_back((*update_norms)[l]);
        } else {
            double update_norm = weight_scale * gradient_norm;
            if (custom_rule) {
                double scaled_weight_scale = weight_scale;
                if (config_.layer_adapt_beta > 0.0) {
                    const double next_second_moment = blend_second_moment(
                        layer_second_moments_[l],
                        config_.layer_adapt_beta,
                        matrix_mean_square_from_norm(gradient_norm, rows * cols));
                    scaled_weight_scale /=
                        (std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon);
                }
                if (momentum_beta_ > 0.0) {
                    update_norm =
                        momentum_beta_ * matrix_norm(cublas_, device_weight_velocities_[l], rows * cols) +
                        scaled_weight_scale * gradient_norm;
                } else {
                    update_norm = scaled_weight_scale * gradient_norm;
                }
            }
            diagnostics.weight_update_norms.push_back(update_norm);
        }
        diagnostics.weight_norms.push_back(matrix_norm(cublas_, device_weights_[l], rows * cols));
    }

    diagnostics.mse = static_cast<double>(config_.dims.front()) > 0
        ? (visible_error_norm * visible_error_norm /
              (static_cast<double>(config_.dims.front()) * static_cast<double>(batch_size)))
        : 0.0;
    return diagnostics;
}

void LTFNCuda::upload_all_weights() {
    for (std::size_t l = 0; l < host_weights_.size(); ++l) {
        const std::size_t bytes =
            sizeof(double) * static_cast<std::size_t>(host_weights_[l].rows() * host_weights_[l].cols());
        throw_if_cuda_error(
            cudaMemcpy(device_weights_[l], host_weights_[l].data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(weight upload)");
    }
}

void LTFNCuda::upload_all_biases() {
    for (std::size_t l = 0; l < host_biases_.size(); ++l) {
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(host_biases_[l].size());
        throw_if_cuda_error(
            cudaMemcpy(device_biases_[l], host_biases_[l].data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(bias upload)");
    }
}

void LTFNCuda::download_all_weights() const {
    for (std::size_t l = 0; l < host_weights_.size(); ++l) {
        const std::size_t bytes =
            sizeof(double) * static_cast<std::size_t>(host_weights_[l].rows() * host_weights_[l].cols());
        throw_if_cuda_error(
            cudaMemcpy(host_weights_[l].data(), device_weights_[l], bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(weight download)");
    }
}

void LTFNCuda::download_all_biases() const {
    for (std::size_t l = 0; l < host_biases_.size(); ++l) {
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(host_biases_[l].size());
        throw_if_cuda_error(
            cudaMemcpy(host_biases_[l].data(), device_biases_[l], bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(bias download)");
    }
}

void LTFNCuda::download_all_states() const {
    for (std::size_t l = 0; l < host_states_.size(); ++l) {
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(host_states_[l].size());
        throw_if_cuda_error(
            cudaMemcpy(host_states_[l].data(), device_states_[l], bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(state download)");
    }
}

void LTFNCuda::download_reconstruction() const {
    throw_if_cuda_error(
        cudaMemcpy(
            host_reconstruction_.data(),
            device_predictions_.front(),
            sizeof(double) * static_cast<std::size_t>(host_reconstruction_.size()),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(reconstruction download)");
}

void LTFNCuda::zero_weight_velocities() {
    for (std::size_t l = 0; l < device_weight_velocities_.size(); ++l) {
        const std::size_t bytes =
            sizeof(double) * static_cast<std::size_t>(config_.dims[l]) * static_cast<std::size_t>(config_.dims[l + 1]);
        throw_if_cuda_error(cudaMemset(device_weight_velocities_[l], 0, bytes), "cudaMemset(weight velocity)");
    }
}

void LTFNCuda::zero_bias_velocities() {
    for (std::size_t l = 0; l < device_bias_velocities_.size(); ++l) {
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims[l]);
        throw_if_cuda_error(cudaMemset(device_bias_velocities_[l], 0, bytes), "cudaMemset(bias velocity)");
    }
}

void LTFNCuda::zero_latent_states() {
    for (std::size_t l = 1; l < config_.dims.size(); ++l) {
        const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(config_.dims[l]);
        throw_if_cuda_error(cudaMemset(device_states_[l], 0, bytes), "cudaMemset(latent state)");
        throw_if_cuda_error(cudaMemset(device_next_states_[l], 0, bytes), "cudaMemset(next latent state)");
    }
}

void LTFNCuda::synchronize() const {
    throw_if_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

}  // namespace ltfn

#endif
