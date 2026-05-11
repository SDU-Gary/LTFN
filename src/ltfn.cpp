#include "ltfn.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace ltfn {

namespace {

double clamp_sigmoid_input(double value) {
    return std::max(-500.0, std::min(500.0, value));
}

double sigmoid_scalar(double value) {
    return 1.0 / (1.0 + std::exp(-clamp_sigmoid_input(value)));
}

double sigmoid_derivative_from_pre_activation(double value) {
    const double s = sigmoid_scalar(value);
    return s * (1.0 - s);
}

double clamp_probability(double value) {
    return std::max(1e-12, std::min(1.0 - 1e-12, value));
}

double binary_cross_entropy_scalar(double target, double prediction) {
    const double p = clamp_probability(prediction);
    return -(target * std::log(p) + (1.0 - target) * std::log(1.0 - p));
}

Eigen::VectorXd sigmoid_vec(const Eigen::VectorXd& values) {
    return values.unaryExpr([](double value) { return sigmoid_scalar(value); });
}

Eigen::VectorXd sigmoid_derivative_vec(const Eigen::VectorXd& values) {
    return values.unaryExpr([](double value) { return sigmoid_derivative_from_pre_activation(value); });
}

Eigen::MatrixXd sigmoid_matrix(const Eigen::MatrixXd& values) {
    const Eigen::ArrayXXd clipped = values.array().max(-500.0).min(500.0);
    return (1.0 / (1.0 + (-clipped).exp())).matrix();
}

Eigen::VectorXd layer_delta_vec(
    const Eigen::VectorXd& errors,
    const Eigen::VectorXd& sigmoid_derivatives,
    bool use_bce_delta) {
    return use_bce_delta ? errors : errors.cwiseProduct(sigmoid_derivatives);
}

double binary_cross_entropy_energy(
    const Eigen::VectorXd& target,
    const Eigen::VectorXd& prediction) {
    double energy = 0.0;
    for (Eigen::Index i = 0; i < target.size(); ++i) {
        energy += binary_cross_entropy_scalar(target(i), prediction(i));
    }
    return energy;
}

double binary_cross_entropy_energy(
    const Eigen::MatrixXd& target,
    const Eigen::MatrixXd& prediction) {
    double energy = 0.0;
    for (Eigen::Index col = 0; col < target.cols(); ++col) {
        for (Eigen::Index row = 0; row < target.rows(); ++row) {
            energy += binary_cross_entropy_scalar(target(row, col), prediction(row, col));
        }
    }
    return energy;
}

double weighted_binary_cross_entropy_energy(
    const Eigen::VectorXd& target,
    const Eigen::VectorXd& prediction,
    const Eigen::VectorXd& precisions) {
    double energy = 0.0;
    for (Eigen::Index i = 0; i < target.size(); ++i) {
        energy += precisions(i) * binary_cross_entropy_scalar(target(i), prediction(i));
    }
    return energy;
}

double weighted_binary_cross_entropy_energy(
    const Eigen::MatrixXd& target,
    const Eigen::MatrixXd& prediction,
    const Eigen::VectorXd& precisions) {
    double energy = 0.0;
    for (Eigen::Index col = 0; col < target.cols(); ++col) {
        for (Eigen::Index row = 0; row < target.rows(); ++row) {
            energy += precisions(row) * binary_cross_entropy_scalar(target(row, col), prediction(row, col));
        }
    }
    return energy;
}

double blend_second_moment(double previous, double beta, double mean_square) {
    return beta * previous + (1.0 - beta) * mean_square;
}

double matrix_mean_square(const Eigen::MatrixXd& matrix) {
    if (matrix.size() == 0) {
        return 0.0;
    }
    return matrix.squaredNorm() / static_cast<double>(matrix.size());
}

Eigen::MatrixXd compute_decorrelation_signal(const Eigen::MatrixXd& batch_states) {
    if (batch_states.cols() <= 1) {
        return Eigen::MatrixXd::Zero(batch_states.rows(), batch_states.cols());
    }
    const Eigen::VectorXd mean = batch_states.rowwise().mean();
    const Eigen::MatrixXd centered = batch_states.colwise() - mean;
    Eigen::MatrixXd covariance =
        centered * centered.transpose() / static_cast<double>(batch_states.cols());
    covariance.diagonal().setZero();
    return covariance * centered;
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

}  // namespace

double compute_mse(const Eigen::VectorXd& expected, const Eigen::VectorXd& actual) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("MSE vectors must have identical shapes.");
    }
    return (expected - actual).squaredNorm() / static_cast<double>(expected.size());
}

LTFN::LTFN(const LTFNConfig& config, std::uint32_t seed)
    : config_(config) {
    validate_dims();
    current_learning_rate_ = config_.lr_w;

    std::mt19937 generator(seed);
    const std::size_t layers = config_.dims.size() - 1;

    weights_.reserve(layers);
    weight_velocities_.reserve(layers);
    bias_velocities_.reserve(layers);
    layer_second_moments_.reserve(layers);
    layer_error_second_moments_.reserve(layers);
    layer_error_precisions_.reserve(layers);
    states_.reserve(config_.dims.size());
    predictions_.reserve(layers);
    errors_.reserve(layers);
    pre_activations_.reserve(layers);
    sigmoid_derivatives_.reserve(layers);
    visible_error_second_moments_ = Eigen::VectorXd::Zero(config_.dims.front());
    visible_error_precisions_ = Eigen::VectorXd::Ones(config_.dims.front());

    for (std::size_t l = 0; l < config_.dims.size(); ++l) {
        states_.emplace_back(Eigen::VectorXd::Zero(config_.dims[l]));
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
        weights_.push_back(std::move(weight));
        weight_velocities_.emplace_back(Eigen::MatrixXd::Zero(n_out, n_in));
        biases_.emplace_back(Eigen::VectorXd::Zero(n_out));
        bias_velocities_.emplace_back(Eigen::VectorXd::Zero(n_out));
        layer_second_moments_.push_back(0.0);
        layer_error_second_moments_.push_back(0.0);
        layer_error_precisions_.push_back(1.0);

        predictions_.emplace_back(Eigen::VectorXd::Zero(n_out));
        errors_.emplace_back(Eigen::VectorXd::Zero(n_out));
        pre_activations_.emplace_back(Eigen::VectorXd::Zero(n_out));
        sigmoid_derivatives_.emplace_back(Eigen::VectorXd::Zero(n_out));
    }
}

const LTFNConfig& LTFN::config() const noexcept {
    return config_;
}

const std::vector<Eigen::MatrixXd>& LTFN::weights() const {
    return weights_;
}

const std::vector<Eigen::VectorXd>& LTFN::biases() const {
    return biases_;
}

std::vector<Eigen::MatrixXd>& LTFN::mutable_weights() noexcept {
    return weights_;
}

void LTFN::set_weights(const std::vector<Eigen::MatrixXd>& new_weights) {
    if (new_weights.size() != weights_.size()) {
        throw std::invalid_argument("Weight count does not match the configured architecture.");
    }
    for (std::size_t l = 0; l < new_weights.size(); ++l) {
        if (new_weights[l].rows() != weights_[l].rows() || new_weights[l].cols() != weights_[l].cols()) {
            throw std::invalid_argument("Checkpoint weight shape does not match the configured architecture.");
        }
    }
    weights_ = new_weights;
    weight_velocities_.resize(weights_.size());
    for (std::size_t l = 0; l < weights_.size(); ++l) {
        weight_velocities_[l] = Eigen::MatrixXd::Zero(weights_[l].rows(), weights_[l].cols());
    }
    bias_velocities_.resize(biases_.size());
    for (std::size_t l = 0; l < bias_velocities_.size(); ++l) {
        bias_velocities_[l] = Eigen::VectorXd::Zero(biases_[l].size());
    }
    layer_second_moments_.assign(weights_.size(), 0.0);
    layer_error_second_moments_.assign(weights_.size(), 0.0);
    layer_error_precisions_.assign(weights_.size(), 1.0);
    visible_error_second_moments_.setZero();
    visible_error_precisions_.setOnes();
    predictions_dirty_ = true;
}

void LTFN::set_biases(const std::vector<Eigen::VectorXd>& new_biases) {
    if (new_biases.size() != biases_.size()) {
        throw std::invalid_argument("Bias count does not match the configured architecture.");
    }
    for (std::size_t l = 0; l < new_biases.size(); ++l) {
        if (new_biases[l].size() != biases_[l].size()) {
            throw std::invalid_argument("Checkpoint bias shape does not match the configured architecture.");
        }
    }

    biases_ = new_biases;
    bias_velocities_.resize(biases_.size());
    for (std::size_t l = 0; l < bias_velocities_.size(); ++l) {
        bias_velocities_[l] = Eigen::VectorXd::Zero(biases_[l].size());
    }
    predictions_dirty_ = true;
}

void LTFN::set_learning_rate(double learning_rate) {
    if (learning_rate < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative.");
    }
    current_learning_rate_ = learning_rate;
}

void LTFN::set_weight_momentum(double momentum_beta) {
    if (momentum_beta < 0.0 || momentum_beta >= 1.0) {
        throw std::invalid_argument("Momentum beta must be in [0, 1).");
    }
    momentum_beta_ = momentum_beta;
}

void LTFN::reset_states(const Eigen::VectorXd& input) {
    ensure_input_shape(input);
    states_[0] = input;
    initialize_latent_states();
    compute_predictions_and_errors();
}

void LTFN::advance(const Eigen::VectorXd& input, bool update_weights) {
    ensure_input_shape(input);
    states_[0] = input;
    predictions_dirty_ = true;
    advance_current(update_weights);
}

void LTFN::advance_current(bool update_weights) {
    ensure_predictions_current();
    const auto precision_weighted_delta = [&](std::size_t layer) -> Eigen::VectorXd {
        const Eigen::VectorXd delta = layer_delta_vec(
            errors_[layer],
            sigmoid_derivatives_[layer],
            config_.visible_loss == VisibleLoss::Bce && layer == 0);
        if (config_.visible_unit_precision && layer == 0) {
            return delta.cwiseProduct(visible_error_precisions_).eval();
        }
        return (layer_error_precisions_[layer] * delta).eval();
    };
    const double state_scale = config_.dt_r / config_.tau_r;
    const std::size_t top_index = states_.size() - 1;

    if (config_.sequential_inference) {
        {
            const Eigen::VectorXd delta_top = precision_weighted_delta(top_index - 1);
            states_[top_index] += (weights_[top_index - 1].transpose() * delta_top) * state_scale;
            compute_layer_prediction_error(top_index - 1);
        }

        for (std::size_t l = top_index - 1; l >= 1; --l) {
            const Eigen::VectorXd back = weights_[l - 1].transpose() * precision_weighted_delta(l - 1);
            states_[l] += state_scale * (back - (layer_error_precisions_[l] * errors_[l]));
            compute_layer_prediction_error(l - 1);
            if (l == 1) {
                break;
            }
        }
    } else {
        {
            const Eigen::VectorXd delta_top = precision_weighted_delta(top_index - 1);
            states_[top_index] += (weights_[top_index - 1].transpose() * delta_top) * state_scale;
        }

        for (std::size_t l = top_index - 1; l >= 1; --l) {
            const Eigen::VectorXd back = weights_[l - 1].transpose() * precision_weighted_delta(l - 1);
            states_[l] -= state_scale * ((layer_error_precisions_[l] * errors_[l]) - back);
            if (l == 1) {
                break;
            }
        }
    }

    if (update_weights) {
        predictions_dirty_ = true;
        compute_predictions_and_errors();
        update_error_precisions_from_errors();
        const double weight_scale = current_learning_rate_ * config_.dt_w;
        for (std::size_t l = 0; l < errors_.size(); ++l) {
            const Eigen::VectorXd delta_w = precision_weighted_delta(l);
            const Eigen::MatrixXd gradient = delta_w * states_[l + 1].transpose();
            const Eigen::MatrixXd effective_gradient = gradient;
            const Eigen::VectorXd bias_gradient = delta_w;
            double layer_scale = 1.0;
            if (config_.layer_adapt_beta > 0.0) {
                const double next_second_moment = blend_second_moment(
                    layer_second_moments_[l],
                    config_.layer_adapt_beta,
                    matrix_mean_square(effective_gradient));
                layer_second_moments_[l] = next_second_moment;
                layer_scale = 1.0 / (std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon);
            }
            const Eigen::MatrixXd scaled_gradient = layer_scale * effective_gradient;
            const Eigen::VectorXd scaled_bias_gradient = layer_scale * bias_gradient;
            if (momentum_beta_ > 0.0) {
                weight_velocities_[l] *= momentum_beta_;
                weight_velocities_[l].noalias() += weight_scale * scaled_gradient;
                bias_velocities_[l] *= momentum_beta_;
                bias_velocities_[l].noalias() += weight_scale * scaled_bias_gradient;
                weights_[l] += weight_velocities_[l];
                if (config_.use_biases) {
                    biases_[l] += bias_velocities_[l];
                }
            } else {
                weights_[l].noalias() += weight_scale * scaled_gradient;
                if (config_.use_biases) {
                    biases_[l].noalias() += weight_scale * scaled_bias_gradient;
                }
            }
        }
        predictions_dirty_ = true;
    } else {
        predictions_dirty_ = true;
    }
}

StepDiagnostics LTFN::step(const Eigen::VectorXd& input, bool update_weights) {
    ensure_input_shape(input);
    states_[0] = input;
    predictions_dirty_ = true;
    return step_current(update_weights);
}

StepDiagnostics LTFN::step_current(bool update_weights) {
    ensure_predictions_current();
    const auto precision_weighted_delta = [&](std::size_t layer) -> Eigen::VectorXd {
        const Eigen::VectorXd delta = layer_delta_vec(
            errors_[layer],
            sigmoid_derivatives_[layer],
            config_.visible_loss == VisibleLoss::Bce && layer == 0);
        if (config_.visible_unit_precision && layer == 0) {
            return delta.cwiseProduct(visible_error_precisions_).eval();
        }
        return (layer_error_precisions_[layer] * delta).eval();
    };
    std::vector<double> state_update_norms(weights_.size(), 0.0);
    const double state_scale = config_.dt_r / config_.tau_r;
    const std::size_t top_index = states_.size() - 1;

    if (config_.sequential_inference) {
        {
            const Eigen::VectorXd delta_top = precision_weighted_delta(top_index - 1);
            const Eigen::VectorXd update = (weights_[top_index - 1].transpose() * delta_top) * state_scale;
            state_update_norms[top_index - 1] = update.norm();
            states_[top_index] += update;
            compute_layer_prediction_error(top_index - 1);
        }

        for (std::size_t l = top_index - 1; l >= 1; --l) {
            const Eigen::VectorXd back = weights_[l - 1].transpose() * precision_weighted_delta(l - 1);
            const Eigen::VectorXd update =
                state_scale * (back - (layer_error_precisions_[l] * errors_[l]));
            state_update_norms[l - 1] = update.norm();
            states_[l] += update;
            compute_layer_prediction_error(l - 1);
            if (l == 1) {
                break;
            }
        }
    } else {
        std::vector<Eigen::VectorXd> next_states = states_;

        {
            const Eigen::VectorXd delta_top = precision_weighted_delta(top_index - 1);
            next_states[top_index] += (weights_[top_index - 1].transpose() * delta_top) * state_scale;
        }

        for (std::size_t l = top_index - 1; l >= 1; --l) {
            const Eigen::VectorXd back = weights_[l - 1].transpose() * precision_weighted_delta(l - 1);
            const Eigen::VectorXd grad = (layer_error_precisions_[l] * errors_[l]) - back;
            next_states[l] -= state_scale * grad;
            if (l == 1) {
                break;
            }
        }

        for (std::size_t l = 1; l < states_.size(); ++l) {
            state_update_norms[l - 1] = (next_states[l] - states_[l]).norm();
            states_[l] = next_states[l];
        }
    }

    compute_predictions_and_errors();
    if (update_weights) {
        update_error_precisions_from_errors();
    }

    StepDiagnostics diagnostics;
    diagnostics.error_norms.reserve(errors_.size());
    diagnostics.state_update_norms = state_update_norms;
    diagnostics.weight_gradient_norms.reserve(weights_.size());
    diagnostics.weight_update_norms.reserve(weights_.size());
    diagnostics.weight_norms.reserve(weights_.size());

    const double weight_scale = current_learning_rate_ * config_.dt_w;
    for (std::size_t l = 0; l < errors_.size(); ++l) {
        diagnostics.error_norms.push_back(errors_[l].norm());
        const Eigen::VectorXd delta_w = precision_weighted_delta(l);
        const Eigen::MatrixXd dW = delta_w * states_[l + 1].transpose();
        diagnostics.weight_gradient_norms.push_back(dW.norm());
        const Eigen::MatrixXd effective_gradient = dW;
        const Eigen::VectorXd bias_gradient = delta_w;
        double layer_scale = 1.0;
        if (config_.layer_adapt_beta > 0.0) {
            const double next_second_moment = blend_second_moment(
                layer_second_moments_[l],
                config_.layer_adapt_beta,
                matrix_mean_square(effective_gradient));
            if (update_weights) {
                layer_second_moments_[l] = next_second_moment;
            }
            layer_scale = 1.0 / (std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon);
        }
        const Eigen::MatrixXd scaled_gradient = layer_scale * effective_gradient;
        const Eigen::VectorXd scaled_bias_gradient = layer_scale * bias_gradient;
        double update_norm = weight_scale * scaled_gradient.norm();
        if (update_weights) {
            if (momentum_beta_ > 0.0) {
                weight_velocities_[l] *= momentum_beta_;
                weight_velocities_[l].noalias() += weight_scale * scaled_gradient;
                bias_velocities_[l] *= momentum_beta_;
                bias_velocities_[l].noalias() += weight_scale * scaled_bias_gradient;
                update_norm = weight_velocities_[l].norm();
                weights_[l] += weight_velocities_[l];
                if (config_.use_biases) {
                    biases_[l] += bias_velocities_[l];
                }
            } else {
                weights_[l] += weight_scale * scaled_gradient;
                if (config_.use_biases) {
                    biases_[l].noalias() += weight_scale * scaled_bias_gradient;
                }
            }
        } else if (momentum_beta_ > 0.0) {
            const Eigen::MatrixXd tentative_update =
                momentum_beta_ * weight_velocities_[l] + weight_scale * scaled_gradient;
            update_norm = tentative_update.norm();
        }
        diagnostics.weight_update_norms.push_back(update_norm);
        diagnostics.weight_norms.push_back(weights_[l].norm());
    }

    if (update_weights) {
        compute_predictions_and_errors();
    }

    diagnostics.energy = current_energy();
    diagnostics.mse = ltfn::compute_mse(states_[0], predictions_[0]);
    return diagnostics;
}

StepDiagnostics LTFN::current_diagnostics() const {
    ensure_predictions_current();
    const auto precision_weighted_delta = [&](std::size_t layer) -> Eigen::VectorXd {
        const Eigen::VectorXd delta = layer_delta_vec(
            errors_[layer],
            sigmoid_derivatives_[layer],
            config_.visible_loss == VisibleLoss::Bce && layer == 0);
        if (config_.visible_unit_precision && layer == 0) {
            return delta.cwiseProduct(visible_error_precisions_).eval();
        }
        return (layer_error_precisions_[layer] * delta).eval();
    };
    StepDiagnostics diagnostics;
    diagnostics.error_norms.reserve(errors_.size());
    diagnostics.state_update_norms.assign(weights_.size(), 0.0);
    diagnostics.weight_gradient_norms.reserve(weights_.size());
    diagnostics.weight_update_norms.reserve(weights_.size());
    diagnostics.weight_norms.reserve(weights_.size());

    const double weight_scale = current_learning_rate_ * config_.dt_w;
    for (std::size_t l = 0; l < errors_.size(); ++l) {
        const double error_norm = errors_[l].norm();
        diagnostics.error_norms.push_back(error_norm);
        const Eigen::VectorXd delta_w = precision_weighted_delta(l);
        const Eigen::MatrixXd dW = delta_w * states_[l + 1].transpose();
        diagnostics.weight_gradient_norms.push_back(dW.norm());
        const Eigen::MatrixXd effective_gradient = dW;
        double layer_scale = 1.0;
        if (config_.layer_adapt_beta > 0.0) {
            const double next_second_moment = blend_second_moment(
                layer_second_moments_[l],
                config_.layer_adapt_beta,
                matrix_mean_square(effective_gradient));
            layer_scale = 1.0 / (std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon);
        }
        const Eigen::MatrixXd scaled_gradient = layer_scale * effective_gradient;
        if (momentum_beta_ > 0.0) {
            const Eigen::MatrixXd tentative_update =
                momentum_beta_ * weight_velocities_[l] + weight_scale * scaled_gradient;
            diagnostics.weight_update_norms.push_back(tentative_update.norm());
        } else {
            diagnostics.weight_update_norms.push_back(weight_scale * scaled_gradient.norm());
        }
        diagnostics.weight_norms.push_back(weights_[l].norm());
        if (config_.visible_loss == VisibleLoss::Bce && l == 0) {
            diagnostics.energy += config_.visible_unit_precision
                ? weighted_binary_cross_entropy_energy(states_[0], predictions_[0], visible_error_precisions_)
                : layer_error_precisions_[l] * binary_cross_entropy_energy(states_[0], predictions_[0]);
        } else if (config_.visible_unit_precision && l == 0) {
            diagnostics.energy +=
                0.5 * (visible_error_precisions_.array() * errors_[0].array().square()).sum();
        } else {
            diagnostics.energy += layer_error_precisions_[l] * 0.5 * error_norm * error_norm;
        }
    }
    diagnostics.mse = static_cast<double>(config_.dims.front()) > 0
        ? (diagnostics.error_norms.front() * diagnostics.error_norms.front() /
              static_cast<double>(config_.dims.front()))
        : 0.0;
    return diagnostics;
}

BatchTrainResult LTFN::train_batch(
    const std::vector<const Eigen::VectorXd*>& inputs,
    int steps,
    const BatchTrainOptions& options) {
    if (inputs.empty()) {
        throw std::invalid_argument("train_batch requires at least one input sample.");
    }
    if (steps < 0) {
        throw std::invalid_argument("train_batch steps must be non-negative.");
    }

    const std::size_t batch_size = inputs.size();
    const double batch_scale = 1.0 / static_cast<double>(batch_size);
    const double batch_norm_scale = 1.0 / std::sqrt(static_cast<double>(batch_size));
    const double weight_scale = current_learning_rate_ * config_.dt_w;
    const double state_scale = config_.dt_r / config_.tau_r;
    const std::size_t layer_count = weights_.size();
    const std::size_t state_count = states_.size();

    std::vector<int> logged_steps = options.logged_relax_steps;
    logged_steps.erase(
        std::remove_if(logged_steps.begin(), logged_steps.end(), [steps](int step) { return step < 0 || step > steps; }),
        logged_steps.end());
    std::sort(logged_steps.begin(), logged_steps.end());
    logged_steps.erase(std::unique(logged_steps.begin(), logged_steps.end()), logged_steps.end());

    std::vector<Eigen::MatrixXd> batch_states;
    std::vector<Eigen::MatrixXd> batch_errors;
    std::vector<Eigen::MatrixXd> batch_deltas;
    std::vector<Eigen::MatrixXd> batch_pre_activations;
    batch_states.reserve(state_count);
    batch_errors.reserve(layer_count);
    batch_deltas.reserve(layer_count);
    batch_pre_activations.reserve(layer_count);

    for (std::size_t l = 0; l < state_count; ++l) {
        batch_states.emplace_back(Eigen::MatrixXd::Zero(config_.dims[l], static_cast<Eigen::Index>(batch_size)));
        if (l == layer_count) {
            continue;
        }
        batch_errors.emplace_back(Eigen::MatrixXd::Zero(config_.dims[l], static_cast<Eigen::Index>(batch_size)));
        batch_deltas.emplace_back(Eigen::MatrixXd::Zero(config_.dims[l], static_cast<Eigen::Index>(batch_size)));
        batch_pre_activations.emplace_back(Eigen::MatrixXd::Zero(config_.dims[l], static_cast<Eigen::Index>(batch_size)));
    }

    for (std::size_t index = 0; index < batch_size; ++index) {
        ensure_input_shape(*inputs[index]);
        batch_states[0].col(static_cast<Eigen::Index>(index)) = *inputs[index];
    }

    if (config_.state_init == StateInit::Tied) {
        for (std::size_t l = 1; l < state_count; ++l) {
            batch_states[l].noalias() = weights_[l - 1].transpose() * batch_states[l - 1];
            batch_states[l] = sigmoid_matrix(batch_states[l]);
        }
    }

    auto compute_batch_layer_error_and_delta = [&](std::size_t layer) {
        batch_pre_activations[layer].noalias() = weights_[layer] * batch_states[layer + 1];
        if (config_.use_biases) {
            batch_pre_activations[layer].colwise() += biases_[layer];
        }
        const Eigen::MatrixXd sigma = sigmoid_matrix(batch_pre_activations[layer]);
        batch_errors[layer] = batch_states[layer] - sigma;
        if (config_.visible_loss == VisibleLoss::Bce && layer == 0) {
            batch_deltas[layer] = batch_errors[layer];
        } else {
            batch_deltas[layer] =
                (batch_errors[layer].array() * sigma.array() * (1.0 - sigma.array())).matrix();
        }
        if (config_.visible_unit_precision && layer == 0) {
            batch_deltas[layer].array().colwise() *= visible_error_precisions_.array();
        }
    };

    auto compute_batch_errors_and_deltas = [&]() {
        for (std::size_t l = 0; l < layer_count; ++l) {
            compute_batch_layer_error_and_delta(l);
        }
    };

    auto update_error_precisions_from_batch_errors = [&]() {
        if (config_.error_precision_beta <= 0.0) {
            return;
        }

        std::size_t start_layer = 0;
        if (config_.visible_unit_precision) {
            const Eigen::ArrayXd mean_square = batch_errors[0].array().square().rowwise().mean();
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
            layer_error_precisions_[0] = 1.0;
            start_layer = 1;
        }

        for (std::size_t l = start_layer; l < layer_count; ++l) {
            const double mean_square =
                batch_errors[l].squaredNorm() /
                static_cast<double>(batch_errors[l].rows() * batch_errors[l].cols());
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
        } else if (layer_count > 1) {
            std::vector<double> hidden_precisions(
                layer_error_precisions_.begin() + 1,
                layer_error_precisions_.end());
            normalize_precision_vector(
                hidden_precisions,
                config_.error_precision_min,
                config_.error_precision_max);
            std::copy(hidden_precisions.begin(), hidden_precisions.end(), layer_error_precisions_.begin() + 1);
        }
    };

    auto collect_batch_diagnostics =
        [&](const std::vector<double>& state_update_norms,
            const std::vector<double>* gradient_norms_override,
            const std::vector<double>* update_norms_override) -> StepDiagnostics {
        StepDiagnostics diagnostics;
        diagnostics.state_update_norms = state_update_norms;
        diagnostics.error_norms.reserve(layer_count);
        diagnostics.weight_gradient_norms.reserve(layer_count);
        diagnostics.weight_update_norms.reserve(layer_count);
        diagnostics.weight_norms.reserve(layer_count);

        for (std::size_t l = 0; l < layer_count; ++l) {
            const double error_norm = batch_errors[l].norm() * batch_norm_scale;
            diagnostics.error_norms.push_back(error_norm);
            if (config_.visible_loss == VisibleLoss::Bce && l == 0) {
                diagnostics.energy += config_.visible_unit_precision
                    ? weighted_binary_cross_entropy_energy(
                        batch_states[0],
                        batch_states[0] - batch_errors[0],
                        visible_error_precisions_) * batch_scale
                    : layer_error_precisions_[l] *
                        binary_cross_entropy_energy(batch_states[0], batch_states[0] - batch_errors[0]) * batch_scale;
            } else if (config_.visible_unit_precision && l == 0) {
                diagnostics.energy += 0.5 *
                    (batch_errors[0].array().square().colwise() * visible_error_precisions_.array()).sum() *
                    batch_scale;
            } else {
                diagnostics.energy +=
                    layer_error_precisions_[l] * 0.5 * batch_errors[l].squaredNorm() * batch_scale;
            }

            double gradient_norm = 0.0;
            double update_norm = 0.0;
            if (gradient_norms_override != nullptr) {
                gradient_norm = (*gradient_norms_override)[l];
            } else {
                Eigen::MatrixXd gradient =
                    (layer_error_precisions_[l] * batch_deltas[l]) * batch_states[l + 1].transpose();
                gradient *= batch_scale;
                Eigen::MatrixXd effective_gradient = gradient;
                if (config_.decorrelation_lambda > 0.0 && batch_size > 1) {
                    const Eigen::MatrixXd decor_signal = compute_decorrelation_signal(batch_states[l + 1]);
                    effective_gradient.noalias() -=
                        config_.decorrelation_lambda *
                        ((layer_error_precisions_[l] * batch_deltas[l]) * decor_signal.transpose() * batch_scale);
                }
                gradient_norm = effective_gradient.norm();
                if (update_norms_override == nullptr) {
                    double layer_scale = 1.0;
                    if (config_.layer_adapt_beta > 0.0) {
                        const double next_second_moment = blend_second_moment(
                            layer_second_moments_[l],
                            config_.layer_adapt_beta,
                            matrix_mean_square(effective_gradient));
                        layer_scale =
                            1.0 / (std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon);
                    }
                    const Eigen::MatrixXd scaled_gradient = layer_scale * effective_gradient;
                    if (momentum_beta_ > 0.0) {
                        const Eigen::MatrixXd tentative_update =
                            momentum_beta_ * weight_velocities_[l] + weight_scale * scaled_gradient;
                        update_norm = tentative_update.norm();
                    } else {
                        update_norm = weight_scale * scaled_gradient.norm();
                    }
                }
            }
            diagnostics.weight_gradient_norms.push_back(gradient_norm);
            if (update_norms_override != nullptr) {
                update_norm = (*update_norms_override)[l];
            } else if (gradient_norms_override != nullptr) {
                update_norm = weight_scale * gradient_norm;
            }
            diagnostics.weight_update_norms.push_back(update_norm);
            diagnostics.weight_norms.push_back(weights_[l].norm());
        }

        diagnostics.mse = batch_errors.front().squaredNorm() /
            (static_cast<double>(config_.dims.front()) * static_cast<double>(batch_size));
        return diagnostics;
    };

    BatchTrainResult result;
    result.batch_size = batch_size;
    if (options.capture_final_gradients) {
        result.final_weight_gradients.resize(layer_count);
    }
    if (options.capture_final_states) {
        result.final_batch_states.resize(state_count);
    }

    compute_batch_errors_and_deltas();
    if (options.update_weights) {
        update_error_precisions_from_batch_errors();
    }

    auto maybe_log_step =
        [&](int relax_step,
            const std::vector<double>& state_update_norms,
            const std::vector<double>* gradient_norms_override,
            const std::vector<double>* update_norms_override) {
        if (std::binary_search(logged_steps.begin(), logged_steps.end(), relax_step)) {
            LoggedRelaxationStep entry;
            entry.relax_step = relax_step;
            entry.diagnostics =
                collect_batch_diagnostics(state_update_norms, gradient_norms_override, update_norms_override);
            result.logged_steps.push_back(std::move(entry));
        }
    };

    maybe_log_step(0, std::vector<double>(layer_count, 0.0), nullptr, nullptr);

    std::vector<double> last_state_update_norms(layer_count, 0.0);
    std::vector<double> last_gradient_norms(layer_count, 0.0);
    std::vector<double> last_update_norms(layer_count, 0.0);
    std::vector<Eigen::MatrixXd> last_gradients;
    if (options.capture_final_gradients) {
        last_gradients.resize(layer_count);
        for (std::size_t l = 0; l < layer_count; ++l) {
            last_gradients[l] =
                (layer_error_precisions_[l] * batch_deltas[l]) * batch_states[l + 1].transpose() * batch_scale;
        }
    }
    for (int relax_step = 1; relax_step <= steps; ++relax_step) {
        std::vector<double> state_update_norms(layer_count, 0.0);
        const double step_gate = config_.transient_gate_tau > 0.0
            ? (1.0 - std::exp(-static_cast<double>(relax_step) / config_.transient_gate_tau))
            : 1.0;
        const std::size_t top_index = state_count - 1;

        if (config_.sequential_inference) {
            {
                const Eigen::MatrixXd top_update =
                    state_scale *
                    (weights_[top_index - 1].transpose() *
                     (layer_error_precisions_[top_index - 1] * batch_deltas[top_index - 1]));
                state_update_norms[top_index - 1] = top_update.norm() * batch_norm_scale;
                batch_states[top_index] += top_update;
                compute_batch_layer_error_and_delta(top_index - 1);
            }

            for (std::size_t l = top_index - 1; l >= 1; --l) {
                const Eigen::MatrixXd back =
                    weights_[l - 1].transpose() * (layer_error_precisions_[l - 1] * batch_deltas[l - 1]);
                const Eigen::MatrixXd update =
                    state_scale * (back - (layer_error_precisions_[l] * batch_errors[l]));
                state_update_norms[l - 1] = update.norm() * batch_norm_scale;
                batch_states[l] += update;
                compute_batch_layer_error_and_delta(l - 1);
                if (l == 1) {
                    break;
                }
            }
        } else {
            {
                const Eigen::MatrixXd top_update =
                    state_scale *
                    (weights_[top_index - 1].transpose() *
                     (layer_error_precisions_[top_index - 1] * batch_deltas[top_index - 1]));
                state_update_norms[top_index - 1] = top_update.norm() * batch_norm_scale;
                batch_states[top_index] += top_update;
            }

            for (std::size_t l = top_index - 1; l >= 1; --l) {
                const Eigen::MatrixXd back =
                    weights_[l - 1].transpose() * (layer_error_precisions_[l - 1] * batch_deltas[l - 1]);
                const Eigen::MatrixXd update =
                    state_scale * (back - (layer_error_precisions_[l] * batch_errors[l]));
                state_update_norms[l - 1] = update.norm() * batch_norm_scale;
                batch_states[l] += update;
                if (l == 1) {
                    break;
                }
            }
        }

        compute_batch_errors_and_deltas();
        if (options.update_weights) {
            update_error_precisions_from_batch_errors();
        }
        std::vector<double> gradient_norms(layer_count, 0.0);
        std::vector<double> update_norms(layer_count, 0.0);
        for (std::size_t l = 0; l < layer_count; ++l) {
            const Eigen::MatrixXd gradient =
                (layer_error_precisions_[l] * batch_deltas[l]) * batch_states[l + 1].transpose() * batch_scale;
            Eigen::MatrixXd effective_gradient = gradient;
            if (config_.decorrelation_lambda > 0.0 && batch_size > 1) {
                const Eigen::MatrixXd decor_signal = compute_decorrelation_signal(batch_states[l + 1]);
                effective_gradient.noalias() -=
                    config_.decorrelation_lambda *
                    ((layer_error_precisions_[l] * batch_deltas[l]) * decor_signal.transpose() * batch_scale);
            }
            const Eigen::VectorXd bias_gradient =
                layer_error_precisions_[l] * batch_deltas[l].rowwise().mean();
            gradient_norms[l] = effective_gradient.norm();
            if (options.capture_final_gradients) {
                last_gradients[l] = effective_gradient;
            }
            double layer_scale = 1.0;
            if (config_.layer_adapt_beta > 0.0) {
                const double next_second_moment = blend_second_moment(
                    layer_second_moments_[l],
                    config_.layer_adapt_beta,
                    matrix_mean_square(effective_gradient));
                if (options.update_weights) {
                    layer_second_moments_[l] = next_second_moment;
                }
                layer_scale = 1.0 / (std::sqrt(std::max(0.0, next_second_moment)) + config_.layer_adapt_epsilon);
            }
            const Eigen::MatrixXd scaled_gradient = layer_scale * effective_gradient;
            const Eigen::VectorXd scaled_bias_gradient = layer_scale * bias_gradient;
            if (options.update_weights) {
                const double gated_weight_scale = weight_scale * step_gate;
                if (momentum_beta_ > 0.0) {
                    weight_velocities_[l] *= momentum_beta_;
                    weight_velocities_[l].noalias() += gated_weight_scale * scaled_gradient;
                    bias_velocities_[l] *= momentum_beta_;
                    bias_velocities_[l].noalias() += gated_weight_scale * scaled_bias_gradient;
                    update_norms[l] = weight_velocities_[l].norm();
                    weights_[l] += weight_velocities_[l];
                    if (config_.use_biases) {
                        biases_[l] += bias_velocities_[l];
                    }
                } else {
                    update_norms[l] = gated_weight_scale * scaled_gradient.norm();
                    weights_[l] += gated_weight_scale * scaled_gradient;
                    if (config_.use_biases) {
                        biases_[l].noalias() += gated_weight_scale * scaled_bias_gradient;
                    }
                }
            } else if (momentum_beta_ > 0.0) {
                const Eigen::MatrixXd tentative_update =
                    momentum_beta_ * weight_velocities_[l] + weight_scale * scaled_gradient;
                update_norms[l] = tentative_update.norm();
            } else {
                update_norms[l] = weight_scale * scaled_gradient.norm();
            }
        }
        compute_batch_errors_and_deltas();
        last_state_update_norms = state_update_norms;
        last_gradient_norms = gradient_norms;
        last_update_norms = update_norms;
        maybe_log_step(relax_step, state_update_norms, &gradient_norms, &update_norms);
    }

    result.final_diagnostics = collect_batch_diagnostics(
        last_state_update_norms,
        steps > 0 ? &last_gradient_norms : nullptr,
        steps > 0 ? &last_update_norms : nullptr);
    if (options.capture_final_gradients) {
        result.final_weight_gradients = std::move(last_gradients);
    }
    if (options.capture_final_states) {
        result.final_batch_states = batch_states;
    }
    result.average_mse = result.final_diagnostics.mse;
    return result;
}

RelaxationResult LTFN::relax(const Eigen::VectorXd& input, int steps, bool capture_trace) {
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
    result.mse = ltfn::compute_mse(input, result.reconstruction);
    result.final_error_norms = diagnostics.error_norms;
    result.final_weight_gradient_norms = diagnostics.weight_gradient_norms;
    return result;
}

RelaxationResult LTFN::reconstruct(const Eigen::VectorXd& input, int steps, bool capture_trace) {
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
            diagnostics = step_current(false);
            result.energy_trace.push_back(diagnostics.energy);
        }
    } else if (steps > 0) {
        for (int t = 1; t < steps; ++t) {
            advance_current(false);
        }
        diagnostics = step_current(false);
    }

    result.reconstruction = current_reconstruction();
    result.final_energy = diagnostics.energy;
    result.mse = ltfn::compute_mse(input, result.reconstruction);
    result.final_error_norms = diagnostics.error_norms;
    result.final_weight_gradient_norms = diagnostics.weight_gradient_norms;
    return result;
}

double LTFN::current_energy() const noexcept {
    ensure_predictions_current();
    double energy = 0.0;
    for (std::size_t l = 0; l < errors_.size(); ++l) {
        if (config_.visible_loss == VisibleLoss::Bce && l == 0) {
            energy += config_.visible_unit_precision
                ? weighted_binary_cross_entropy_energy(states_[0], predictions_[0], visible_error_precisions_)
                : layer_error_precisions_[l] * binary_cross_entropy_energy(states_[0], predictions_[0]);
        } else if (config_.visible_unit_precision && l == 0) {
            energy += 0.5 * (visible_error_precisions_.array() * errors_[0].array().square()).sum();
        } else {
            energy += layer_error_precisions_[l] * 0.5 * errors_[l].squaredNorm();
        }
    }
    return energy;
}

Eigen::VectorXd LTFN::current_reconstruction() const {
    ensure_predictions_current();
    return predictions_.front();
}

const std::vector<Eigen::VectorXd>& LTFN::states() const {
    return states_;
}

double LTFN::compute_mse(const Eigen::VectorXd& expected, const Eigen::VectorXd& actual) {
    return ltfn::compute_mse(expected, actual);
}

void LTFN::validate_dims() const {
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

void LTFN::initialize_latent_states() {
    if (config_.state_init == StateInit::Zero) {
        for (std::size_t l = 1; l < states_.size(); ++l) {
            states_[l].setZero();
        }
        return;
    }

    for (std::size_t l = 1; l < states_.size(); ++l) {
        states_[l] = sigmoid_vec(weights_[l - 1].transpose() * states_[l - 1]);
    }
}

void LTFN::compute_predictions_and_errors() const {
    for (std::size_t l = 0; l < weights_.size(); ++l) {
        compute_layer_prediction_error(l);
    }
    predictions_dirty_ = false;
}

void LTFN::compute_layer_prediction_error(std::size_t layer) const {
    pre_activations_[layer].noalias() = weights_[layer] * states_[layer + 1];
    if (config_.use_biases) {
        pre_activations_[layer] += biases_[layer];
    }
    predictions_[layer] = sigmoid_vec(pre_activations_[layer]);
    sigmoid_derivatives_[layer] = sigmoid_derivative_vec(pre_activations_[layer]);
    errors_[layer] = states_[layer] - predictions_[layer];
}

void LTFN::ensure_predictions_current() const {
    if (predictions_dirty_) {
        compute_predictions_and_errors();
    }
}

void LTFN::ensure_input_shape(const Eigen::VectorXd& input) const {
    if (input.size() != config_.dims.front()) {
        throw std::invalid_argument("Input dimension does not match the configured visible layer.");
    }
}

void LTFN::update_error_precisions_from_errors() {
    if (config_.error_precision_beta <= 0.0) {
        return;
    }

    std::size_t start_layer = 0;
    if (config_.visible_unit_precision) {
        const Eigen::ArrayXd mean_square = errors_[0].array().square();
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
        layer_error_precisions_[0] = 1.0;
        start_layer = 1;
    }

    for (std::size_t l = start_layer; l < errors_.size(); ++l) {
        const double mean_square = errors_[l].squaredNorm() / static_cast<double>(errors_[l].size());
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

}  // namespace ltfn
