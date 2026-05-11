#pragma once

#include "ltfn.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ltfn {

namespace fs = std::filesystem;

enum class RunMode {
    Train,
    Eval,
    SmokeTest
};

struct ExperimentConfig {
    RunMode mode{RunMode::Train};
    ComputeBackend backend{ComputeBackend::Cpu};
    fs::path data_dir;
    fs::path output_dir{"runs/latest"};
    fs::path checkpoint_path;
    fs::path resume_path;
    fs::path save_initial_checkpoint_path;
    std::vector<int> dims{784, 256, 64, 32};
    double tau_r{0.1};
    double lr_w{1e-5};
    double lr_w_final{1e-6};
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
    double momentum_beta{0.0};
    double layer_adapt_beta{0.0};
    double layer_adapt_epsilon{1e-8};
    double decorrelation_lambda{0.0};
    int steps{200};
    std::size_t batch_size{32};
    std::size_t max_epochs{1};
    std::size_t max_train_samples{0};
    std::size_t eval_samples{1000};
    std::size_t eval_interval{500};
    std::size_t checkpoint_interval{5000};
    std::size_t probe_index{0};
    std::size_t recon_samples_to_save{4};
    std::size_t logger_step_interval{10};
    bool cosine_lr_schedule{false};
    bool shuffle{true};
    std::uint32_t seed{42U};
    std::string logger_tags{"all"};
};

struct Dataset {
    std::vector<Eigen::VectorXd> images;
    std::size_t rows{28};
    std::size_t cols{28};
};

struct CheckpointState {
    std::size_t samples_seen{0};
    std::size_t epochs_completed{0};
    std::uint32_t seed{42U};
};

struct MetricRow {
    std::size_t epoch{0};
    std::size_t samples_seen{0};
    double train_window_mse{0.0};
    double eval_mse{0.0};
    double eval_energy{0.0};
    double probe_mse{0.0};
    double probe_energy{0.0};
    double images_per_sec{0.0};
    double elapsed_sec{0.0};
    double avg_weight_norm{0.0};
    double avg_gradient_norm{0.0};
};

struct EffectiveDimensionRow {
    std::size_t layer{0};
    int width{0};
    std::size_t samples{0};
    int effective_dim_90{0};
    double effective_dim_ratio{0.0};
    double largest_singular{0.0};
    double smallest_singular{0.0};
    double spectrum_ratio{0.0};
};

struct ErrorVarianceRow {
    std::size_t layer{0};
    int width{0};
    std::size_t samples{0};
    double error_mean{0.0};
    double error_variance{0.0};
    double error_mean_square{0.0};
};

struct EvaluationResult {
    double average_mse{0.0};
    double average_energy{0.0};
    std::vector<Eigen::VectorXd> reconstructions;
    std::vector<Eigen::VectorXd> top_representations;
    std::vector<EffectiveDimensionRow> effective_dimensions;
    std::vector<ErrorVarianceRow> error_variances;
};

struct MonitoringContext {
    fs::path root_dir;
    fs::path reconstructions_dir;
    fs::path checkpoints_dir;
    std::ofstream metrics_csv;
    std::ofstream events_jsonl;
};

bool parse_args(int argc, char** argv, ExperimentConfig& config, std::string& error_message);
std::string usage_text(const char* program_name);
std::string mode_to_string(RunMode mode);
std::string backend_to_string(ComputeBackend backend);
std::string join_command_line(int argc, char** argv);

Dataset load_mnist_images(const fs::path& image_file, std::size_t limit = 0);
Dataset generate_smoke_dataset(std::size_t count, std::uint32_t seed);

void ensure_directory(const fs::path& path);
MonitoringContext create_monitoring_context(const ExperimentConfig& config, int argc, char** argv);
void write_config_snapshot(const MonitoringContext& monitoring, const ExperimentConfig& config, int argc, char** argv);
void append_metric(MonitoringContext& monitoring, const MetricRow& row);
void append_event(MonitoringContext& monitoring, const std::string& type, const std::string& payload_json);

void write_pgm(const fs::path& file_path, const Eigen::VectorXd& image, std::size_t rows, std::size_t cols);
void write_side_by_side_pgm(
    const fs::path& file_path,
    const Eigen::VectorXd& left,
    const Eigen::VectorXd& right,
    std::size_t rows,
    std::size_t cols);

double average_weight_norm(const std::vector<Eigen::MatrixXd>& weights);
double average_norm(const std::vector<double>& values);

bool save_checkpoint(
    const fs::path& file_path,
    const ILTFNModel& model,
    const CheckpointState& state,
    std::string& error_message);

bool load_checkpoint(
    const fs::path& file_path,
    ILTFNModel& model,
    CheckpointState& state,
    std::string& error_message);

EvaluationResult evaluate_model(
    ILTFNModel& model,
    const Dataset& dataset,
    int steps,
    std::size_t limit,
    std::size_t reconstructions_to_keep);

}  // namespace ltfn
