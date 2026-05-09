#pragma once

#include "utils.h"

#include <array>
#include <fstream>
#include <string>

namespace ltfn {

enum class LogTag : std::size_t {
    Energy = 0,
    LayerErrors,
    Gradients,
    Eval,
    Latents,
    Reconstructions,
    Events,
    Count
};

class Logger {
public:
    Logger(const ExperimentConfig& config, int argc, char** argv);

    bool is_enabled(LogTag tag) const noexcept;
    bool should_log_step(int relax_step) const noexcept;

    const fs::path& root_dir() const noexcept;
    const fs::path& checkpoints_dir() const noexcept;
    const fs::path& reconstructions_dir() const noexcept;

    void log_event(const std::string& type, const std::string& payload_json);
    void log_eval_summary(const MetricRow& row);
    void log_relaxation_step(
        const std::string& phase,
        std::size_t epoch,
        std::size_t samples_seen,
        std::size_t sample_index,
        int relax_step,
        const StepDiagnostics& diagnostics);
    void save_eval_reconstructions(
        const Dataset& dataset,
        const EvaluationResult& evaluation,
        std::size_t samples_seen) const;
    void log_eval_latents(
        const EvaluationResult& evaluation,
        std::size_t epoch,
        std::size_t samples_seen);
    void log_eval_effective_dimensions(
        const EvaluationResult& evaluation,
        std::size_t epoch,
        std::size_t samples_seen);
    void log_eval_error_variances(
        const EvaluationResult& evaluation,
        std::size_t epoch,
        std::size_t samples_seen);

private:
    void open_streams();
    void write_config_snapshot(int argc, char** argv);
    void write_headers();

    ExperimentConfig config_;
    fs::path root_dir_;
    fs::path reconstructions_dir_;
    fs::path checkpoints_dir_;
    std::size_t step_interval_{10};
    std::array<bool, static_cast<std::size_t>(LogTag::Count)> enabled_{};
    std::ofstream metrics_csv_;
    std::ofstream events_jsonl_;
    std::ofstream energy_csv_;
    std::ofstream layer_errors_csv_;
    std::ofstream grad_norms_csv_;
    std::ofstream latent_top_csv_;
    std::ofstream effective_dims_csv_;
    std::ofstream error_variances_csv_;
};

}  // namespace ltfn
