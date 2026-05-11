#include "ltfn.h"

#if defined(LTFN_USE_CUDA_BACKEND)
#include "ltfn_cuda.h"
#endif

#include <stdexcept>
#include <utility>

namespace ltfn {

std::string visible_loss_to_string(VisibleLoss visible_loss) {
    switch (visible_loss) {
        case VisibleLoss::Mse:
            return "mse";
        case VisibleLoss::Bce:
            return "bce";
    }
    return "unknown";
}

std::string state_init_to_string(StateInit state_init) {
    switch (state_init) {
        case StateInit::Zero:
            return "zero";
        case StateInit::Tied:
            return "tied";
    }
    return "unknown";
}

std::string backend_to_string(ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::Cpu:
            return "cpu";
        case ComputeBackend::Cuda:
            return "cuda";
    }
    return "unknown";
}

bool is_cuda_backend_compiled() noexcept {
#if defined(LTFN_USE_CUDA_BACKEND)
    return true;
#else
    return false;
#endif
}

std::unique_ptr<ILTFNModel> create_model(const LTFNConfig& config, std::uint32_t seed, ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::Cpu:
            return std::make_unique<LTFN>(config, seed);
        case ComputeBackend::Cuda:
#if defined(LTFN_USE_CUDA_BACKEND)
            return std::make_unique<LTFNCuda>(config, seed);
#else
            throw std::runtime_error("CUDA backend requested but this build does not include CUDA support.");
#endif
    }

    throw std::runtime_error("Unknown compute backend requested.");
}

}  // namespace ltfn
