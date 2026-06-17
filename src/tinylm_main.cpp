#include "tinylm.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        ltfn::tinylm::TinyLMConfig config;
        bool show_help = false;
        std::string error;
        if (!ltfn::tinylm::parse_tinylm_args(argc, argv, config, show_help, error)) {
            std::cerr << "Argument error: " << error << "\n\n";
            std::cout << ltfn::tinylm::tinylm_usage(argv[0]);
            return 1;
        }
        if (show_help) {
            std::cout << ltfn::tinylm::tinylm_usage(argv[0]);
            return 0;
        }
        ltfn::tinylm::run_tinylm(config, argc, argv);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << '\n';
        return 1;
    }
}
