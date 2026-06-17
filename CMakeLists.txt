cmake_minimum_required(VERSION 3.20)

project(ltfn
    VERSION 1.0.0
    DESCRIPTION "Layered Transient Free Energy Network experiment system"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LTFN_BUILD_TESTING "Enable smoke-test registration with CTest" ON)
option(LTFN_FETCH_EIGEN "Fetch Eigen automatically when it is not installed" ON)
option(LTFN_ENABLE_CUDA "Build the optional CUDA backend when a CUDA toolchain is available" ON)

if(LTFN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
        endif()
        find_package(CUDAToolkit REQUIRED)
        set(LTFN_HAVE_CUDA_BACKEND ON)
        message(STATUS "LTFN CUDA backend enabled with compiler: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "LTFN_ENABLE_CUDA is ON but no CUDA compiler was detected. Building CPU-only.")
    endif()
endif()

include(GNUInstallDirs)

find_package(Eigen3 3.4 QUIET CONFIG)

if(NOT Eigen3_FOUND AND LTFN_FETCH_EIGEN)
    include(FetchContent)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()

    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
endif()

if(NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "Eigen3 was not found. Install Eigen3 or enable LTFN_FETCH_EIGEN.")
endif()

add_library(ltfn_core
    src/logger.cpp
    src/ltfn.cpp
    src/ltfn_factory.cpp
    src/utils.cpp
)

if(LTFN_HAVE_CUDA_BACKEND)
    target_sources(ltfn_core PRIVATE src/ltfn_cuda.cu)
endif()

target_include_directories(ltfn_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(ltfn_core
    PUBLIC
        Eigen3::Eigen
)

if(LTFN_HAVE_CUDA_BACKEND)
    target_compile_definitions(ltfn_core PUBLIC LTFN_USE_CUDA_BACKEND=1)
    target_link_libraries(ltfn_core PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_options(ltfn_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(ltfn_core PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(MSVC)
    target_compile_options(ltfn_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(ltfn_core
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-pedantic>
    )
endif()

add_executable(ltfn_train
    src/main.cpp
)

target_link_libraries(ltfn_train PRIVATE ltfn_core)

add_executable(ltfn_h1_diag
    src/h1_gradient_bias.cpp
)

target_link_libraries(ltfn_h1_diag PRIVATE ltfn_core)

add_executable(ltfn_h3_dump
    src/h3_repr_dump.cpp
)

target_link_libraries(ltfn_h3_dump PRIVATE ltfn_core)

add_executable(ltfn_h4_sequence
    src/h4_sequence_diag.cpp
)

target_link_libraries(ltfn_h4_sequence PRIVATE ltfn_core)

add_executable(ltfn_tinylm
    src/tinylm_main.cpp
    src/tinylm.cpp
)

target_include_directories(ltfn_tinylm
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(ltfn_tinylm
    PRIVATE
        Eigen3::Eigen
)

if(MSVC)
    target_compile_options(ltfn_tinylm PRIVATE /W4 /permissive-)
else()
    target_compile_options(ltfn_tinylm PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS ltfn_train
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(LTFN_BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_test(
        NAME ltfn_smoke
        COMMAND ltfn_train
            --mode smoke-test
            --output-dir ${CMAKE_BINARY_DIR}/smoke_test_output
            --steps 20
            --max-train-samples 32
            --eval-samples 8
    )
endif()
