cmake_minimum_required(VERSION 3.18)
if(DEFINED ENV{CUDA_HOME})
  set(CUDA_HOME $ENV{CUDA_HOME})
else()
  set(CUDA_HOME "/usr/local/cuda-12.1")
endif()

set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc")
set(CMAKE_LANGUAGE_COMPILER "${CUDA_HOME}/bin/nvcc")
set(LD_LIBRARY_PATH "${CUDA_HOME}/lib64")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project("DualUniQuant4ERQ" LANGUAGES CXX CUDA)

find_package(Git REQUIRED)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    message(STATUS "Populating Git submodule.")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR 
                "git submodule updata --init --recursive failed with ${GIT_SUBMOD_RESULT}.")
    endif()
endif()

##* Make Process Settings
include(ProcessorCount)
ProcessorCount(NUM_CORES)
# set to 90% of the available cores
math(EXPR NUM_CORES_90PERCENT "${NUM_CORES} * 90 / 100")
if(NUM_CORES_90PERCENT GREATER 0)
  set(CTEST_BUILD_FLAGS -j${NUM_CORES_90PERCENT})
  set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${NUM_CORES_90PERCENT})
else()
  set(CTEST_BUILD_FLAGS -j1)
  set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL 1)
endif()

##* Essential Library
# Python Requirements
find_package(PythonInterp REQUIRED)
# CUDA Platform Requirements
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_KNOWN_FEATURES "80;86")   # For NVIDIA Ampere Architecture
set(CMAKE_CUDA_ARCHITECTURES "80;86")    # For NVIDIA Ampere Architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic -Xcompiler -Wall")
# cuda and cutlass include dependencies
if(DEFINED ENV{TRITON_HOME})
    message(STATUS "TritonInclude: $ENV{TRITON_HOME}/backends/nvidia/include/")
    include_directories($ENV{TRITON_HOME}/backends/nvidia/include/)
    include_directories(${CUDA_HOME}/include)
else()
    include_directories(${CUDA_HOME}/include)
endif()

include_directories(lib/cutlass/include)
include_directories(lib/cutlass/tools/util/include)

# LibTorch Requirements
option(require_libtorch "Download and configure libtorch" ON)
if(require_libtorch)
    message(STATUS "[Libtorch required], configuring...")
    set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch")
    if(NOT EXISTS "${LIBTORCH_DIR}/build-version")
        message(STATUS "libtorch not found, will download...")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.5.1%2Bcu121.zip")
        set(LIBTORCH_ZIP "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch.zip")
        set(LIBTORCH_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib/libtorch")
        file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ZIP}
            SHOW_PROGRESS
            TIMEOUT 12000)
        message(STATUS "Extracting libtorch...")
        execute_process(COMMAND unzip ${LIBTORCH_ZIP} -d ${LIBTORCH_DIR})
        execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${LIBTORCH_ZIP})
    else()
        message(STATUS "libtorch found, configuring...")
    endif()
    # set(Torch_DIR ${LIBTORCH_DIR})
    set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch" CACHE PATH "Torch dir")
    file(READ "${LIBTORCH_DIR}/build-version" BUILD_VERSION_CONTENT)
    string(STRIP "${BUILD_VERSION_CONTENT}" BUILD_VERSION_CONTENT)
    find_package(Torch REQUIRED)
    message(STATUS "LibTorch status:")
    message(STATUS "  - version: ${BUILD_VERSION_CONTENT}")
    message(STATUS "  - libraries: ${TORCH_LIBRARIES}")
    include_directories(${Torch_DIR}/include)
else()
    message(STATUS "[Libtorch NOT required], skipping configuration.")
endif()

##* Third-party Library
# spdlog Requirements
add_subdirectory(lib/spdlog)
if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

# tabulate Requirements
include_directories(lib/tabulate/include)

# indicators Requirements
include_directories(lib/indicators/include)

## Project include header files
include_directories(include)

## Project sources
aux_source_directory(src/kernels SRC_LIST)
aux_source_directory(src/module MOD_LIST)
aux_source_directory(src/tools TOOL_LIST)

add_executable(EfficientLinear4bitTest EXCLUDE_FROM_ALL ${SRC_LIST} ${MOD_LIST} ${TOOL_LIST} test/test.cu)
set_property(TARGET EfficientLinear4bitTest PROPERTY CXX_STANDARD 17)
target_link_libraries(EfficientLinear4bitTest PRIVATE spdlog::spdlog_header_only)
target_link_libraries(EfficientLinear4bitTest PUBLIC -lcurand ${TORCH_LIBRARIES})

add_executable(pipeline EXCLUDE_FROM_ALL ${SRC_LIST} ${MOD_LIST} ${TOOL_LIST} test/pipeline.cu)
set_property(TARGET pipeline PROPERTY CXX_STANDARD 17)
target_link_libraries(pipeline PRIVATE spdlog::spdlog_header_only)
target_link_libraries(pipeline PUBLIC -lcurand ${TORCH_LIBRARIES})

add_custom_target(PYSETUP ALL
    COMMAND ${CMAKE_COMMAND} -E echo "This target is used for python setup build preparation."
)

