cmake_minimum_required(VERSION 3.14)
project(TestInferenceEngineCPU)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Source files
set(SOURCES
    main.cpp
)

# Paths to the libraries
set(LIBRARY_PATH        "../build/lib")
set(LLAMA_LIBRARY_PATH  "../build/external/llama.cpp")

# Include directories for headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../            # For inference.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/llama.cpp/ggml/include
)

# Find the libraries
find_library(INFERENCE_ENGINE_LIB   InferenceEngineLib  PATHS ${LIBRARY_PATH}                   REQUIRED)
find_library(COMMON_LIB             common              PATHS ${LLAMA_LIBRARY_PATH}/common      REQUIRED)
find_library(GGML_LIB               ggml                PATHS ${LLAMA_LIBRARY_PATH}/ggml/src    REQUIRED)
find_library(LLAMA_LIB              llama               PATHS ${LLAMA_LIBRARY_PATH}/src         REQUIRED)

if(NOT INFERENCE_ENGINE_LIB)
    message(FATAL_ERROR "InferenceEngineLib not found in ${LIBRARY_PATH}. Please build the InferenceEngine library first.")
endif()
if(NOT COMMON_LIB)
    message(FATAL_ERROR "common library not found in ${LIBRARY_PATH}.")
endif()
if(NOT GGML_LIB)
    message(FATAL_ERROR "ggml library not found in ${LIBRARY_PATH}.")
endif()
if(NOT LLAMA_LIB)
    message(FATAL_ERROR "llama library not found in ${LIBRARY_PATH}.")
endif()

# Create the executable
add_executable(TestInferenceEngineCPU ${SOURCES})

# Link the executable against the libraries
target_link_libraries(TestInferenceEngineCPU PRIVATE
    ${INFERENCE_ENGINE_LIB}
    ${COMMON_LIB}
    ${LLAMA_LIB}
    ${GGML_LIB}
    pthread
)

# Set output directory
set_target_properties(TestInferenceEngineCPU PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Optionally, set the RPATH to find shared libraries at runtime
set_target_properties(TestInferenceEngineCPU PROPERTIES
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../build/lib"
)
