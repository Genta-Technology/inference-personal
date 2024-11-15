cmake_minimum_required(VERSION 3.14)
project(TestInferenceEngineCPU)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    main.cpp
)

# Create the executable
add_executable(TestInferenceEngineCPU ${SOURCES})

# Include directories for headers
target_include_directories(TestInferenceEngineCPU PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../            # For inference.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/llama.cpp/ggml/include
)

# Set the library paths
if(MSVC)
    # For Visual Studio, use generator expressions to get the correct paths
    set(CONFIGURATION $<CONFIG>)
    set(LIBRARY_PATH        "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release/lib")
    set(DLL_PATH            "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release/bin")
    set(LLAMA_LIBRARY_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release/external/llama.cpp")
else()
    # For other systems
    set(LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/lib")
    set(LLAMA_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/external/llama.cpp")
endif()

# Find the libraries
find_library(INFERENCE_ENGINE_LIB NAMES InferenceEngineLib PATHS "${LIBRARY_PATH}" REQUIRED)
find_library(COMMON_LIB NAMES common PATHS "${LLAMA_LIBRARY_PATH}/common" REQUIRED)
find_library(GGML_LIB NAMES ggml PATHS "${LLAMA_LIBRARY_PATH}/ggml/src" REQUIRED)
find_library(LLAMA_LIB NAMES llama PATHS "${LLAMA_LIBRARY_PATH}/src" REQUIRED)

if(NOT INFERENCE_ENGINE_LIB)
    message(FATAL_ERROR "InferenceEngineLib not found in ${LIBRARY_PATH}. Please build the InferenceEngine library first.")
endif()
if(NOT COMMON_LIB)
    message(FATAL_ERROR "common library not found in ${LLAMA_LIBRARY_PATH}/common.")
endif()
if(NOT GGML_LIB)
    message(FATAL_ERROR "ggml library not found in ${LLAMA_LIBRARY_PATH}/ggml/src.")
endif()
if(NOT LLAMA_LIB)
    message(FATAL_ERROR "llama library not found in ${LLAMA_LIBRARY_PATH}/src.")
endif()

# Link the executable against the libraries
target_link_libraries(TestInferenceEngineCPU PRIVATE
    ${INFERENCE_ENGINE_LIB}
    ${COMMON_LIB}
    ${LLAMA_LIB}
    ${GGML_LIB}
)

# Include Threads
find_package(Threads REQUIRED)
target_link_libraries(TestInferenceEngineCPU PRIVATE Threads::Threads)

# Set output directory
set_target_properties(TestInferenceEngineCPU PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

# Optionally, set the RPATH to find shared libraries at runtime (for Unix systems)
if(UNIX)
    set_target_properties(TestInferenceEngineCPU PROPERTIES
        INSTALL_RPATH "${LIBRARY_PATH}"
    )
endif()

if(WIN32)
    # Copy required DLLs after build
    add_custom_command(TARGET TestInferenceEngineCPU POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL_PATH}/ggml.dll"
            "${DLL_PATH}/llama.dll"
            $<TARGET_FILE_DIR:TestInferenceEngineCPU>
    )
endif()