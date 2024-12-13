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
)

# Set the library paths
if(MSVC)
    # For Visual Studio, use generator expressions to get the correct paths
    set(CONFIGURATION $<CONFIG>)
    # set(LIBRARY_PATH        "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release")
    # set(LIBRARY_PATH_CUDA   "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release-cuda")
    set(LIBRARY_PATH_VULKAN "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release-vulkan")
else()
    # For other systems
    set(LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/lib")
endif()

# Find the libraries
# find_library(INFERENCE_ENGINE_LIB        NAMES InferenceEngineLib       PATHS ${LIBRARY_PATH}/lib        REQUIRED)
# find_library(INFERENCE_ENGINE_LIB_CUDA   NAMES InferenceEngineLibCuda   PATHS ${LIBRARY_PATH_CUDA}/lib   REQUIRED)
find_library(INFERENCE_ENGINE_LIB_VULKAN NAMES InferenceEngineLibVulkan PATHS ${LIBRARY_PATH_VULKAN}/lib REQUIRED)

#if(NOT INFERENCE_ENGINE_LIB OR NOT INFERENCE_ENGINE_LIB_CUDA OR NOT INFERENCE_ENGINE_LIB_VULKAN)
if(NOT INFERENCE_ENGINE_LIB_VULKAN)
    message(FATAL_ERROR "One or more inference libraries not found. Please build the InferenceEngine libraries first.")
endif()

message(STATUS "Found InferenceEngine libraries:")
# message(STATUS "  - OpenBLAS: ${INFERENCE_ENGINE_LIB}")
# message(STATUS "  - CUDA:     ${INFERENCE_ENGINE_LIB_CUDA}")
message(STATUS "  - Vulkan:   ${INFERENCE_ENGINE_LIB_VULKAN}")

# Link the executable against the libraries
#target_link_libraries(TestInferenceEngineCPU PRIVATE ${INFERENCE_ENGINE_LIB} ${INFERENCE_ENGINE_LIB_CUDA} ${INFERENCE_ENGINE_LIB_VULKAN})
target_link_libraries(TestInferenceEngineCPU PRIVATE ${INFERENCE_ENGINE_LIB_VULKAN})

# Include Threads
find_package(Threads REQUIRED)
target_link_libraries(TestInferenceEngineCPU PRIVATE Threads::Threads)

# Set output directory
set_target_properties(TestInferenceEngineCPU PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

# Copy the inference engine library .dll from the build directory /bin to the output directory
if(MSVC)
    add_custom_command(TARGET TestInferenceEngineCPU POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
#            "${LIBRARY_PATH}/bin/InferenceEngineLib.dll"
#            "${LIBRARY_PATH_CUDA}/bin/InferenceEngineLibCuda.dll"
            "${LIBRARY_PATH_VULKAN}/bin/InferenceEngineLibVulkan.dll"
            "$<TARGET_FILE_DIR:TestInferenceEngineCPU>"
    )
endif()