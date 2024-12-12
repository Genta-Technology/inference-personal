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

option(USE_CUDA "Use CUDA for inference" OFF)
option(USE_VULKAN "Use Vulkan for inference" OFF)

# Set the library paths
if(MSVC)
    # For Visual Studio, use generator expressions to get the correct paths
    set(CONFIGURATION $<CONFIG>)
    if(USE_CUDA)
        set(LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release-cuda/")
    elseif(USE_VULKAN)
        set(LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release-vulkan/")
    else()
        set(LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/x64-Release/")
    endif()
else()
    # For other systems
    set(LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../out/build/lib")
endif()

# Find the libraries
if(USE_CUDA)
    find_library(INFERENCE_ENGINE_LIB NAMES inference_cuda PATHS "${LIBRARY_PATH}" REQUIRED)
elseif(USE_VULKAN)
    find_library(INFERENCE_ENGINE_LIB NAMES inference_vulkan PATHS "${LIBRARY_PATH}" REQUIRED)
else()
    find_library(INFERENCE_ENGINE_LIB NAMES inference PATHS "${LIBRARY_PATH}" REQUIRED)
endif()

if(NOT INFERENCE_ENGINE_LIB)
    message(FATAL_ERROR "inference not found in ${LIBRARY_PATH}. Please build the InferenceEngine library first.")
endif()

message(STATUS "Found InferenceEngine library: ${INFERENCE_ENGINE_LIB}")

# Link the executable against the libraries
target_link_libraries(TestInferenceEngineCPU PRIVATE ${INFERENCE_ENGINE_LIB})

# Include Threads
find_package(Threads REQUIRED)
target_link_libraries(TestInferenceEngineCPU PRIVATE Threads::Threads)

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        target_include_directories(TestInferenceEngineCPU PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(TestInferenceEngineCPU PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})
    else()
        message(FATAL_ERROR "CUDA not found. Please install CUDA toolkit.")
    endif()
elseif(USE_VULKAN)
    find_package(Vulkan REQUIRED)
    if(Vulkan_FOUND)
        target_include_directories(TestInferenceEngineCPU PRIVATE ${Vulkan_INCLUDE_DIRS})
        target_link_libraries(TestInferenceEngineCPU PRIVATE ${Vulkan_LIBRARIES})
    else()
        message(FATAL_ERROR "Vulkan not found. Please install Vulkan SDK.")
    endif()
endif()

# Set output directory
set_target_properties(TestInferenceEngineCPU PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)