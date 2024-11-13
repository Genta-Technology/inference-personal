cmake_minimum_required(VERSION 3.14)
project(InferenceEngine)

# Options
option(USE_GPU "Compile with GPU support (CUDA and TensorRT)" OFF)
option(DEBUG "Debug the program using the -g flag on the compiler" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

if(DEBUG)
  message(STATUS "DEBUG MODE ON")
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

# Source files
set(SOURCES
  inference.cpp
)

# Handle USE_GPU option
if(USE_GPU)
  message(STATUS "Compiling with GPU support (USE_GPU is ON)")

  # Find CUDA
  find_package(CUDA REQUIRED)
  if (CUDA_FOUND)
    message(STATUS "CUDA library status:")
    message(STATUS "    version: ${CUDA_VERSION}")
    message(STATUS "    libraries: ${CUDA_LIBRARIES}")
    message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "CUDA not found. Please install CUDA or disable USE_GPU.")
  endif()

  # Find TensorRT
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/local/include /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
    DOC "TensorRT include directory"
  )
  find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
    DOC "TensorRT library"
  )

  if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    message(STATUS "TensorRT library status:")
    message(STATUS "    libraries: ${TENSORRT_LIBRARY}")
    message(STATUS "    include path: ${TENSORRT_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "TensorRT not found. Please install TensorRT or disable USE_GPU.")
  endif()

  # Add definitions and flags for GPU
  add_definitions(-DUSE_GPU)

  # Library dependencies for GPU
  set(LIBS
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${TENSORRT_LIBRARY}
    ${CUDA_cudart_LIBRARY}
  )
else()
  message(STATUS "Compiling without GPU support (USE_GPU is OFF)")
  set(LIBS "")
endif()

# Find llama.cpp library
set(LLAMA_CPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp")
if (NOT EXISTS "${LLAMA_CPP_PATH}/CMakeLists.txt")
  message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_PATH}. Please clone it or adjust LLAMA_CPP_PATH.")
endif()

# Set llama.cpp options before adding the subdirectory
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "Disable llama.cpp tests"        FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "Disable llama.cpp examples"     FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "Disable llama.cpp server"       FORCE)
set(LLAMA_ALL_WARNINGS    OFF CACHE BOOL "Disable warnings in llama.cpp"  FORCE)

# Enable GGML CUDA support if USE_GPU is ON
if(USE_GPU)
  set(GGML_CUDA ON CACHE BOOL "Enable GGML CUDA support" FORCE)
endif()

# Add llama.cpp subdirectory
add_subdirectory(${LLAMA_CPP_PATH})

# Create the library target
add_library(InferenceEngineLib STATIC ${SOURCES})

# Link libraries
target_link_libraries(InferenceEngineLib PUBLIC llama ${LIBS})

# Include directories for InferenceEngineLib
target_include_directories(InferenceEngineLib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLAMA_CPP_PATH}/include
  ${LLAMA_CPP_PATH}/common
  ${LLAMA_CPP_PATH}/ggml/include
)

# Include CUDA and TensorRT include directories if USE_GPU is ON
if(USE_GPU)
  target_include_directories(InferenceEngineLib PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
endif()

# Set the output directory
set_target_properties(InferenceEngineLib PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
