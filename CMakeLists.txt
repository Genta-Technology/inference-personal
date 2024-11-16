cmake_minimum_required(VERSION 3.14)
project(InferenceEngine)

# Options
option(USE_GPU "Compile with GPU support (CUDA and TensorRT)" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
  inference.cpp
)

# Header files (optional, but useful for IDEs)
set(HEADERS
  job.h
)

# Create the library target
add_library(InferenceEngineLib STATIC ${SOURCES} ${HEADERS})

# Include directories for InferenceEngineLib
target_include_directories(InferenceEngineLib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler flags and definitions
if(MSVC)
  # MSVC-specific compiler flags
  target_compile_options(InferenceEngineLib PRIVATE
    $<$<CONFIG:Debug>:/W3 /MDd /Zi /Od /RTC1>
    $<$<CONFIG:Release>:/W3 /MD /O2>
  )
  target_compile_definitions(InferenceEngineLib PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )
else()
  # GCC/Clang-specific compiler flags
  target_compile_options(InferenceEngineLib PRIVATE
    $<$<CONFIG:Debug>:-Wall -g>
    $<$<CONFIG:Release>:-Wall -O3 -DNDEBUG>
  )
  target_compile_definitions(InferenceEngineLib PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )
endif()

# Handle USE_GPU option
if(USE_GPU)
  message(STATUS "Compiling with GPU support (USE_GPU is ON)")
  target_compile_definitions(InferenceEngineLib PUBLIC USE_GPU)

  # Find CUDA
  find_package(CUDA REQUIRED)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install CUDA or disable USE_GPU.")
  else()
    message(STATUS "CUDA found: version ${CUDA_VERSION}")
  endif()

  # Find TensorRT
  if(WIN32)
    # Windows-specific TensorRT paths
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      HINTS "$ENV{TRT_DIR}/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include"
            "C:/Program Files/NVIDIA Corporation/TensorRT/include"
      DOC "TensorRT include directory"
    )
    find_library(TENSORRT_LIBRARY nvinfer
      HINTS "$ENV{TRT_DIR}/lib"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64"
            "C:/Program Files/NVIDIA Corporation/TensorRT/lib"
      DOC "TensorRT library"
    )
  else()
    # UNIX-specific TensorRT paths
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      HINTS /usr/include /usr/local/include
      DOC "TensorRT include directory"
    )
    find_library(TENSORRT_LIBRARY nvinfer
      HINTS /usr/lib /usr/local/lib
      DOC "TensorRT library"
    )
  endif()

  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please install TensorRT or disable USE_GPU.")
  else()
    message(STATUS "TensorRT found")
  endif()

  # Include directories and libraries
  target_include_directories(InferenceEngineLib PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(InferenceEngineLib PUBLIC ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY})
else()
  message(STATUS "Compiling without GPU support (USE_GPU is OFF)")
endif()

# Find llama.cpp library
set(LLAMA_CPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp")
if(NOT EXISTS "${LLAMA_CPP_PATH}/CMakeLists.txt")
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

# Link llama.cpp library
target_link_libraries(InferenceEngineLib PUBLIC llama)

# Include directories for InferenceEngineLib
target_include_directories(InferenceEngineLib PUBLIC
  ${LLAMA_CPP_PATH}/include
  ${LLAMA_CPP_PATH}/common
  ${LLAMA_CPP_PATH}/ggml/include
)

# Include Threads
find_package(Threads REQUIRED)
target_link_libraries(InferenceEngineLib PUBLIC Threads::Threads)

# Set the output directory
set_target_properties(InferenceEngineLib PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Custom target to create a release archive (optional)
if(WIN32)
  # Use zip on Windows
  set(RELEASE_NAME "InferenceEngineLib-${CMAKE_PROJECT_VERSION}")
  add_custom_target(release
    COMMAND ${CMAKE_COMMAND} -E tar "cfv" "${RELEASE_NAME}.zip" --format=zip
      "$<TARGET_FILE_DIR:InferenceEngineLib>"
      "$<TARGET_FILE:InferenceEngineLib>"
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Creating release archive ${RELEASE_NAME}.zip"
  )
else()
  # Use tar.gz on UNIX
  set(RELEASE_NAME "InferenceEngineLib-${CMAKE_PROJECT_VERSION}")
  add_custom_target(release
    COMMAND ${CMAKE_COMMAND} -E tar "cfv" "${RELEASE_NAME}.tar.gz" --format=gnutar
      "$<TARGET_FILE_DIR:InferenceEngineLib>"
      "$<TARGET_FILE:InferenceEngineLib>"
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Creating release archive ${RELEASE_NAME}.tar.gz"
  )
endif()
