cmake_minimum_required(VERSION 3.14)
project(InferenceEngine)

# Options
option(USE_CUDA   "Compile with CUDA support" OFF)
option(USE_VULKAN "Compile with VULKAN support" OFF)

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
set(GGML_BLAS             ON  CACHE BOOL "Enable GGML BLAS support"       FORCE)

# Build llama.cpp as a static library
set(BUILD_SHARED_LIBS	  OFF CACHE BOOL "Build llama.cpp as a static lib" FORCE)
set(GGML_STATIC_LINK      ON  CACHE BOOL "Static link ggml libraries"      FORCE)
set(GGML_STATIC           ON  CACHE BOOL "Static link ggml libraries"      FORCE)

# Enable GGML acceleration
if(USE_CUDA)
  set(GGML_CUDA     ON CACHE BOOL "Enable GGML CUDA support"       FORCE)
  message(STATUS "Using CUDA for GGML acceleration")
elseif(USE_VULKAN)
  set(GGML_VULKAN   ON CACHE BOOL "Enable GGML Vulkan support"     FORCE)
  message(STATUS "Using ROCm for GGML acceleration")
else()
  set(GGML_OPENBLAS ON CACHE BOOL "Enable GGML OpenBLAS support"   FORCE)
  message(STATUS "Using OpenBLAS for GGML acceleration")
endif()

# define USE_CUDA and USE_VULKAN for inference.cpp
target_compile_definitions(InferenceEngineLib PUBLIC
  $<$<BOOL:${USE_CUDA}>:USE_CUDA>
  $<$<BOOL:${USE_VULKAN}>:USE_VULKAN>
)

# Add llama.cpp subdirectory
add_subdirectory(${LLAMA_CPP_PATH})

# Link llama.cpp library
target_link_libraries(InferenceEngineLib PRIVATE llama)

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
