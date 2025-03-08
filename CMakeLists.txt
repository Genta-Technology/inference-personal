cmake_minimum_required(VERSION 3.14)
project(InferenceEngine)

include(cmake/ucm.cmake)

ucm_set_runtime(STATIC)

# Options
option(USE_CUDA   "Compile with CUDA support" OFF)
option(USE_VULKAN "Compile with VULKAN support" OFF)
option(DEBUG      "Compile with debugging information" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Detect if the architecture is ARM (AArch64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64|arm64|armv8|armv7)")
    message(STATUS "Building on ARM: Enabling -fPIC")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message(STATUS "Building on x86: No -fPIC required")
endif()

ucm_print_flags()

# Source files
set(SOURCES
  inference.cpp
)

# Header files (optional, but useful for IDEs)
set(HEADERS
  job.h
)

# Determine the target name based on options
if(USE_CUDA)
  set(TARGET_NAME "InferenceEngineLibCuda")
elseif(USE_VULKAN)
  set(TARGET_NAME "InferenceEngineLibVulkan")
else()
  set(TARGET_NAME "InferenceEngineLib")
endif()

# Create the library target
add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})

# Define INFERENCE_EXPORTS for the DLL
target_compile_definitions(${TARGET_NAME} PRIVATE INFERENCE_EXPORTS)

# Include directories for ${TARGET_NAME}
target_include_directories(${TARGET_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

install(TARGETS ${TARGET_NAME} RUNTIME)

# Find llama.cpp library
set(LLAMA_CPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp")
if(NOT EXISTS "${LLAMA_CPP_PATH}/CMakeLists.txt")
  message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_PATH}. Please clone it or adjust LLAMA_CPP_PATH.")
endif()

# Disable GGML_NATIVE for llama.cpp for cross-compilation
set(GGML_NATIVE           OFF CACHE BOOL "Disable LLAMA_NATIVE in llama.cpp" FORCE)
set(INS_ENB               ON  CACHE BOOL "Enable INS_ENB in llama.cpp"       FORCE)

# Set llama.cpp options before adding the subdirectory
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "Disable llama.cpp tests"        FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "Disable llama.cpp examples"     FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "Disable llama.cpp server"       FORCE)
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "Enable  llama.cpp common"       FORCE)
set(LLAMA_ALL_WARNINGS    OFF CACHE BOOL "Disable warnings in llama.cpp"  FORCE)

# Build llama.cpp as a static library
set(BUILD_SHARED_LIBS	  OFF CACHE BOOL "Build llama.cpp as a static lib" FORCE)
set(GGML_STATIC_LINK      ON  CACHE BOOL "Static link ggml libraries"      FORCE)
set(GGML_STATIC           ON  CACHE BOOL "Static link ggml libraries"      FORCE)

# Disable AVX512 for llama.cpp
set(LLAMA_AVX512          OFF CACHE BOOL "Disable AVX512 in llama.cpp"     FORCE)

# Enable GGML acceleration
if(USE_CUDA)
  set(GGML_CUDA ON CACHE BOOL "Enable GGML CUDA support" FORCE)
  message(STATUS "Using CUDA for GGML acceleration")

  find_package(CUDA REQUIRED)
  if(CUDA_FOUND)
    target_include_directories(${TARGET_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})
  else()
    message(FATAL_ERROR "CUDA not found. Please install CUDA toolkit.")
  endif()
elseif(USE_VULKAN)
  set(GGML_VULKAN ON CACHE BOOL "Enable GGML Vulkan support" FORCE)
  message(STATUS "Using vulkan for GGML acceleration")

  find_package(Vulkan REQUIRED)
  if(Vulkan_FOUND)
    target_include_directories(${TARGET_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${Vulkan_LIBRARIES})
  else()
    message(FATAL_ERROR "Vulkan not found. Please install Vulkan SDK.")
  endif()
else()
  message(STATUS "Using OpenBLAS for GGML acceleration")
endif()

# define USE_CUDA and USE_VULKAN and DEBUG for inference.cpp
target_compile_definitions(${TARGET_NAME} PUBLIC
  $<$<BOOL:${USE_CUDA}>:USE_CUDA>
  $<$<BOOL:${USE_VULKAN}>:USE_VULKAN>
  $<$<BOOL:${DEBUG}>:DEBUG>
)

# Add llama.cpp subdirectory
add_subdirectory(${LLAMA_CPP_PATH})

# Link llama.cpp library
target_link_libraries(${TARGET_NAME} PRIVATE llama common ggml)

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

# Include directories for ${TARGET_NAME}
target_include_directories(${TARGET_NAME} PUBLIC
  ${LLAMA_CPP_PATH}/include
  ${LLAMA_CPP_PATH}/common
  ${LLAMA_CPP_PATH}/ggml/include
)

# Include Threads
find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PUBLIC Threads::Threads)

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_11)

# Set the output directory
set_target_properties(${TARGET_NAME} PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Custom target to create a release archive (optional)
if(WIN32)
  # Use zip on Windows
  set(RELEASE_NAME "${TARGET_NAME}-${CMAKE_PROJECT_VERSION}")
  add_custom_target(release
    COMMAND ${CMAKE_COMMAND} -E tar "cfv" "${RELEASE_NAME}.zip" --format=zip
      "$<TARGET_FILE_DIR:${TARGET_NAME}>"
      "$<TARGET_FILE:${TARGET_NAME}>"
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Creating release archive ${RELEASE_NAME}.zip"
  )
else()
  # Use tar.gz on UNIX
  set(RELEASE_NAME "${TARGET_NAME}-${CMAKE_PROJECT_VERSION}")
  add_custom_target(release
    COMMAND ${CMAKE_COMMAND} -E tar "cfv" "${RELEASE_NAME}.tar.gz" --format=gnutar
      "$<TARGET_FILE_DIR:${TARGET_NAME}>"
      "$<TARGET_FILE:${TARGET_NAME}>"
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Creating release archive ${RELEASE_NAME}.tar.gz"
  )
endif()


# Add test
add_executable(test_inference test.cpp)
target_include_directories(test_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_inference PRIVATE ${TARGET_NAME})

# Put the test in the bin directory
set_target_properties(test_inference PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
