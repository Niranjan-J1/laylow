cmake_minimum_required(VERSION 3.16)
project(laylow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect AVX2 support and enable SIMD optimizations
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 supported - enabling SIMD optimizations")
    add_compile_options(-mavx2 -mfma)
else()
    message(WARNING "AVX2 not supported - falling back to scalar math")
endif()

# Release build optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()

# Engine library
add_library(laylow_engine STATIC
    engine/src/gguf.cpp
    engine/src/tensor.cpp
    engine/src/matmul.cpp
    engine/src/transformer.cpp
    engine/src/tokenizer.cpp
)

target_include_directories(laylow_engine PUBLIC
    engine/include
)

# Main executable
add_executable(laylow
    engine/src/main.cpp
)

target_link_libraries(laylow PRIVATE laylow_engine)

# Print build summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")