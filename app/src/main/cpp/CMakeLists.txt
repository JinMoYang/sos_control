cmake_minimum_required(VERSION 3.22.1)
project(sos_tensor LANGUAGES CXX)

add_library(sos_tensor SHARED native_tensor_preprocessor.cpp)
target_compile_features(sos_tensor PRIVATE cxx_std_17)
target_compile_options(sos_tensor PRIVATE
    -O3 -ffast-math -fno-exceptions -fno-rtti -march=armv8-a+simd)
