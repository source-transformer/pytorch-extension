cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_pytorch_use)

find_package(Torch REQUIRED)

add_executable(test_pytorch ref_pytorch.cpp)
target_link_libraries(test_pytorch "${TORCH_LIBRARIES}")
set_property(TARGET test_pytorch PROPERTY CXX_STANDARD 14)
