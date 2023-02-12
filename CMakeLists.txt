cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rnn_text)

set(CMAKE_PREFIX_PATH "/Users/tcassidy/gitlab/omg-pytorch-mac-arm64/torch/")

find_package(Torch REQUIRED)

add_executable(rnn-text main.cpp)
target_link_libraries(rnn-text "${TORCH_LIBRARIES}")
set_property(TARGET rnn-text PROPERTY CXX_STANDARD 14)

# the following doesn't appear to do anything
#set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
#set_property( CMAKE_EXPORT_COMPILE_COMMANDS ON )

#include_directories("/Users/tcassidy/gitlab/omg-pytorch-mac-arm64/torch/include") 
#target_include_directories(Torch, "${CMAKE_CURRENT_SOURCE_DIR}/include")
#target_include_directories(rnn-text, "${TORCH_INCLUDES}")
target_include_directories(rnn-text PRIVATE "${TORCH_INCLUDE_DIRS}")

# to get this included file:
# torch/include/torch/csrc/api/include/torch/torch.h
target_include_directories(rnn-text PRIVATE "${TORCH_INCLUDE_DIRS}/torch/csrc/api/include/")


