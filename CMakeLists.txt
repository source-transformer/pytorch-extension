#cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_minimum_required(VERSION 3.12)

project(rnn_text)

set(CMAKE_PREFIX_PATH "/Users/tcassidy/gitlab/omg-pytorch-mac-arm64/torch/")

find_package(Torch REQUIRED)


# Attempt to include char-rnn lib
# Apparently this way of wildcard including files is bleeding edge and only barely supported
#file(GLOB charrnn_SRC
#     "char-rnn/*.h"
#     "char-rnn/*.cpp"
#)
#file(GLOB charrnn_SRC CONFIGURE_DEPENDS "*.h" "*.cpp")
#add_executable(helloworld ${helloworld_SRC})

# we could just manually append .cpp files into the add_executable() call - but that seems painful - exp:
# add_executable(my_project_executable main.cpp include/helper.cpp ...) 
#add_executable(rnn-text main.cpp ${charrnn_SRC})

add_subdirectory(char-rnn/char-rnn)

add_executable(rnn-text main.cpp)
#target_link_libraries(rnn-text "${TORCH_LIBRARIES}")
target_link_libraries(rnn-text "${TORCH_LIBRARIES}"
    charrnn
)
set_property(TARGET rnn-text PROPERTY CXX_STANDARD 14)

# the following doesn't appear to do anything
#set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
#set_property( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# the following were misguided attempts to get intellisense working with PyTorch

#include_directories("/Users/tcassidy/gitlab/omg-pytorch-mac-arm64/torch/include") 
#target_include_directories(Torch, "${CMAKE_CURRENT_SOURCE_DIR}/include")
#target_include_directories(rnn-text, "${TORCH_INCLUDES}")
# this one at least didn't break the build
#target_include_directories(rnn-text PRIVATE "${TORCH_INCLUDE_DIRS}")

# to get this included file:
# torch/include/torch/csrc/api/include/torch/torch.h
# this one at least didn't break the build
#target_include_directories(rnn-text PRIVATE "${TORCH_INCLUDE_DIRS}/torch/csrc/api/include/")


