
add_library(charrnn "")

# the following block is essentially a copy/paste from the master CMakeLists.txt
set_property(TARGET charrnn PROPERTY CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/Users/tcassidy/gitlab/omg-pytorch-mac-arm64/torch/")
find_package(Torch REQUIRED)
target_link_libraries(charrnn "${TORCH_LIBRARIES}")


target_sources(charrnn
  PRIVATE
    RnnTrainer.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/RnnTrainer.h
  )

target_include_directories(charrnn
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )