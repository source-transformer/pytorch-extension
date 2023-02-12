mkdir -p ../build
pushd ../build

#cmake -DCMAKE_PREFIX_PATH=/Users/tcassidy/gitlab/omg-pytorch-mac-arm64/torch/ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cmake ..

popd
