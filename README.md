# PyTorch Extension

Allow for the creation of library/includes such that additional functionality can be added to PyTorch in C++.


## register_module

The code for this was taken from here:
https://github.com/pytorch/examples/tree/main/cpp/dcgan

### Data

Not sure if this was documented and I missed it - but - the training data is read in C++ using this statement:

```
  auto dataset = torch::data::datasets::MNIST(kDataFolder)
                     .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                     .map(torch::data::transforms::Stack<>());
```

the name of the files (train-images-idx3-ubyte + train-labels-idx1-ubyte) are apparently hardcoded in the above call (from PyTorch) and can be grabbed from here: 

https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data


### Build Instructions

Run the following script: 

```
scripts/gen_makefile.sh
```

and then

```
scripts/build_executable.sh
```

After this - you should have an executable dcgan in the build directory.

### Intellisense

To get intellisense to work in VS Code - you'll need to use

"CMake: Configure"

"CMake: Build"

after that - intellisense (code completion) should work.

### How to Look at Progress/Data Refinement

use the python script display_samples.py like this:

```
python display_samples.py -i dcgan-sample-10.pt
```

which should output:

```
Saved out.png
```
