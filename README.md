# PyTorch Extension

Allow for the creation of library/includes such that additional functionality can be added to PyTorch in C++.


## char-rnn

### Build Instructions

Run the following script: 

```
scripts/gen_makefile.sh
```

and then

```
scripts/build_executable.sh
```

After this - you should have an executable rnn-text in the build directory.

### CMake Integration into VS Code + Intellisense

#### CMake in VS Code

This project using cmake.  So you might need to tell vs code where cmake is located.  So if you get the following error:

```
Bad CMake executable: "". Check to make sure it is installed or the value of the "cmake.cmakePath" setting contains the correct path
```

just run the command "where cmake" and then update your vscode settings like this:

```
"cmake.cmakePath": "<full file path>"
```

#### Intellisense

To get intellisense to work in VS Code - you'll need to use vs code's "Command Palette" (on Mac cmd-shift-p) and then run these two:

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

## CMake Resources

- Create .so (shared object) instead of .a from lib
https://stackoverflow.com/questions/11293572/cmake-create-a-shared-object

- Add directory of files
    - Apparently this approach (glob) isn't endorsed by owners of CMake
    https://stackoverflow.com/questions/3201154/automatically-add-all-files-in-a-folder-to-a-target-using-cmake
    - ended up using the lib/subdirectory approach recommended here:
    https://stackoverflow.com/a/68360488/5692144

## VS Code CMake Integration

Created the following issue - which I figured out on my own:
https://github.com/microsoft/vscode-cmake-tools/issues/3018




