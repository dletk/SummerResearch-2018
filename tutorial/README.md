# Tutorial on using openCV with CUDA.

## Make openCV project with cmake

The current version of openCV in this project is 4.0.0. This version requires the have c++11 as the compiled version. Therefore, besides the provided content of the offical openCV documentation, the CMakeLists.txt file should indicate c++11 as well.

```
cmake_minimum_required(VERSION 2.8)
project( project_name )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( executable_name source_code_name.cpp )
target_link_libraries( executable_name ${OpenCV_LIBS} )
set_target_properties( executable_name PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
```

## NOTICE: Timing sequential code and CUDA code in openCV

Usually, we would like to compare the runtime between normal (sequential) openCV version and CUDA version to see the difference and the speedup (if any) of our implementation.

Because CUDA needs to be initialized, it **SHOULD NOT** be timed at the first run. In this run, the initialization of CUDA Rntime API happens implicitly. In addition, some GPU setting will also run to make a configuration for the first usage. Therefore, in order to do a performance measure, the first run of CUDA version should be a dummy trial (or some dummy function) before performing the actual test.
