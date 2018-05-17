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

## Using Jetson TX2 onboard webcam

In order to use the onboard webcam of the Jetson TX2, the openCV VideoCapure code needs to receive several inputs indicating the setting for the webcam. The complete guide of NVIDIA for this webcam can be found [here](https://developer.download.nvidia.com/embedded/L4T/r24_Release_v2.0/Docs/L4T_Tegra_X1_Multimedia_User_Guide_Release_24.2.pdf?X3y2hPgEp4kolJ_SW-jeGK3DRxEXakPWxExnt0S2WM3LoFnDeOXAvCGaFjm7NxTenIo5MHRsEYEAaUcA3DazzNmwEe45VNRPq1REAqhxHiIZYCtxLGj1uRgyJO-xisdsXLg-gkbfPNDLyXZU6Vwp6nz1JV2gSXgfFdPhFtwSsfpZVeq_Uzl9bl38twCJKe9lAHYSnus). However, most of the time we only pay attention to the normal, raw input, which can be established as follow:

```
const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,\
			format=(string)I420, framerate=(fraction)120/1 ! \
			nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! \
			videoconvert ! video/x-raw, format=(string)BGR ! \
			appsink";

VideoCapture vidCap(gst);
```

The setting above uses the default flip-method, 0, which indicates no rotation. The complete sets of flip-method option involve 8 different ways of rotating the raw input:

| Flip method | Property Value|
|-------------|--------------:|
|identity - no rotation (default) | 0 |
|counterclockwise - 90 degrees | 1 |
|rotate-180 degrees| 2 |
|clockwise - 90 degrees | 3 |
|horizontal flip| 4 |
|upper right diagonal flip| 5 |
|vertical flip| 6 |
|uppet-left diagonal | 7 |
