# SummerResearch-2018

## Jetson TX2 set up

Install the Jetpack version 3.2 from NVIDIA.

*Attention*: The host machine and the TX2 should connect to the same router (so they can find each other by IP address later on). Using ```ifconfig``` to check for the address.

## Setting up the openCV 3.4 on TX2

The version of openCV on the TX2 installed by Jetpack is not complete and missing some libraries. Uninstall it and follow the guidelines [here](https://jkjung-avt.github.io/opencv3-on-tx2/).

```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D CUDA_GENERATION=Auto -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_NVCUVID=ON -D WITH_CUFFT=ON -D WITH_EIGEN=ON -D WITH_IPP=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
```
