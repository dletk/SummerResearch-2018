# SummerResearch-2018

## Jetson TX2 set up

Install the Jetpack version 3.2 from NVIDIA.

*Attention*: The host machine and the TX2 should connect to the same router (so they can find each other by IP address later on). Using ```ifconfig``` to check for the address.

## Setting up the openCV 3.4 on TX2

The version of openCV on the TX2 installed by Jetpack is not complete and missing some libraries. Uninstall it and follow the guidelines [here](https://jkjung-avt.github.io/opencv3-on-tx2/).

```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_CUDA=ON -D CUDA_ARCH_BIN="6.2" -D CUDA_ARCH_PTX="" \
        -D WITH_CUBLAS=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	   -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON \
        -D ENABLE_NEON=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
        -D WITH_QT=ON -D WITH_OPENGL=ON ..
```
