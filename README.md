# SummerResearch-2018

## Jetson TX2 set up

Install the Jetpack version 3.2 from NVIDIA.

*Attention*: The host machine and the TX2 should connect to the same router (so they can find each other by IP address later on). Using ```ifconfig``` to check for the address.

## Setting up the openCV library on Jetson TX2

The version of openCV on the TX2 installed by Jetpack is not complete and missing some libraries. Follow the guidelines [here](https://jkjung-avt.github.io/opencv3-on-tx2/) to uninstall it and compile a new version. **ONLY FOLLOW THAT GUIDELINE UP TO THE POINT OF MODYFYING libGL**. Then, continue with these instructions to install openCV.

1. Clone the newest version of openCV from its source code

By the time of writing, the newest version is 4.0.0.

```
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

2. Prepare to build openCV from Source using CMake

```
cd opencv
mkdir build
cd build
```

3. Enter the following command to initialize the cmake.

```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_CUDA=ON -D CUDA_ARCH_BIN="6.2" -D CUDA_ARCH_PTX="" \
        -D WITH_CUBLAS=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	-D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON \
        -D ENABLE_NEON=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
	-D ENABLE_PRECOMPILED_HEADERS=OFF \
        -D WITH_QT=ON -D WITH_OPENGL=ON ..
```

**IMPORTANT:** The command above turns off precompiled headers option because the limitation of storage space on the TX2. Also, it inlcludes the support of opencv\_contrib, which is a collections of libraries that is free of use for academic purpose only. These libraries are not well maintaiend and tested, so they are not inlcuded in the official release. However, many of them are useful for important tasks of computer vision program. Please read more above this repository from [here](https://github.com/opencv/opencv_contrib) and decide whether you want to have it or not. Delete the option ```-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules``` if you don't.

4. Make the project

```
make
```

This option is usually recommended as ```make -j``` in order to run on several threads. However, this often leads to a laggy situation on the performance of the TX2, and sometimes freezes the whole process. Therefore, I have decided not to use it. That being said, this process can be slower than normal (potentially hours).

After the making process, execute this command to install the libraries.

```
sudo make install
```

Then restart your terminal and enter the following python code. If the returned result should be the version of the installed openCV library.

```
$ python3 -c 'import cv2; print(cv2.__version__)'
4.0.0-pre
```

We are all good!


