cmake_minimum_required(VERSION 2.8)
project( NeuralNetwork )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( tutorialOpenCVNN tutorialOpenCVNN.cpp )
target_link_libraries( tutorialOpenCVNN ${OpenCV_LIBS} )
set_target_properties( tutorialOpenCVNN PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)

