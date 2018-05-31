cmake_minimum_required(VERSION 2.8)
project( HOG )
add_subdirectory(../../Facial_Recognition/dlib dlib_build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( hogDetector hogDetector.cpp )
target_link_libraries( hogDetector ${OpenCV_LIBS} dlib::dlib)
set_target_properties( hogDetector PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
