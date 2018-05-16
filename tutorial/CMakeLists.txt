cmake_minimum_required(VERSION 2.8)
project( ImageRead )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( imageRead image.cpp )
target_link_libraries( imageRead ${OpenCV_LIBS} )
set_target_properties( imageRead PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
