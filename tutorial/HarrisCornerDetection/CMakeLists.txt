cmake_minimum_required(VERSION 2.8)
project( HarrisCorner )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( harris harrisCorner.cpp )
target_link_libraries( harris ${OpenCV_LIBS} )
set_target_properties( harris PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
