cmake_minimum_required(VERSION 2.8)
project( tutorial )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( videoCap videoCap.cpp )
target_link_libraries( videoCap ${OpenCV_LIBS} )
set_target_properties( videoCap PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
