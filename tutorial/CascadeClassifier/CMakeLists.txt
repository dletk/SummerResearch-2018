cmake_minimum_required(VERSION 2.8)
project( CascadeClassifier )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( hharCascade hharCascade.cpp )
target_link_libraries( hharCascade ${OpenCV_LIBS} )
set_target_properties( hharCascade PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
