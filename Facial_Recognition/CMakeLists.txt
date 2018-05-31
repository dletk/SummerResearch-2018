cmake_minimum_required(VERSION 2.8)
project( FacialRecognition )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( createHOGDetector createHOGDetector.cpp )
target_link_libraries( createHOGDetector ${OpenCV_LIBS} )
set_target_properties( createHOGDetector PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
