cmake_minimum_required(VERSION 2.8)
project( ORBTutorial )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( orbDetector orbDetector.cpp )
target_link_libraries( orbDetector ${OpenCV_LIBS} )
set_target_properties( orbDetector PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
