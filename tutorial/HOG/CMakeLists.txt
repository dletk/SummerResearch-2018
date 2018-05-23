cmake_minimum_required(VERSION 2.8)
project( HOG )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( creatingHogSobel creatingHogSobel.cpp )
target_link_libraries( creatingHogSobel ${OpenCV_LIBS} )
set_target_properties( creatingHogSobel PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)
