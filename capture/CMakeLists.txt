#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamplesCapture)
set(OpenCV_DIR /home/huynv/installation/OpenCV-3.4.4/share/OpenCV)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ${realsense2_INCLUDE_DIR})
add_executable(capture capture.cpp)
target_link_libraries(capture ${OpenCV_LIBS} ${realsense2_LIBRARY})
