#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamplesPointcloud)

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

find_package(realsense2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

# Find glfw header
# find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h /usr/include)

# Find glfw library
# find_library(GLFW_LIBRARIES libglfw3 /usr/lib)

# Include directories for this project
set(INCLUDE_PATH 
    ${realsense2_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ../)

#  Libraries needed for all platforms for this project
set(LIBRARIES 
    ${realsense2_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${GLFW_LIBRARIES})
#pointcloud
include_directories(${INCLUDE_PATH})
add_executable(rs-pointcloud rs-pointcloud.cpp ../example.hpp)
target_link_libraries(rs-pointcloud ${LIBRARIES})
