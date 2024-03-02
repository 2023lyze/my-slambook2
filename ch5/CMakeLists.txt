cmake_minimum_required(VERSION 2.8)
project(RGBD)
set(CMAKE_BUILD_TYPE "Release")
# 添加c++ 11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
find_package(OpenCV 4 REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(/home/shark/ku/opencv/include)
message(STATUS "---------------------------------------------")
message(STATUS "1.OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "---------------------------------------------")
# include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)
