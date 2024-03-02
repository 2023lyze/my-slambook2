cmake_minimum_required(VERSION 3.0)
project(useSophus)

# 为使用 sophus，需要使用find_package命令找到它

include_directories("/usr/include/eigen3")
set(CMAKE_BUILD_TYPE "Debug")
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS}) 
add_executable( useSophus useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus fmt)


add_subdirectory(example)
  