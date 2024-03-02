project(imageBasics)

# 寻找OpenCv库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imageBasics imageBasics.cpp)
add_executable(undistortImage undistortImage.cpp)
add_executable(undistort_cv_undistort undistort_cv_undistort.cpp)
# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})
# 链接OpenCV库
target_link_libraries(undistortImage ${OpenCV_LIBS})
# 链接OpenCV库
target_link_libraries(undistort_cv_undistort ${OpenCV_LIBS})
