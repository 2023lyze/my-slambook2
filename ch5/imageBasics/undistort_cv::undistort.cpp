#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png";   // 请确保路径正确

int main(int argc, char **argv) {
    // 畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file, 0);   // 图像是灰度图，CV_8UC1

    // 畸变参数矩阵
    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, 0);

    // 内参矩阵
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    cv::Mat image_undistort;
    
    // 使用cv::undistort函数进行去畸变
    cv::undistort(image, image_undistort, cameraMatrix, distCoeffs);

    // 画图显示
    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey();

    return 0;
}
