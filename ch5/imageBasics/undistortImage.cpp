#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png";   // 请确保路径正确

int main(int argc, char **argv) {

  // 本程序实现去畸变部分的代码。尽管我们可以调用OpenCV的去畸变（cv::Undistort），但自己实现一遍有助于理解。
  // 畸变参数
  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
  // 内参
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

  cv::Mat image = cv::imread(image_file, 0);   // 图像是灰度图，CV_8UC1
  /*
    在 OpenCV 中，cv::imread 函数的第二个参数是用于指定图像读取的颜色模式（color mode）或图像加载标志（image loading flags）。
    在这里，参数 0 表示以灰度图像的形式加载图像。
    具体的颜色模式或图像加载标志可以影响 cv::imread 函数的行为。一些常见的加载标志包括：
    cv::IMREAD_COLOR (或者简单地使用 1)：加载彩色图像，这是默认值。
    cv::IMREAD_GRAYSCALE (或者简单地使用 0)：以灰度模式加载图像。
    cv::IMREAD_UNCHANGED (或者简单地使用 -1)：加载图像，包括 alpha 通道（如果有的话）
  */
  int rows = image.rows, cols = image.cols;
  cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // 去畸变以后的图

  // 计算去畸变后图像的内容
  for (int v = 0; v < rows; v++) {
    for (int u = 0; u < cols; u++) {
      // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
      double x = (u - cx) / fx, y = (v - cy) / fy;  //这是三维投影点投影到归一化图像平面的归一化坐标；
      double r = sqrt(x * x + y * y);
      double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
      double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
      double u_distorted = fx * x_distorted + cx;
      double v_distorted = fy * y_distorted + cy;

      // 赋值 (最近邻插值)
      if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
        image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
      } else {
        image_undistort.at<uchar>(v, u) = 0;
      }
      /*
      .at<uchar>(v, u) 返回图像在坐标 (v, u) 处的像素值，并且这个值的数据类型是无符号的 8 位整数（0 到 255 之间的整数，对应于灰度图像的像素强度）。
      .at<uchar>(v, u): 这表示从图像中获取位于 (v, u) 处的像素值。在这里，v 是行坐标，u 是列坐标。
      <uchar>: 这是模板参数，表示你想要获取的像素值的数据类型。在这里，uchar 表示无符号的 8 位整数，即灰度图像的像素值的数据类型。
      */
    }
  }

  // 画图去畸变后图像
  cv::imshow("distorted", image);
  cv::imshow("undistorted", image_undistort);
  cv::waitKey();
  return 0;
}
