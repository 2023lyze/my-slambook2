#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;

// 文件路径
string left_file = "./left.png";
string right_file = "./right.png";

// 在pangolin中画图，已写好，无需调整
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char **argv) {

    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线
    double b = 0.573;

    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);  // 以灰度图的方式读取左图像
    cv::Mat right = cv::imread(right_file, 0);  // 以灰度图的方式读取右图像

    // 创建StereoSGBM对象并设置参数
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);  // 设置SGBM算法参数

    // 计算视差图
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    /*
    这段代码的目标是计算图像左右视图之间的视差图，并将其转换为点云。
    cv::Mat disparity_sgbm, disparity;:
    创建两个 OpenCV 的 cv::Mat 对象，disparity_sgbm 用于存储原始的整型视差图，而 disparity 用于存储经过转换后的浮点型视差图。
    sgbm->compute(left, right, disparity_sgbm);:
    使用之前创建的 StereoSGBM 对象 sgbm 计算左右图像之间的视差。
    left 和 right 是左右相机拍摄的灰度图像。
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);:
    将原始的整型视差图 disparity_sgbm 转换为浮点型视差图 disparity。
    CV_32F 指定输出的数据类型为 32 位浮点型。
    1.0 / 16.0f 是一个缩放因子，用于将原始视差值缩小，以得到更为合适的浮点表示。这通常与 SGBM 算法输出的视差图的数据范围相关。
    */

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    // 遍历视差图，生成点云
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) {
            // 如果视差值小于等于0或大于等于96，跳过当前像素
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0);  // 前三维为xyz，第四维为颜色

            // 根据双目模型计算点的位置
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
            /*
            at<float>(v, u) 是 OpenCV 中 cv::Mat 类的成员函数，用于获取矩阵（图像）中特定位置 (v, u) 处的像素值。
            at<float>(v, u)): 这表示从图像中获取位于 (v, u) 处的像素值。在这里，v 是行坐标，u 是列坐标。
            这里的 float 表示视差图的像素值的数据类型是单精度浮点数。在视差图中，每个像素的值表示对应点的视差（或深度）信息。
            */
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);
        }

    // 显示视差图
    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);

    // 画出点云
    showPointCloud(pointcloud);

    return 0;
}

// 通过Pangolin库显示点云
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    // 设置Pangolin视窗
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    /*
    启用深度测试。深度测试是OpenGL中一种用于确定哪些像素应该被渲染到屏幕上的技术。
    启用深度测试后，OpenGL会根据每个像素的深度值与深度缓冲区中的值进行比较，决定是否渲染该像素。
    */
    glEnable(GL_BLEND);
    /*
    启用混合。混合是指在绘制像素时将其颜色与背景颜色混合的过程。启用混合后，你可以通过设置混合因子来调整颜色的透明度。
    GL_BLEND 是一个 OpenGL 定义的常量，它表示启用混合功能。glEnable(GL_BLEND) 的调用告诉 OpenGL 开启混合。
    */
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    /*
    设置混合因子。这里使用的混合因子是源颜色的 alpha 值（GL_SRC_ALPHA）与目标颜色的 alpha 值的互补值（GL_ONE_MINUS_SRC_ALPHA）。
    这种混合因子通常用于实现透明效果，其中源颜色的透明度（alpha 值）控制了最终颜色的透明度。
    */

    // 设置相机参数
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);

        // 遍历点云，画出每个点
        // 范围循环
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);  // 根据点的颜色设置绘制颜色
            glVertex3d(p[0], p[1], p[2]);  // 绘制点
        }

        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}




/*
这个双目相机构建点云地图的代码，并没有提供相机的位姿消息，为什么能构建点云地图？
在提供的代码中，虽然没有明确提供相机的位姿信息，但通过双目相机的视差图计算，可以在图像空间中获取每个像素对应的三维点坐标，从而构建点云地图。
在双目视觉中，通过匹配左右相机拍摄的图像，可以计算视差图，即每个像素在水平方向上的偏移量。这个视差图包含了深度信息，而深度信息可以用于计算每个像素对应的三维空间坐标。
在代码的点云生成部分，通过以下计算得到了每个点的三维坐标：
double x = (u - cx) / fx;
double y = (v - cy) / fy;
double depth = fx * b / (disparity.at<float>(v, u));
point[0] = x * depth;
point[1] = y * depth;
point[2] = depth;
其中，u 和 v 分别是像素在图像中的列坐标和行坐标，cx 和 cy 是图像的光学中心坐标，fx 和 fy 是相机的焦距。
b 是相机的基线长度，而 disparity.at<float>(v, u) 则是对应像素的视差值。通过这些信息，可以计算出每个像素对应的三维坐标。


尽管相机的位姿信息对于进行更精确的三维重建和地图构建非常重要，但在一些简单的应用场景中，通过仅仅使用视差图的信息，也能得到一个初步的点云地图。
这样的点云地图可能不具有绝对的尺度和准确的姿态，但仍然可以用于某些应用，如避障、目标检测等。
*/