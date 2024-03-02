#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // 用于格式化字符串
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>   // Sophus 库提供的 SE3 类型

using namespace std;

// 定义一些类型别名
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;


// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv) {
    // 存储彩色图、深度图和相机位姿的容器
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    TrajectoryType poses;         // 相机位姿  7位，X,Y,Z,qx,qy,qz,qw  为平移向量加上四元数   
    // 打开包含相机位姿的文件 "pose.txt"
    ifstream fin("./pose.txt");//这行代码声明了一个输入文件流对象 fin，并尝试打开名为 "pose.txt" 的文件进行读取。"./" 表示当前工作目录。这行代码的意思是尝试打开当前工作目录下的 "pose.txt" 文件。
    if (!fin) {//文件没有成功打开（即 fin 的状态为假）
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }
    // 读取彩色图、深度图和相机位姿信息，构建点云
    for (int i = 0; i < 5; i++) {
        // 使用 Boost 的 format 类构造图像文件路径格式
        boost::format fmt("./%s/%d.%s");
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); // 使用-1读取原始图像
        /*
        在这一行代码中，cv::imread 函数加载深度图像。深度图像可能以不同的格式存储，常见的有灰度图像（单通道图像）和16位无符号整数图像（16UC1）。
        在这里，深度图像采用16位无符号整数格式（.pgm 文件通常以这种格式存储），因此使用 cv::imread 时，读取的是原始图像数据，不进行颜色解析。
        参数 -1 表示以原始格式加载图像，而不进行任何颜色解析或通道转换。
        这对于读取深度信息等特殊图像格式很有用，因为深度图通常以灰度图像或特殊的格式存储，不需要进行常规的颜色通道解析。
        所以，这里的 -1 选项确保深度图像被加载为原始数据，以便后续代码能够直接使用图像中的深度值。
        */
        double data[7] = {0};
        for (auto &d : data) // 范围循环
            fin >> d;

        // 使用 Sophus 库构造 SE3 类型的相机位姿，前四个数字代表旋转，后三个数字代表平移
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                          Eigen::Vector3d(data[0], data[1], data[2]));

        // 存储相机位姿
        poses.push_back(pose);
    }

    // 计算点云并拼接  内参K
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;
    //depthScale 是用于将深度值从原始深度图的像素单位转换为真实世界中的尺度的缩放因子。
    //深度图通常以毫米（mm）为单位表示每个像素的深度值，但在计算点云时，可能需要将深度值转换为米（m）等更方便的尺度。
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);
    //pointcloud.reserve(1000000); 这行代码用于预留 pointcloud 向量的容量，使用 reserve 可以避免每次添加新点时都重新分配内存，从而提高效率。

    for (int i = 0; i < 5; i++) 
    {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];

        for (int v = 0; v < color.rows; v++)
        {
            for (int u = 0; u < color.cols; u++) 
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                //depth.ptr<unsigned short>(v): 获取深度图的第 v 行的指针，返回一个指向该行起始位置的指针。[u]: 通过偏移 u 来访问该行中第 u 个元素的深度值，并将其赋值给变量 d。
                if (d == 0) continue; // 为0表示没有测量到  continue 在==循环语句==中，跳过本次循环中余下尚未执行的语句，继续执行下一次循环

                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;

                // 通过相机位姿将相机坐标系下的点转换到世界坐标系
                Eigen::Vector3d pointWorld = T * point;

                // 存储点的坐标和颜色信息
                Vector6d p;
                //这部分代码将三维点的坐标和颜色信息组合成一个 Vector6d 类型的向量，并将该向量添加到 pointcloud 向量中。这个向量的定义是,typedef Eigen::Matrix<double, 6, 1> Vector6d;
                p.head<3>() = pointWorld;
                //将 pointWorld 中的前三个元素（x、y、z坐标）复制给 p 的前三个元素。这是 Eigen 库的语法，表示将 pointWorld 的前三个元素赋值给 p 的前三个元素。
                p[5] = color.data[v * color.step + u * color.channels()];   // blue
                //将深度图中 (u, v) 处像素的蓝色通道值（颜色值范围为0到255）赋给 p 的第六个元素。
                //color.data 是图像数据的指针，v * color.step + u * color.channels() 是获取图像中 (u, v) 处像素的索引。
                /*
                color.data: 这是图像数据的指针，它指向图像的第一个像素的内存位置。
                v * color.step: 这是垂直方向上的偏移，其中 color.step 表示图像的行步幅，即每一行像素占用的字节数。
                u * color.channels(): 这是水平方向上的偏移，其中 color.channels() 表示每个像素的通道数，即每个像素由多少个颜色通道组成。
                将上述偏移相加，得到的值是 (u, v) 处像素在图像数据中的索引。然后通过加上颜色通道的偏移，可以访问到该像素的蓝色通道值。
                */
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                //对于典型的彩色图像，通道顺序通常是蓝色（channel 0）、绿色（channel 1）、红色（channel 2）。通过使用 + 1，你能够获取到绿色通道的值。同样，+ 2 将允许你获取到红色通道的值。
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);//pointcloud.push_back(p);: 将构建好的 Vector6d 向量添加到 pointcloud 向量中，这样就存储了每个点的三维坐标和颜色信息。
            }
        }   
    }

    // 输出点云信息
    cout << "点云共有 " << pointcloud.size() << " 个点." << endl;

    // 在 Pangolin 中显示点云
    showPointCloud(pointcloud);

    return 0;
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {
    // 检查点云是否为空
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    // 设置 Pangolin 窗口和渲染状态
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
        for (auto &p : pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }

        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}