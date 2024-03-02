#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings 用于格式化字符串
//解决不了错误，解决出错的代码 #include <pcl/point_typclpes.h>  
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>


////////和/home/shark/slambook2/ch5/rgbd/joinMap.cpp 基本一致////////


int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    vector<Eigen::Isometry3d> poses;         // 相机位姿

    ifstream fin("./data/pose.txt");
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) 
    {
        boost::format fmt("./data/%s/%d.%s"); //图像文件格式  使用 boost::format 构造图像文件的路径格式。通过循环读取，读取彩色图像和深度图像，然后存储到对应的向量中。
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // 使用-1读取原始图像

        double data[7] = {0};
        for (int i = 0; i < 7; i++) {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]); //四元数
        Eigen::Isometry3d T(q); //根据四元数创造变换矩阵
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2])); //左乘平移向量
        /*
        读取相机位姿的数据，其中包括一个平移向量和一个四元数。通过四元数构造一个 Eigen::Quaterniond 对象，并使用它初始化一个 Eigen::Isometry3d 对象 T。
`       使用 T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2])); 将平移向量左乘到变换矩阵中，得到最终的相机位姿，并将其存储到 poses 向量中。
        */
        poses.push_back(T); 
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;
    //depthScale 是用于将深度值从原始深度图的像素单位转换为真实世界中的尺度的缩放因子。
    //深度图通常以毫米（mm）为单位表示每个像素的深度值，但在计算点云时，可能需要将深度值转换为米（m）等更方便的尺度。
    cout << "正在将图像转换为点云..." << endl;

    // 定义点云使用的格式：这里用的是XYZRGB
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    // 这里定义了两个类型别名，PointT 代表点的类型，PointCloud 代表点云的类型。点类型 PointXYZRGB 包含三维坐标和 RGB 颜色信息。

    // 新建一个点云
    PointCloud::Ptr pointCloud(new PointCloud);
    //PointCloud::Ptr 是 pcl::PointCloud<PointT>::Ptr 的别名，表示指向 pcl::PointCloud<PointT> 类型的智能指针。

    //循环处理每个时间步的图像数据：
    for (int i = 0; i < 5; i++) {
        //新建当前时间步的点云对象：
        PointCloud::Ptr current(new PointCloud);//为当前时间步新建一个指向 PointCloud 类型的智能指针 current，用于存储当前时间步的点云数据。

        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];  //颜色数据
        cv::Mat depth = depthImgs[i];  //深度数据
        Eigen::Isometry3d T = poses[i];//位姿数据
        //图像转换为点云：
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                //depth.ptr<unsigned short>(v): 获取深度图的第 v 行的指针，返回一个指向该行起始位置的指针。[u]: 通过偏移 u 来访问该行中第 u 个元素的深度值，并将其赋值给变量 d。
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()]; // blue
                //将深度图中 (u, v) 处像素的蓝色通道值（颜色值范围为0到255）赋给 p 的第六个元素。
                p.g = color.data[v * color.step + u * color.channels() + 1]; // green
                // 对于典型的彩色图像，通道顺序通常是蓝色（channel 0）、绿色（channel 1）、红色（channel 2）。通过使用 + 1，你能够获取到绿色通道的值。同样，+ 2 将允许你获取到红色通道的值。
                p.r = color.data[v * color.step + u * color.channels() + 2]; // red
                current->points.push_back(p); //current 是一个指向 PointCloud 的智能指针，它代表了当前时间步骤的点云。current->points.push_back(p) 利用点云对象的 points 成员（一个存储点的容器），将当前点 p 添加到点云中。
            }

        // depth filter and statistical removal 
        // 点云滤波：
        PointCloud::Ptr tmp(new PointCloud);
        //创建临时点云对象 tmp：  创建了一个指向 PointCloud 类型的智能指针 tmp，用于存储滤波后的点云数据。
        
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        //创建统计滤波器对象 statistical_filter：  创建了一个 pcl::StatisticalOutlierRemoval 类型的对象 statistical_filter，用于执行统计滤波。
        
        //设置统计滤波器的参数：
        statistical_filter.setMeanK(50);//setMeanK(50): 设置用于计算点云中每个点的邻近点的数量。这里设置为50，表示每个点的邻域中包含50个最近的点。
        statistical_filter.setStddevMulThresh(1.0);//setStddevMulThresh(1.0): 设置标准差倍数阈值，用于确定点是否被认为是离群点。这里设置为1.0，表示标准差的倍数阈值为1.0。
        statistical_filter.setInputCloud(current);//将当前时间步的点云 current 设置为统计滤波器的输入。
        statistical_filter.filter(*tmp);//调用 filter 函数，将滤波后的点云存储在 tmp 中。这里会移除掉被认为是离群点的部分。

        //整体点云拼接：
        //将当前时间步处理后的点云 tmp 加入整体点云 pointCloud 中。
        (*pointCloud) += *tmp;
        //这个过程的目的是通过统计滤波去除点云中的离群点，以提高整体点云的质量和准确性。这对于许多应用场景，特别是在三维重建和SLAM中，是一个常见的预处理步骤。
    }

    pointCloud->is_dense = false;
    cout << "点云共有" << pointCloud->size() << "个点." << endl;

    // voxel grid filter  体素网格的降采样滤波 
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.03; // 体素滤波器的分辨率设置为0.03，表示每个0.03*0.03*0.03的格子中只存在一个点。这是一个比较高的分辨率
    voxel_filter.setLeafSize(resolution, resolution, resolution);       // resolution
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointCloud);//通过交换智能指针指向的内容，将滤波后的点云 tmp 的数据传递给整体点云 pointCloud。这样可以避免在大量数据操作时发生不必要的拷贝，提高效率。

    cout << "滤波之后，点云共有" << pointCloud->size() << "个点." << endl;

    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;
}