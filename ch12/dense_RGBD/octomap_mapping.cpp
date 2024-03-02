#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h>    // for octomap 

#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings


/*根据RGB-D图像和相机的位姿信息，先将点的坐标转至世界坐标，然后放入八叉树的点云，
  最后交给八叉树地图，之后，八叉树地图会根据投影信息，跟新内部的占据概率，保存成压缩后的八叉树地图*/


int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    vector<Eigen::Isometry3d> poses;         // 相机位姿

    ifstream fin("./data/pose.txt"); // 创建流对象
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }
    // 读取图像和位姿信息：
    for (int i = 0; i < 5; i++) {
        boost::format fmt("./data/%s/%d.%s"); //图像文件格式
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // 使用-1读取原始图像

        double data[7] = {0};
        for (int i = 0; i < 7; i++) {
            fin >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "正在将图像转换为 Octomap ..." << endl;

    // octomap tree 
    octomap::OcTree tree(0.01); // 参数为分辨率
    //创建了一个OctoMap地图对象 tree，设置分辨率为0.01。

    // 遍历每一帧图像，获取彩色图、深度图和相机位姿。
    for (int i = 0; i < 5; i++) 
    {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];

        octomap::Pointcloud cloud;  // the point cloud in octomap  八叉树地图提供的点云结构
        // 处理每个像素点，构建点云
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) 
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                // 将世界坐标系的点放入点云
                //遍历每个像素点，通过深度信息计算点的三维坐标，然后将这些世界坐标系下的点添加到OctoMap提供的点云结构 cloud 中。
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
            }

        // 将点云存入八叉树地图，给定原点，这样可以计算投射线
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3))); // 这一行代码的目的是将点云数据 cloud 插入到 OctoMap 八叉树地图中。
        /*
        在插入点云时，需要提供点云的原点，这是为了确定点云在地图中的位置。这个原点的选择通常与相机位姿相关。
        octomap::point3d(T(0, 3), T(1, 3), T(2, 3)) 创建了一个 octomap::point3d 对象，表示点云的原点。这里使用了相机位姿矩阵 T 的第四列（T(0, 3)，T(1, 3)，T(2, 3)）作为原点的坐标。
        为什么要设置原点？!!!!!八叉树地图需要知道点云在全局坐标系中的位置，以便正确插入和更新地图。！！！！！
        通过提供原点，可以将点云从相机坐标系转换到全局坐标系，并将其插入到正确的位置。
        */
    }

    // 更新中间节点的占据信息并写入磁盘
    tree.updateInnerOccupancy();
    cout << "saving octomap ... " << endl;
    tree.writeBinary("octomap.bt");// 更新OctoMap地图中间节点的占据信息，并将地图以二进制格式写入磁盘文件 "octomap.bt"。
    return 0;
}
