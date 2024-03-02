//
// Created by gaoxiang on 19-4-25.
//

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/impl/mls.hpp>

// typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointXYZRGBNormal SurfelT; //pcl::PointXYZRGBNormal。这种类型通常用于表示具有三维坐标、RGB颜色和法线信息的点云数据。
typedef pcl::PointCloud<SurfelT> SurfelCloud;
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr;

//定义了reconstructSurface 函数，使用Moving Least Squares（MLS）表面重建方法，对输入的点云进行表面重建，并返回表面元素的点云。MLS方法有助于平滑和拟合表面。
SurfelCloudPtr reconstructSurface(
        const PointCloudPtr &input, float radius, int polynomial_order) 
        {
    // 创建 Moving Least Squares（MLS）对象
    pcl::MovingLeastSquares<PointT, SurfelT> mls;
    // 创建 KdTree 用于搜索
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    mls.setSearchMethod(tree);

    // 设置 MLS 参数
    mls.setSearchRadius(radius);
    mls.setComputeNormals(true);
    mls.setSqrGaussParam(radius * radius);
    mls.setPolynomialFit(polynomial_order > 1);
    mls.setPolynomialOrder(polynomial_order);

    // 设置输入点云
    mls.setInputCloud(input);

    // 创建输出点云对象
    SurfelCloudPtr output(new SurfelCloud);

    // 执行 MLS 表面重建
    mls.process(*output);

    // 返回表面元素的点云
    return (output);
        }

//定义了一个函数 triangulateMesh，用于对表面元素的点云进行三角化，生成三角网格
pcl::PolygonMeshPtr triangulateMesh(const SurfelCloudPtr &surfels) 
{//const SurfelCloudPtr &surfels: 表面元素的点云智能指针。这是函数的输入，表示要进行三角化的点云数据。
    // Create search tree  s创建搜索树
    pcl::search::KdTree<SurfelT>::Ptr tree(new pcl::search::KdTree<SurfelT>); //创建一个 KdTree 用于搜索，将表面元素的点云设置为搜索树的输入。
    tree->setInputCloud(surfels);

    // Initialize objects 初始化对象
    pcl::GreedyProjectionTriangulation<SurfelT> gp3; //创建一个 Greedy Projection Triangulation 对象，用于进行三角化。
    pcl::PolygonMeshPtr triangles(new pcl::PolygonMesh); //创建一个指向 pcl::PolygonMesh 类型的智能指针，用于存储三角网格。

    // Set the maximum distance between connected points (maximum edge length)  设置连接点之间的最大距离（最大边长）
    gp3.setSearchRadius(0.05); //设置搜索半径，表示连接点之间的最大距离，即最大边长。

    // Set typical values for the parameters 设置参数的典型值
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees 度
    gp3.setMinimumAngle(M_PI / 18); // 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(true);

    // Get result 获取结果
    gp3.setInputCloud(surfels); //将表面元素的点云设置为输入。
    gp3.setSearchMethod(tree); //将搜索树设置为搜索方法。
    gp3.reconstruct(*triangles); //执行三角化，将结果存储在 triangles 中。

    return triangles;
}

int main(int argc, char **argv) {

    // Load the points 加载PCD文件中的点云数据
    PointCloudPtr cloud(new PointCloud);
    if (argc == 0 || pcl::io::loadPCDFile(argv[1], *cloud)) {
        cout << "failed to load point cloud!";
        return 1;
    }
    cout << "point cloud loaded, points: " << cloud->points.size() << endl;

    // Compute surface elements 表面重建
    cout << "computing normals ... " << endl;
    double mls_radius = 0.05, polynomial_order = 2;
    auto surfels = reconstructSurface(cloud, mls_radius, polynomial_order);

    // Compute a greedy surface triangulation
    cout << "computing mesh ... " << endl;

    //三角化
    pcl::PolygonMeshPtr mesh = triangulateMesh(surfels);

    cout << "display mesh ... " << endl;
    //可视化结果：
    pcl::visualization::PCLVisualizer vis;
    vis.addPolylineFromPolygonMesh(*mesh, "mesh frame");
    vis.addPolygonMesh(*mesh, "mesh");
    vis.resetCamera();
    vis.spin();
}