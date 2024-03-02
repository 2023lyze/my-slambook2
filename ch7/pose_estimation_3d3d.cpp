#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>

//书中使用的是opencv3.1，现在使用的是opencv4.4，CV_LOAD_IMAGE_COLOR旧版的声明已经没了，但是我可以在constants_c.h 这个头文件中获得
#include "opencv2/imgcodecs/legacy/constants_c.h"


using namespace std;
using namespace cv;

// 函数声明
// 声明一个函数，用于在两个图像中找到特征点并进行匹配
void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// 声明一个函数，用于像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// 定义一个函数，用于实现3D-3D位姿估计(利用SVD方法)
void pose_estimation_3d3d(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t
);

// 声明一个函数，进行图优化的捆绑调整
void bundleAdjustment(
  const vector<Point3f> &points_3d,
  const vector<Point3f> &points_2d,
  Mat &R, Mat &t
);

/// vertex and edges used in g2o ba
// 定义一个继承自g2o::BaseVertex的顶点类，表示相机位姿
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};

/// g2o edge
// 定义一个继承自g2o::BaseUnaryEdge的g2o边类，表示3D点的投影
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

  virtual void computeError() override {
    const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
    _error = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override {
    VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d xyz_trans = T * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }

  bool read(istream &in) {}

  bool write(ostream &out) const {}

protected:
  Eigen::Vector3d _point;
};

// 主函数
int main(int argc, char **argv) {
  // 检查命令行参数是否正确
  if (argc != 5) {
    cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  //vector<KeyPoint> 中的 KeyPoint 是指关键点的数据结构，在OpenCV中，KeyPoint 类包含了关键点的信息，如坐标、响应值、角度等。
  vector<DMatch> matches;

  // 在两个图像中找到特征点并进行匹配
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;

  // 读取深度图和相机内参
  Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); //相机的内参矩阵
  vector<Point3f> pts1, pts2;

  // 根据匹配的特征点，构建三维点云。
  for (DMatch m:matches) {
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    // 检查深度值是否有效（不为零）
    if (d1 == 0 || d2 == 0)   // bad depth
      continue;
    // 将像素坐标转换为相机归一化坐标
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    // 将深度值转换为浮点数，并进行归一化
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    // 构建三维点，并将其添加到pts1和pts2中
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
  }

  cout << "3d-3d pairs: " << pts1.size() << endl;

/*----------------------------------------------------------------------------*/
  // 使用SVD方法进行相对位姿估计
  Mat R, t;
  pose_estimation_3d3d(pts1, pts2, R, t);
  cout << "ICP via SVD results: " << endl;
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;
  cout << "R_inv = " << R.t() << endl;
  cout << "t_inv = " << -R.t() * t << endl;

  cout << "calling bundle adjustment" << endl;

  // 进行图优化的捆绑调整   ----？
  //bundleAdjustment 函数通过优化相机位姿和3D点，以最小化观测和测量之间的误差，从而提高SLAM系统的精度。
  bundleAdjustment(pts1, pts2, R, t);

  // verify p1 = R * p2 + t
  // 验证优化后的结果
  for (int i = 0; i < 5; i++) {
    cout << "p1 = " << pts1[i] << endl;
    cout << "p2 = " << pts2[i] << endl;
    cout << "(R*p2+t) = " <<
         R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
         << endl;
    cout << endl;
  }
}

//函数用于在两个图像中找到特征点并进行匹配
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);
  //detector: 这是先前创建的 ORB 特征检测器的指针，它指向一个 ORB 特征检测器对象。
  //detect(img_1, keypoints_1): 这是特征检测器的成员函数，用于在输入图像 img_1 中检测关键点。检测到的关键点将被存储在 keypoints_1 向量中。


  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  //descriptor: 这是先前创建的 ORB 特征描述器的指针，它指向一个 ORB 特征描述器对象。
  //compute(img_1, keypoints_1, descriptors_1): 这是特征描述器的成员函数，用于计算给定图像 img_1 中指定关键点 keypoints_1 的特征描述子。计算结果将被存储在 descriptors_1 中。


  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //matcher: 这是之前创建的特征匹配器对象，用于匹配特征描述子之间的相似性。
  //match(descriptors_1, descriptors_2, match): 这是匹配器的成员函数，用于匹配两组特征描述子 descriptors_1 和 descriptors_2。匹配的结果将被存储在 match 向量中。

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;



  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    //match[i].distance 表示第 i 对匹配点的距离，距离越小表示匹配越好
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      //
      //max(2 * min_dist, 30.0) 是一个阈值，用于判断是否保留匹配点。2 * min_dist 是最小距离的两倍，30.0 是一个经验值，作为距离的下限。
      matches.push_back(match[i]);
      //matches.push_back(match[i]);: 如果满足上述条件，将符合条件的匹配点添加到 matches 向量中。
    }
  }
}

// 将像素坐标转换为相机归一化坐标的函数实现
Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d(
    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    //像素坐标转换为相机归一化坐标的公式
  );
}

// 实现3D-3D位姿估计的函数(利用SVD方法)
void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
  // 计算两组点云的中心质心                          
  Point3f p1, p2;     // center of mass
  int N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = Point3f(Vec3f(p1) / N);
  p2 = Point3f(Vec3f(p2) / N);
  // 将两组点云中心移到原点
  vector<Point3f> q1(N), q2(N); // remove the center
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  // compute q1*q2^T
  // 计算 q1 * q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }
  cout << "W=" << W << endl;

  // SVD on W
  // 对 W 进行奇异值分解
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  cout << "U=" << U << endl;
  cout << "V=" << V << endl;

  // 计算旋转矩阵 R 和平移矢量 t
  Eigen::Matrix3d R_ = U * (V.transpose());
  if (R_.determinant() < 0) {
    R_ = -R_;
  }
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

  // convert to cv::Mat
  // 将结果转换为 OpenCV 的 Mat 类型
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

// 进行图优化的捆绑调整的函数实现
void bundleAdjustment(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t) {
  // 构建图优化，先设定g2o
  typedef g2o::BlockSolverX BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
    std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // vertex
  VertexPose *pose = new VertexPose(); // camera pose
  pose->setId(0);
  pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose);

  // edges
  for (size_t i = 0; i < pts1.size(); i++) {
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
      Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
    edge->setVertex(0, pose);
    edge->setMeasurement(Eigen::Vector3d(
      pts1[i].x, pts1[i].y, pts1[i].z));
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

  cout << endl << "after optimization:" << endl;
  cout << "T=\n" << pose->estimate().matrix() << endl;

  // convert to cv::Mat
  Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
  Eigen::Vector3d t_ = pose->estimate().translation();
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}
