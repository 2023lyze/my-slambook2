#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace Sophus;
using namespace std;

string groundtruth_file = "./example/groundtruth.txt";
string estimated_file = "./example/estimated.txt";
/*
这段代码用于比较地面真实轨迹和估计轨迹的均方根误差（RMSE），并在Pangolin窗口中绘制两条轨迹。
函数 ReadTrajectory 从文件中读取轨迹数据，函数 DrawTrajectory 则使用Pangolin库绘制轨迹。
*/

// 使用Sophus库定义SE3类型的轨迹
//定义了一个新的数据类型 TrajectoryType，它是一个容器，用于存储 Sophus::SE3d 类型的对象。
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
/*
TrajectoryType：这是一个用户定义的数据类型名称，类似于取了个别名，方便在代码中使用。通过这个别名，我们可以更方便地声明和使用这种特定类型的容器。
vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>：这是实际的数据类型，它是一个使用 vector 容器模板的数据类型，其中存储的元素是 Sophus::SE3d 类型的对象。
Eigen::aligned_allocator<Sophus::SE3d>：这是一个特殊的内存分配器，确保 Sophus::SE3d 类型的对象在内存中按照 Eigen 库的对齐要求进行存储。Eigen 是一个用于线性代数操作的 C++ 模板库，而 Sophus 本身也依赖于 Eigen。
因此，TrajectoryType 是一个向量，其中的每个元素都是 Sophus::SE3d 类型的对象，而且在内存中按照 Eigen 库的对齐要求进行存储。这种定义方式简化了代码，并提高了数据存储的效率。
*/

// 函数声明，用于绘制两个轨迹
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

// 函数声明，用于从文件中读取轨迹数据
TrajectoryType ReadTrajectory(const string &path);

int main(int argc, char **argv) {
  // 从文件中读取地面真实轨迹和估计轨迹
  TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
  TrajectoryType estimated = ReadTrajectory(estimated_file);
  // 确保轨迹不为空，且两个轨迹点数相同
  assert(!groundtruth.empty() && !estimated.empty());
  assert(groundtruth.size() == estimated.size());
  //assert 是一个在程序中用于调试的宏，它在运行时检查一个条件是否为真。如果条件为假，assert 将打印一条错误消息并终止程序的执行。

  // 计算并输出RMSE（均方根误差）
  double rmse = 0;
  //遍历每一帧的位姿：
  for (size_t i = 0; i < estimated.size(); i++) {
    //获取当前帧的地面真实位姿和估计位姿：
    //这是一个循环，对于每一帧（i 表示帧的索引），从估计轨迹中取出一个位姿 p1。
    Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
    //计算两个位姿之间的误差：
    //p2.inverse()：表示地面真实位姿的逆。
    //p2.inverse() * p1：表示估计位姿相对于地面真实位姿的相对变换。
    //.log()：表示将相对变换转换为李代数（Lie algebra）形式。
    //.norm()：表示求李代数的范数，即误差的大小。
    double error = (p2.inverse() * p1).log().norm();
    //累加误差的平方：
    rmse += error * error;
  }
  //计算均方根误差：
  rmse = rmse / double(estimated.size());
  rmse = sqrt(rmse);
  /*
  rmse / double(estimated.size())：表示将累加的误差平方和除以估计轨迹的总帧数，得到平均误差的平方。
  sqrt(rmse)：表示取平方根，得到均方根误差。
  */
  cout << "RMSE = " << rmse << endl;
  cout << "红色表示估计轨迹"<< endl;
  cout << "蓝色表示真实轨迹"<< endl;

  // 绘制轨迹
  DrawTrajectory(groundtruth, estimated);
  return 0;
}

// 从文件中读取轨迹数据
TrajectoryType ReadTrajectory(const string &path) {
  ifstream fin(path);
  TrajectoryType trajectory;
  if (!fin) {
    cerr << "trajectory " << path << " not found." << endl;
    return trajectory;
  }

  // 从文件中读取每一行的时间戳、位移和四元数，并构造SE3对象，将其加入轨迹

  //fin.eof() 是 ifstream 类提供的一个成员函数，用于检查文件是否已经到达末尾。如果已经到达末尾，eof() 返回 true；否则返回 false。 
  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    /*
    fin 是一个 ifstream 类型的文件输入流，用于从文件中读取数据。
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw; 是一个链式的输入运算符的使用。它从文件中一次性读取一行数据，并将这一行中的各个值按顺序赋给对应的变量。
  
    time、tx、ty、tz、qx、qy、qz、qw 是 double 类型的变量，它们分别表示时间戳、位移和四元数的各个分量。
    >> 运算符用于将输入流的内容提取到相应的变量中，它会根据!!!空格或换行符!!!等分隔符将输入的数据分!!!隔开!!!。

    */
    Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
    /*
    push_back 是 C++ 中用于在容器尾部添加新元素的方法。
    这行代码的作用是将 p1 追加到 trajectory 向量的末尾。
    在这里，trajectory 是用于存储 Sophus::SE3d 对象的轨迹（一系列位姿变换）的容器。
    通过 push_back 操作，新的位姿 p1 被添加到轨迹的末尾。
    */
    trajectory.push_back(p1);
  }
  return trajectory;
}

// 绘制两个轨迹
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // 创建Pangolin窗口和绑定
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // 设置OpenGL渲染状态
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  // 创建Pangolin视图并设置显示区域和事件处理器
  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));

  // 主循环，绘制轨迹
  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // 绘制地面真实轨迹
    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // 蓝色表示地面真实轨迹
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    // 绘制估计轨迹
    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // 红色表示估计轨迹
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    
    pangolin::FinishFrame();
    usleep(5000);   // 休眠5毫秒
  }
}
