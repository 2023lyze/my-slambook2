#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  // 定义两个四元数
  Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);

  // 对四元数进行归一化
  q1.normalize();
  q2.normalize();

  // 定义两个平移向量
  Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);

  // 定义一个三维点
  Vector3d p1(0.5, 0, 0.2);

  // 定义两个变换矩阵 T1w 和 T2w
  Isometry3d T1w(q1), T2w(q2);

  // 分别设置平移部分
  T1w.pretranslate(t1);
  T2w.pretranslate(t2);

  // 进行坐标变换：p2 = T2w * T1w.inverse() * p1
  Vector3d p2 = T2w * T1w.inverse() * p1;

  // 输出结果
  cout << endl << p2.transpose() << endl;

  return 0;
}
