#include <iostream>


/*


基本定义:Matrix<数据类型, 行数, 列数>

Eigen 通过 typedef 提供了许多内置类型
typedef Matrix<double,Dynamic,Dynamic> MatrixXd;
typedef Matrix<double, 3, 1> Vector3d;
typedef Matrix<double, 3, 3> Matrix3d;

矩阵的行列数可以是固定的也可以是动态的，能够确定大小的矩阵事先指定行列数处理起来会更快一些。

Eigen矩阵不支持自动类型转化，对两个不同类型的矩阵进行操作时，须手动转化
Matrix<double,2,1>result=matrix_23.cast<double>()*v_3d;

矩阵赋值
用“<<”操作符给矩阵赋值，从上到下从左到右
m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;


矩阵运算
常见的四则运算等直接通过符号实现即可

转置：mymatrix.transpose()
迹：mymatrix.trace()
各元素和：mymatrix.sum()
行列式：mymatrix.determinant()
逆：mymatrix.inverse()
特征值：mymatrix.eigenvalues()
特征向量：mymatrix.eigenvectors()

*/

using namespace std;

#include <ctime>
// Eigen 核心部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等
#include <Eigen/Dense>
/*
此处错误，eigen 库默认安装在了 /usr/include/eigen3/Eigen 路径下，需使用下面命令映射到 /usr/include 路径下。
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
*/

using namespace Eigen;

#define MATRIX_SIZE 50

/****************************
* 本程序演示了 Eigen 基本类型的使用
****************************/

int main(int argc, char **argv) {
  // Eigen 中所有向量和矩阵都是Eigen::Matrix，它是一个模板类。它的前三个参数为：数据类型，行，列
  // 声明一个2*3的float矩阵
  Matrix<float, 2, 3> matrix_23;
  // 同时，Eigen 通过 typedef 提供了许多内置类型，不过底层仍是Eigen::Matrix
  // 例如 Vector3d 实质上是 Eigen::Matrix<double, 3, 1>，即三维向量
  Vector3d v_3d;
  // 这是一样的      Vector 向量 Matrix 矩阵
  Matrix<float, 3, 1> vd_3d;

  // Matrix3d 实质上是 Eigen::Matrix<double, 3, 3>,即三维矩阵
  Matrix3d matrix_33 = Matrix3d::Zero(); //初始化为零
  // 如果不确定矩阵大小，可以使用动态大小的矩阵
  Matrix<double, Dynamic, Dynamic> matrix_dynamic;
  // 更简单的
  MatrixXd matrix_x;
  // 这种类型还有很多，我们不一一列举

  // 下面是对Eigen阵的操作
  // 输入数据（初始化）
  matrix_23 << 1, 2, 3, 4, 5, 6;
  // 输出
  cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;

  // 用()访问矩阵中的元素  Matrix（i，j）,索引从0开始
  cout << "print matrix 2x3: " << endl;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
    cout << endl;
  }

  // 矩阵和向量相乘（实际上仍是矩阵和矩阵）
  v_3d << 3, 2, 1;
  vd_3d << 4, 5, 6;

  // 但是在Eigen里你不能混合两种不同类型的矩阵，像这样是错的
  /*
     Matrix<float, 2, 3> matrix_23;matrix_23是float类型
     Eigen矩阵不支持自动类型转化，对两个不同类型的矩阵进行操作时，须手动转化
     Matrix<double,2,1>result=matrix_23.cast<double>()*v_3d;
  */
  // Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
  // 应该显式转换
  Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
  cout << "[1,2,3;4,5,6]*[3,2,1]=\n" << result << endl;

  Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
  cout << "[1,2,3;4,5,6]*[4,5,6]:\n " << result2.transpose() << endl; //.transpose() 转置

  // 同样你不能搞错矩阵的维度
  // 试着取消下面的注释，看看Eigen会报什么错
  // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

  // 一些矩阵运算
  // 四则运算就不演示了，直接用+-*/即可。
  matrix_33 = Matrix3d::Random();      // 随机数矩阵
  cout << "random matrix: \n" << matrix_33 << endl;
  cout << "transpose: \n" << matrix_33.transpose() << endl; // 转置
  cout << "sum: " << matrix_33.sum() << endl;               // 各元素和
  cout << "trace: " << matrix_33.trace() << endl;           // 迹
  cout << "times 10: \n" << 10 * matrix_33 << endl;         // 数乘
  cout << "inverse: \n" << matrix_33.inverse() << endl;     // 逆
  cout << "det: " << matrix_33.determinant() << endl;       // 行列式

  // 特征值
  // 实对称矩阵可以保证对角化成功
  SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
  cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
  cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

  // 解方程
  // 我们求解 matrix_NN * x = v_Nd 这个方程
  // N的大小在前边的宏里定义，它由随机数生成
  // 直接求逆自然是最直接的，但是求逆运算量大

  Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
      = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
  matrix_NN = matrix_NN * matrix_NN.transpose();  // 保证半正定,
  Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

  clock_t time_stt = clock(); // 计时
  // 直接求逆
  Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
  cout << "time of normal inverse is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  // 通常用矩阵分解来求，例如QR分解，速度会快很多
  time_stt = clock();
  x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout << "time of Qr decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  // 对于正定矩阵，还可以用cholesky分解来解方程
  time_stt = clock();
  x = matrix_NN.ldlt().solve(v_Nd);
  cout << "time of ldlt decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
  cout << "x = " << x.transpose() << endl;

  return 0;
}