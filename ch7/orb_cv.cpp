#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

// 使用OpenCV 4.4版本，CV_LOAD_IMAGE_COLOR已经不再使用，但可以在constants_c.h头文件中找到对应的声明
#include "opencv2/imgcodecs/legacy/constants_c.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) 
{
  // 检查命令行参数是否满足要求
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }

  //-- 读取图像
  Mat img_1 = imread(argv[1], IMREAD_COLOR); // 使用IMREAD_COLOR替代CV_LOAD_IMAGE_COLOR
  Mat img_2 = imread(argv[2], IMREAD_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr); // 确保图像读取成功
/*
在OpenCV中，imread 函数用于读取图像，它的常用参数包括：
IMREAD_COLOR: 以彩色模式读取图像，忽略透明度信息。
IMREAD_GRAYSCALE: 以灰度模式读取图像。
IMREAD_UNCHANGED: 以包含透明度信息的方式读取图像，保留透明度通道。
*/

//-- 初始化
  std::vector<KeyPoint> keypoints_1, keypoints_2; //定义了两个vector容器keypoints_1和 keypoints_2，存放的对象则是KeyPoint类型
  Mat descriptors_1, descriptors_2;
 
// “Ptr<FeatureDetector> detector = ”等价于 “FeatureDetector * detector =”
//Ptr是OpenCV中使用的智能指针模板类
 //ORB::create() 是 ORB 算法的静态成员函数，用于创建 ORB 特征检测器或描述子提取器的实例。这个函数返回一个指向 Feature2D 类型对象的指针。
  Ptr<FeatureDetector> detector = ORB::create();//特征检测器FeatureDetetor，通过定义FeatureDetector的对象可以使用多种特征检测及匹配方法，通过create()函数调用。
  Ptr<DescriptorExtractor> descriptor = ORB::create();//描述子提取器DescriptorExtractor是提取关键点的描述向量类抽象基类。
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");//描述子匹配器DescriptorMatcher用于特征匹配，"Brute-force-Hamming"表示使用汉明距离进行匹配。


  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时 t1时刻
  //C++11 中的 <chrono> 头文件，创建了一个 steady_clock 类型的时间点 t1，记录了当前时间。这是为了测量特征检测过程的执行时间。
  detector->detect(img_1, keypoints_1);// detector 指针指向的 ORB 特征检测器的 detect 方法，该方法用于检测图像 img_1 中的特征点（这里是 Oriented FAST 角点），并将检测到的特征点存储在 keypoints_1 中。
  detector->detect(img_2, keypoints_2);

  
//第二步，根据角点计算BREIF描述子
   descriptor->compute(img_1, keypoints_1, descriptors_1);//computer()计算关键点的描述子向量
   descriptor->compute(img_2, keypoints_2, descriptors_2);
 //这一行代码使用 descriptor 指针，调用了 compute 方法。该方法用于计算图像 img_1 中由 keypoints_1 中的角点位置确定的 BRIEF 描述子，并将结果存储在 descriptors_1 中
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now(); //计时 t2时刻
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//(t2-t1)的时间，其实就是在计算 提取ORB及计算BRIEF的时间
   cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;
   
   Mat outimg1;// 用于存储绘制关键点后的图像
   drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);//将ORB圈出来
   imshow("ORB features", outimg1); 
  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;    //DMatch是匹配关键点描述子 类, matches用于存放匹配项
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches); //对参数1 2的描述子进行匹配，并将匹配项存放于matches中
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match the ORB cost: " << time_used.count() << "seconds. " << endl;

//第四步，匹配点对筛选
  //------------------------------------------------------------
  //方法一：“查询描述子集合”和“训练描述子集合”
   //计算最小距离和最大距离
   auto min_max = minmax_element(matches.begin(), matches.end(),
       [](const DMatch &m1, const DMatch &m2){ return m1.distance < m2.distance; });
 
   // auto 可以在声明变量的时候根据变量初始值的类型自动为此变量选择匹配的类型
   /*
   minmax_element 是 C++ 标准库 <algorithm> 头文件中的一个函数，用于找到范围内元素的最小和最大值，并返回包含这两个元素的迭代器对。
   minmax_element()返回指向范围内最小和最大元素的一对迭代器。参数1 2为起止迭代器范围
   */ 
  
   double min_dist = min_max.first->distance;  // min_max存储了一堆迭代器，first指向最小元素
   double max_dist = min_max.second->distance; // second指向最大元素


//当描述子之间的距离大于两倍最小距离时，就认为匹配有误。但有时最小距离会非常小，所以要设置一个经验值30作为下限。
   vector<DMatch> good_matches;  //存放良好的匹配项
 
 //当描述自之间的距离大于两倍的min_dist，即认为匹配有误，舍弃掉。
 //但是有时最小距离非常小，比如趋近于0了，所以这样就会导致min_dist到2*min_dist之间没有几个匹配。
 // 所以，在2*min_dist小于30的时候，就取30当上限值，小于30即可，不用2*min_dist这个值了
   for(int i = 0; i < descriptors_1.rows; ++i)
   {
       if(matches[i].distance <= max(2 * min_dist, 30.0))//max(2 * min_dist, 30.0)这是一个阈值，取最小距离的两倍和30之中的较大值
       {
           good_matches.push_back(matches[i]);
       }
   }
  //------------------------------------------------------------
  //方法二：暴力匹配法 BFMatcher
  //开启交叉检测
  BFMatcher matcher_BF(NORM_HAMMING,1);
  vector<DMatch> matches_BF;
  vector<DMatch> good_matches_BF;
  matcher_BF.match(descriptors_1, descriptors_2, matches_BF);
  //方法二优化：
  auto min_max_BF = minmax_element(matches_BF.begin(), matches_BF.end(),
                                    [](const DMatch &m1_BF, const DMatch &m2_BF) { return m1_BF.distance < m2_BF.distance; });
  double min_dist_BF = min_max_BF.first->distance;
  double max_dist_BF = min_max_BF.second->distance;
  
  for(int i=0;i<matches_BF.size();i++)
  {
      if(matches_BF[i].distance <= max(2 * min_dist_BF, 30.0))
      {
          good_matches_BF.push_back(matches_BF[i]);
      }
  }
  //------------------------------------------------------------
  //方法三：FLANN
  //效果出奇的差，估计哪里错了
  FlannBasedMatcher matcher_FLANN;
  vector<DMatch> matches_FLANN;
  vector<DMatch> good_matches_FLANN;
  //注意数据类型转换
  descriptors_1.convertTo(descriptors_1,CV_32F);
  descriptors_2.convertTo(descriptors_2,CV_32F);
  matcher_FLANN.match(descriptors_1, descriptors_2, matches_FLANN);
  //方案三优化：
  auto min_max_FLANN = minmax_element(matches_FLANN.begin(), matches_FLANN.end(),
                                    [](const DMatch &m1_FLANN, const DMatch &m2_FLANN) { return m1_FLANN.distance < m2_FLANN.distance; });
  double min_dist_FLANN = min_max_FLANN.first->distance;
  double max_dist_FLANN = min_max_FLANN.second->distance;
  for(int i=0;i<matches_FLANN.size();i++)
  {
      if(matches_FLANN[i].distance <= 0.35*max_dist_FLANN)
      {
          good_matches_FLANN.push_back(matches_FLANN[i]);
      }
  }

  //-- 第五步:绘制匹配结果
Mat img_match;//所有匹配点图
Mat img_goodmatch;//筛选后的匹配点图
Mat img_match_BF, img_goodmatch_BF;
Mat img_match_FLANN, img_goodmatch_FLANN;
//drawMatches 是 OpenCV 中用于在两个图像之间绘制特征匹配的函数。
drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);//drawMatches 是 OpenCV 中用于在两个图像之间绘制特征匹配的函数。
drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches_BF, img_match_BF);
drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches_BF, img_goodmatch_BF);
drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches_FLANN, img_match_FLANN);
drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches_FLANN, img_goodmatch_FLANN);
imshow("all matches", img_match);
imshow("good matches", img_goodmatch);
imshow("matches_BF",img_match_BF);
imshow("good_matches_BF",img_goodmatch_BF);
imshow("matches_FLANN",img_match_FLANN);
imshow("good_matches_FLANN",img_goodmatch_FLANN);
waitKey(0);
  
return 0;
}
