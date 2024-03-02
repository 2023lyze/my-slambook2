#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main( int argc, char** argv ) {
    // read the image  读取图像
    cout<<"reading images... "<<endl;
    vector<Mat> images; 
    for ( int i=0; i<10; i++ )
    {
        string path = "../data/"+to_string(i+1)+".png";
        // to_string() 是C++标准库中的一个函数，它用于将不同类型的数值转换为对应的字符串表示。
        images.push_back( imread(path) );
    }
    // detect ORB features 检测ORB特征
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    /*
    Ptr 是OpenCV库中的智能指针类，它提供了一个通用的指针接口，用于管理不同类型的指针。在这里，Ptr< Feature2D > 表示一个指向 Feature2D 类型的智能指针。
    Feature2D 是一个抽象类，代表图像特征的基类。它定义了一些接口，其中包括特征检测和描述的方法。具体的特征算法（如ORB、SIFT、SURF等）都是 Feature2D 的派生类。
    ORB::create() 是一个静态方法，用于创建 ORB 特征检测器的实例。create() 方法返回一个指向 Feature2D 类型的指针，因此可以通过 Ptr< Feature2D > 类型的智能指针来管理和使用它。
    */
    vector<Mat> descriptors;// 用于存储每张图像的特征描述子的向量
    // 对于 images 中的每一张图像执行以下操作
    for ( Mat& image:images ) // 范围循环
    {
        vector<KeyPoint> keypoints; // 存储图像中检测到的特征点的向量
        Mat descriptor;// 存储计算得到的特征描述子的矩阵
        // 使用之前创建的特征检测器 detector 来检测特征点并计算描述子
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        // 将计算得到的描述子添加到 descriptors 向量中
        descriptors.push_back( descriptor );
    }
    /*
    在这个循环中，每一次迭代都处理 images 向量中的一张图像。首先，为当前图像创建一个存储特征点的向量 keypoints 和一个存储特征描述子的矩阵 descriptor。
    然后，使用之前创建的 detector 对当前图像执行特征检测和描述子计算。最后，将计算得到的描述子添加到 descriptors 向量中，以便后续用于创建视觉词汇。
    */
    // create vocabulary  创建视觉词汇 
    cout<<"creating vocabulary ... "<<endl;
    //通过 DBoW3::Vocabulary 类创建了一个视觉词汇对象 vocab。
    DBoW3::Vocabulary vocab;
    // 使用先前计算得到的描述子来创建视觉词汇 使用先前计算得到的图像描述子调用 vocab.create(descriptors) 方法，该方法根据描述子创建了视觉词汇。
    vocab.create( descriptors );
    // 输出视觉词汇的信息
    cout<<"vocabulary info: "<<vocab<<endl;
    // 保存视觉词汇为 "vocabulary.yml.gz" 文件
    vocab.save( "vocabulary.yml.gz" );
    cout<<"done"<<endl;
    
    return 0;
}