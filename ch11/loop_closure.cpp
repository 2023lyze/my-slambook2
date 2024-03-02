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
 * 本节演示了如何根据前面训练的字典计算相似性评分
 * ************************************************/
int main(int argc, char **argv) {
    // read the images and database  读取图像和数据库
    cout << "reading database" << endl;
    DBoW3::Vocabulary vocab("../vocabulary.yml.gz");
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // use large vocab if you want:  如果需要，可以使用更大的字典
    if (vocab.empty()) {
        cerr << "Vocabulary does not exist." << endl;
        return 1;
    }
    cout << "reading images... " << endl;
    vector<Mat> images;
    for (int i = 0; i < 10; i++) {
        string path = "../data/" + to_string(i + 1) + ".png";
        images.push_back(imread(path));
    }

    // NOTE: in this case we are comparing images with a vocabulary generated by themselves, this may lead to overfit.
    // 注意：在这种情况下，我们正在比较由它们自身生成的词汇表的图像，这可能导致过拟合
    // detect ORB features
    // 检测ORB特征
    cout << "detecting ORB features ... " << endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for (Mat &image:images) {
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // we can compare the images directly or we can compare one image to a database 
    // 我们可以直接比较图像，或者将一个图像与数据库比较
    // images :
    // 图像之间的比较
    cout << "comparing images with images " << endl;
    // 遍历图像 
    for (int i = 0; i < images.size(); i++) {
        // 创建第一个图像的词袋向量
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        /*这一行代码使用视觉词汇 vocab 将图像的特征描述子 descriptors[i] 转换为词袋向量 v1。
        vocab: 是之前训练得到的视觉词汇，通过 DBoW3::Vocabulary 类加载。
        descriptors[i]: 是第 i 张图像的特征描述子，这是通过 ORB 特征检测器计算得到的。
        v1: 是用于存储转换后的词袋向量的变量，类型为 DBoW3::BowVector。*/
        // 遍历与第一个图像相同或之后的图像
        for (int j = i; j < images.size(); j++) {
            // 创建第二个图像的词袋向量
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            // 计算第一个图像与第二个图像之间的相似性评分
            double score = vocab.score(v1, v2);
            // 输出相似性评分
            cout << "image " << i << " vs image " << j << " : " << score << endl;
        }
        cout << endl; // 换行，用于分隔不同图像的输出
    }

    // or with database 
    // 或者与数据库比较
    cout << "comparing images with database " << endl;
    // 创建一个 DBoW3::Database 对象，该对象将用于存储图像的词袋向量
    DBoW3::Database db(vocab, false, 0);

    for (int i = 0; i < descriptors.size(); i++)
        // 将每个图像的特征描述子添加到数据库
        db.add(descriptors[i]);

    // 输出数据库的信息
    cout << "database info: " << db << endl;

    // 使用数据库进行查询，查找与每个图像相似的图像
    for (int i = 0; i < descriptors.size(); i++) {
        DBoW3::QueryResults ret;
        /*创建了一个 DBoW3::QueryResults 类型的对象 ret，用于存储数据库查询的结果。
        DBoW3::QueryResults 是 DBoW3 库中用于存储查询结果的数据结构，它包含了每个查询结果的信息，包括相似性得分和对应的图像索引。
        */

        // 查询数据库，返回与当前图像相似性最高的前4个结果（max result=4）
        db.query(descriptors[i], ret, 4);      // max result=4

        // 输出查询结果
        cout << "searching for image " << i << " returns " << ret << endl << endl;
    }
    cout << "done." << endl;
}