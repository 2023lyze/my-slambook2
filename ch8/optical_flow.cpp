#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
// 文件路径
string file_1 = "./LK1.png";  // first image
string file_2 = "./LK2.png";  // second image

// Optical flow tracker and interface   光流跟踪器和接口
//定义光流跟踪器类 OpticalFlowTracker
class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        bool inverse_ = true, bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_) {}

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

/**
 * 单层光流跟踪
 * @param [in] img1 第一幅图像
 * @param [in] img2 第二幅图像
 * @param [in] kp1 第一幅图像中的关键点
 * @param [in|out] kp2 第二幅图像中的关键点，如果为空，则使用kp1中的初始猜测
 * @param [out] success 若关键点成功跟踪则为true
 * @param [in] inverse 是否使用反向形式？
 * @param [in] has_initial_guess 是否有初始猜测？
 */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_guess = false
);

/**
 * 多层光流跟踪，默认金字塔尺度为2
 * 在函数内部创建图像金字塔
 * @param [in] img1 第一幅图像金字塔
 * @param [in] img2 第二幅图像金字塔
 * @param [in] kp1 第一幅图像中的关键点
 * @param [out] kp2 第二幅图像中的关键点
 * @param [out] success 若关键点成功跟踪则为true
 * @param [in] inverse 设置为true以启用反向形式
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);

/**
 * 获取灰度像素值 GetPixelValue 函数：
 * 从参考图像中获取灰度值（双线性插值）
 * @param img 图像
 * @param x x坐标
 * @param y y坐标
 * @return 插值得到的像素值
 */

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}
 
int main(int argc, char **argv) 
{

    // 加载图像，注意它们的类型为CV_8UC1，而不是CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // 关键点，这里使用GFTT算法
    vector<KeyPoint> kp1; //声明了一个存储关键点的向量
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // 这个检测器的参数是最大关键点数（500）、质量水平（0.01），以及最小距离（20）。 //创建了一个 GFTT 特征检测器的指针 detector //
    detector->detect(img1, kp1); //使用 GFTT 检测器检测图像 img1 中的关键点，并将结果存储在 kp1 中。这个过程会根据 GFTT 算法寻找图像中最具有区分度的关键点，检测结果保存在 kp1 向量中。

    // now lets track these key points in the second image
    // first usev single level LK in the validation picture 单层LK光流在验证图片上进行跟踪
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // then test multi-level LK 多层LK测试
    vector<KeyPoint> kp2_multi;
    //声明一个空的关键点向量 kp2_multi，用于存储多层光流跟踪的结果。
    vector<bool> success_multi;
    //声明一个 bool 类型的向量 success_multi，用于存储每个关键点在多层光流跟踪中的跟踪状态。
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //记录多层光流跟踪开始的时间点。
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    /*
    调用自定义的多层光流跟踪函数 OpticalFlowMultiLevel，将图像 img1
    中的关键点 kp1 在图像 img2 中进行多层光流跟踪，结果存储在 kp2_multi 中，跟踪状态存储在 success_multi 中。
    */
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now(); // 记录多层光流跟踪结束的时间点。
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1); // 计算多层光流跟踪所用的时间。
    cout << "optical flow by gauss-newton: " << time_used.count() << endl; // 输出多层光流跟踪的时间。

    // use opencv's flow for validation  使用OpenCV的光流进行验证
    vector<Point2f> pt1, pt2;
    //声明两个空的 Point2f 向量 pt1 和 pt2，用于存储两个图像中的关键点。
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    //将光流跟踪之前的关键点位置 kp1 转换为 Point2f 类型，并存储在 pt1 中。
    vector<uchar> status;
    //声明一个 uchar 类型的向量 status，用于存储每个关键点的跟踪状态。
    vector<float> error;
    //声明一个 float 类型的向量 error，用于存储每个关键点的跟踪误差。
    t1 = chrono::steady_clock::now();//记录光流跟踪开始的时间点。
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    //调用 OpenCV 的金字塔 Lucas-Kanade 光流跟踪函数。这个函数使用图像金字塔进行光流跟踪，将初始关键
    //点位置 pt1 在图像 img1 中跟踪到图像 img2 中，结果存储在 pt2 中，跟踪状态存储在 status 中，跟踪误差存储在 error 中。
    t2 = chrono::steady_clock::now(); // 记录光流跟踪结束的时间点。
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1); // 计算光流跟踪所用的时间。
    cout << "optical flow by opencv: " << time_used.count() << endl; // 输出光流跟踪的时间。

    // plot the differences of those functions 绘制三种方法的差异
    Mat img2_single; //声明了一个新的图像 img2_single，用于绘制单层光流跟踪的结果。
    cv::cvtColor(img2, img2_single, COLOR_GRAY2BGR);
    //将灰度图 img2 转换为 BGR 彩色图，以便后续的绘图。
    for (int i = 0; i < kp2_single.size(); i++) {
        //遍历单层光流跟踪的关键点，其中 kp2_single 存储了跟踪后的关键点位置。
        if (success_single[i]) {
            //检查关键点是否成功跟踪。如果 success_single[i] 为真，表示该关键点跟踪成功。
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            //图像上绘制一个半径为 2 的绿色圆圈，表示跟踪后的关键点位置。
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
            //在图像上绘制一条从初始关键点位置 kp1[i].pt 到跟踪后的关键点位置 kp2_single[i].pt 的绿色线，表示光流的方向。
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    // 显示结果
    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}


// 光流跟踪的单层实现
//通过调用 OpticalFlowTracker 类的成员函数 calculateOpticalFlow 来执行光流跟踪的计算。
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_initial) 
    {
    kp2.resize(kp1.size());
    //整 kp2 向量的大小，使其与 kp1 向量大小相同。kp2 存储的是跟踪后的关键点位置。
    success.resize(kp1.size());
    //调整 success 向量的大小，使其与 kp1 向量大小相同。success 存储的是每个关键点的跟踪是否成功的信息。
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    //OpticalFlowTracker 类的实例 tracker，并将输入参数传递给它。OpticalFlowTracker 类负责实际的光流跟踪计算。
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
                  //并行地遍历关键点, 调用 OpticalFlowTracker 类的 calculateOpticalFlow 成员函数进行光流跟踪计算。parallel_for_ 是一个并行循环函数，它将计算任务分配到多个线程上，提高计算效率。
    }

// 光流跟踪器实现 OpticalFlowTracker 的成员函数实现：
// 光流跟踪器的计算光流 calculateOpticalFlow 函数实现
void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // 参数
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy需要被估计
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;// 指示该点是否成功

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // Hessian矩阵
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias 偏置
        Eigen::Vector2d J;  // Jacobian矩阵
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // 仅重置b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            /// 计算 cost 和 Jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                        // 在反向模式下，J在所有迭代中保持不变
                        // 注意这里的J在更新dx, dy时不会改变，因此我们可以存储它并仅计算error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // 计算H，b和设置cost
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // 同时更新H
                        H += J * J.transpose();
                    }
                }

            // 计算更新
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // 有时在黑色或白色块上会出现，且H不可逆时发生
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // 更新dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // 收敛
                break;
            }
        }

        success[i] = succ;

        // 设置kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}

//光流跟踪器的计算光流 calculateOpticalFlow 函数实现
// 多层光流跟踪的实现
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) {


    // 参数
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // 创建金字塔
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2; // 图像金字塔
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // 粗到细的LK跟踪在金字塔上进行
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // 从粗到细
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        if (level > 0) {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);
}
