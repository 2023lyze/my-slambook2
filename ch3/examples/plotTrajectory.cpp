#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>

// 本例演示了如何画出一个预先存储的轨迹

using namespace std;
using namespace Eigen;

//trajectory文件的路径（可以使用相对路径，也可以使用绝对路径）
string trajectory_file = "../trajectory.txt";  

//画轨迹函数(自定义)的声明
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

/*
vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>：是函数的参数，它是一个包含 Isometry3d 类型元素的向量。
Isometry3d 表示一个3D变换矩阵。Eigen::aligned_allocator<Isometry3d> 是为了确保 Isometry3d 类型的对齐分配器。
*/

int main(int argc, char **argv) {

    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;//容器的方式存储
    //因为类型是Eigen的Isometry3d和cpp的类内存分配不一样，所以要指定内存的分配方式
    //即:Eigen::aligned_allocator< Isometry3d > 
     /*
    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>：这是一个 C++ 的标准库容器 std::vector，它是一个动态数组，用于存储 Isometry3d 类型的元素。
    Eigen::aligned_allocator<Isometry3d> 是一个用于分配内存的分配器，它确保分配的内存是按照 Eigen 库的要求进行对齐的。
    poses：这是一个向量的变量名，即用于存储 Isometry3d 类型元素的向量。
    因此，这行代码创建了一个用于存储 Isometry3d 类型元素的动态数组，即存储了一系列3D变换矩阵的位姿信息。
    */
    
    ifstream fin(trajectory_file); //读文件
    //未成功读取

    if (!fin) {  
        cout << "cannot find trajectory file at " << trajectory_file << endl;
        return 1;
    }
    /*ifstream fin(trajectory_file);：
    创建一个输入文件流 fin，用于从轨迹文件中读取数据。trajectory_file 是轨迹文件的路径。
    if (!fin) { cout << "cannot find trajectory file at " << trajectory_file << endl; return 1; }：
    检查文件流是否成功打开。如果文件流没有成功打开，输出错误消息并返回 1，表示程序异常终止。*/

    // 从文件中读取位姿数据
    /*eof() 是 C++ 中流（如文件流）的成员函数，它用于检测文件流的结束标志。
    具体来说，eof() 返回 true，当且仅当读取文件流时已经达到文件末尾时，即在尝试读取下一个字符之前检测到文件结束符。
    while (!fin.eof()) 的循环条件意味着只要文件流 fin 还未到达文件末尾，就会继续执行循环。*/
    //一行一行读取，直到文件尾标志（efo标志）
    while (!fin.eof()) {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        
        Isometry3d Twr(Quaterniond(qw, qx, qy, qz)); //四元数转换为旋转矩阵
        Twr.pretranslate(Vector3d(tx, ty, tz));//旋转矩阵加上平移变为转换矩阵
        poses.push_back(Twr);//添加到容器中
    }
    cout << "read total " << poses.size() << " pose entries" << endl;

    // draw trajectory in pangolin    绘制轨迹
    DrawTrajectory(poses);
    return 0;
}

/*******************************************************************************************/
// 函数定义：绘制轨迹
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
    // 创建Pangolin窗口并绑定，使用Pangolin库创建一个窗口，启用OpenGL深度测试，并启用混合功能。
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);//新建窗口，参数分别为：窗口名称、窗口的长和宽
    glEnable(GL_DEPTH_TEST);//启用深度渲染，当需要显示3D模型时需要打开，根据目标的远近自动隐藏被遮挡的模型
    glEnable(GL_BLEND);//表示窗口使用颜色混合模式，让物体显示半透明效
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	/***
	设置颜色RGBA混合因子,前面参数表示源因子，后面参数表示目标因子
	GL_ZERO：表示使用0.0作为权重因子
	GL_ONE ： 表示使用1.0作为权重因子
	GL_SRC_ALPHA ：表示使用源颜色的alpha值来作为权重因子
	GL_DST_ALPHA ：表示使用目标颜色的alpha值来作为权重因子
	GL_ONE_MINUS_SRC_ALPHA： 表示用1.0-源颜色的alpha的值来作为权重因子
	GL_ONE_MINUS_DST_ALPHA： 表示用1.0-目标颜色的alpha的值来作为权重因子
	GL_SRC_COLOR>：表示用源颜色的四个分量作为权重因子
	在画图的过程中如果程序glClearColor()；glColor3f()则后者为源颜色，前者的颜色为目标颜色以上的GL_SRC_ALPHA和GL_ONE_MINUS_SRC_ALPHA是最常用的混合模式 ***/

	//创建相机视图，设置OpenGL渲染状态
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

	//ProjectionMatrix()设置相机内参,参数分别表示相机分辨率(2)，焦距(1)，相机光心(3)，最远最大距离(2)
	//ModelViewLookAt()设置观看视角,上文对应的意思是在世界坐标(0，-0.1，-1.8)处观看坐标原点（0,0,0）并设置Y轴向上
	/**  另一种解释
	 定义观测方位向量：观测点位置：(mViewpointX mViewpointY mViewpointZ)
                     观测目标位置：(0, 0, 0)
		             观测的方位向量：(0.0,-1.0, 0.0)**/
    
    // 创建Pangolin视图
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
	/**创建交互视图view,用于显示上一步摄像机所“拍摄”到的内容，,setBounds()函数前四个参数依次表示视图在视窗中的范围（下、上、左、右）,显示界面长宽比*/

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清空颜色和深度缓存
        d_cam.Activate(s_cam);//激活之前设定好的视窗对象，// 激活相机视图
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//为颜色缓存区指定确定的值
        glLineWidth(2); // 设置线宽为2
        for (size_t i = 0; i < poses.size(); i++) {
            // 画每个位姿的三个坐标轴
            Vector3d Ow = poses[i].translation();//无参数，返回当前变换平移部分的向量表示(可修改)，可以索引[]获取各分量
            //对三个坐标轴分别旋转
            Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));//0.1是为了让图看起来舒服，不会太大
            Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
            Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
            glBegin(GL_LINES);//开始画线
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();//结束画线
        }
        // 画出连线
        for (size_t i = 0; i < poses.size(); i++) {
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        // 完成当前帧的绘制
        pangolin::FinishFrame();
        // 等待一段时间，以控制帧率
        usleep(5000);   // sleep 5 ms
    }
        /*
    创建 Pangolin 视图，并设置视图边界和处理器。
    在 Pangolin 窗口中循环渲染，每一帧执行以下操作：
    清空颜色缓冲区和深度缓冲区。
    激活相机视图。
    设置背景颜色为白色，线宽为2。
    绘制每个位姿的坐标轴（X、Y、Z轴）。
    绘制相邻位姿之间的连线。
    完成当前帧的绘制并等待一段时间以控制帧率。
    */
}
