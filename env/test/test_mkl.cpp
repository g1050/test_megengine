#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <time.h>

using namespace std;
using namespace Eigen;

// 使用Eigen+Intel MKL
int main(int argc, char *argv[])
{
    MatrixXd a = MatrixXd::Random(1000, 1000);  // 随机初始化矩阵
    MatrixXd b = MatrixXd::Random(1000, 1000);

    double start = clock();
    MatrixXd c = a * b;    // 乘法好简洁
    double endd = clock();
    double thisTime = (double)(endd - start) / CLOCKS_PER_SEC;

    cout << thisTime << endl;

    //system("PAUSE");
    return 0;
}
