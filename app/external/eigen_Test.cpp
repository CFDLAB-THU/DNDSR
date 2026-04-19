// #define EIGEN_NO_MALLOC
#include "Eigen/Dense"
#include "DNDS/Defines.hpp"
#include <unsupported/Eigen/CXX11/TensorSymmetry>

int main()
{
    using namespace DNDS;
    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> A;
    // A.setRandom(5,5);
    A.resize(5, 5);
    A << 1, 2, 3, 4, 5,
        1, 1, 3, 4, 6,
        1, 3, 3, 4, 6,
        1, 2, 5, 4, 6,
        1, 2, 3, 1, 6;
    std::cout << A << std::endl;
    std::cout << "Ainv" << std::endl;
    Eigen::MatrixXd AI = A.colPivHouseholderQr().inverse();
    std::cout << AI << std::endl;
    std::cout << AI * A << std::endl;
    std::cout << "Test Meta:\n";
    std::cout << std::is_trivial_v<Eigen::Matrix3d> << std::endl;
    std::cout << std::is_trivial_v<Eigen::Matrix3Xd> << std::endl;
    std::cout << std::is_trivial_v<Eigen::Matrix<real, -1, -1, Eigen::ColMajor, 3, 3>> << std::endl;
    std::cout << std::is_trivially_copyable_v<Eigen::Matrix3d> << std::endl;
    std::cout << std::is_pod_v<Eigen::Matrix3d> << std::endl;
    std::cout << std::is_trivially_copyable_v<Eigen::Matrix3Xd> << std::endl;
    std::cout << std::is_trivially_copyable_v<Eigen::Matrix<real, -1, -1, Eigen::ColMajor, 3, 3>> << std::endl;
    std::cout << std::is_trivially_copyable_v<std::array<real, 5>> << std::endl;
    std::cout << std::is_trivially_copyable_v<real[5][5]> << std::endl;
    std::cout << "Test Size:\n";
    std::cout << sizeof(Eigen::Matrix<double, 5, 5>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, 100, 100>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, 100, Eigen::Dynamic>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, 100, Eigen::Dynamic, Eigen::DontAlign, 100, 100>) << std::endl;
    std::cout << sizeof(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::DontAlign, 100, 100>) << std::endl;
    std::cout << sizeof(Eigen::Map<Eigen::Matrix<double, 5, 5>>) << std::endl;
    std::cout << sizeof(Eigen::Map<Eigen::Matrix<double, 100, 100>>) << std::endl;

    std::cout << "Test Array:\n";
    using T = real[3][4];
    std::vector<T> Tarray(10);
    std::cout << sizeof(T) << std::endl;
    std::cout << (Tarray.data() + 1) << std::endl;
    std::cout << (Tarray.data()) << std::endl;

    struct Mat33Wrapper : public Eigen::Matrix3d
    {
        using t_base = Eigen::Matrix3d;
        // using t_base::t_base;
        Mat33Wrapper() = default;
        Mat33Wrapper(const Mat33Wrapper &r) = default;
        Mat33Wrapper(Mat33Wrapper &&r) = default;

    } w33_a, w33_b;

    std::cout << "Test Wrap:\n";
    std::cout << std::is_trivially_copyable_v<Mat33Wrapper> << std::endl;

    Eigen::Matrix3d a;
    a.setRandom();

    std::cout << a.determinant() << std::endl;

    Eigen::Matrix3d b;
    b.setIdentity();
    std::cout << b.array().sign() << std::endl;
    for (int i = 0; i < b.size(); i++)
        std::cout << b(i) << std::endl;

    std::cout << fmt::format("r,c stride: {},{}", b.rowStride(), b.colStride()) << std::endl;

    auto data = std::vector<double>();
    for (auto i : {1, 2, 3, 4, 5, 6, 7, 8, 9})
        data.push_back(i);

    std::cout << std::endl;
    std::cout << "=== col major, stride o,i= 3, 1" << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double,
                                          3, 3, Eigen::DontAlign | Eigen::ColMajor>,
                            Eigen::Unaligned, Eigen::Stride<3, 1>>(data.data(), 3, 3)
              << std::endl;
    std::cout << "=== col major, stride o,i= 1, 3" << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double,
                                          3, 3, Eigen::DontAlign | Eigen::ColMajor>,
                            Eigen::Unaligned, Eigen::Stride<1, 3>>(data.data(), 3, 3)
              << std::endl;
    std::cout << "=== row major, stride o,i= 3, 1" << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double,
                                          3, 3, Eigen::DontAlign | Eigen::RowMajor>,
                            Eigen::Unaligned, Eigen::Stride<3, 1>>(data.data(), 3, 3)
              << std::endl;
    std::cout << "=== row major, stride o,i= 1, 3" << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double,
                                          3, 3, Eigen::DontAlign | Eigen::RowMajor>,
                            Eigen::Unaligned, Eigen::Stride<1, 3>>(data.data(), 3, 3)
              << std::endl;

    return 0;
}