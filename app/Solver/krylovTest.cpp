#include "Solver/Linear.hpp"

namespace DNDS
{
    struct WrappedVXD : public Eigen::VectorXd
    {
        using Eigen::VectorXd::Matrix;
        void addTo(WrappedVXD &R, double r)
        {
            this->operator+=(r * R);
        }
    };
    void testPCG()
    {
        int n = 10;
        using tPCG = Linear::PCG_PreconditionedRes<WrappedVXD, real>;
        auto pcg = tPCG([n](WrappedVXD &v)
                        { v.resize(n); });
        Eigen::MatrixXd A;
        A.resize(n, n);
        A.setRandom();
        A = A * A.transpose();
        WrappedVXD b, x;
        b.setOnes(n);
        x.setZero(n);
        Eigen::MatrixXd M = A.diagonal().asDiagonal();
        M.setIdentity();
        Eigen::MatrixXd MIA = M.inverse() * A;

        std::cout << "A" << std::endl;
        std::cout << A << std::endl;
        std::cout << "M" << std::endl;
        std::cout << M << std::endl;

        pcg.solve(
            [&](WrappedVXD &x, WrappedVXD &Ax)
            {
                Ax = MIA * x;
                Ax = M * Ax;
            },
            [&](WrappedVXD &x, WrappedVXD &Mx)
            {
                Mx = M * x;
            },
            [&](WrappedVXD &x, WrappedVXD &res)
            {
                res = M.inverse() * (b - A * x);
            },
            [&](WrappedVXD &a, WrappedVXD &b)
            {
                return a.dot(b);
            },
            x, 5,
            [&](int iiter, real res, real resZ)
            {
                std::cout << "PCG: " << iiter << ", res = " << res << ", ";
                std::cout << 0.5 * (A * x).dot(x) - b.dot(x) << std::endl;
                return false;
            });
        std::cout << (A * x - b).transpose() << std::endl;
    }
}

int main()
{
    DNDS::testPCG();
    return 0;
}