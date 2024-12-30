#include "Solver/Linear.hpp"
#include <random>

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

    static int __seed = 1121;

    void testPCG()
    {
        int n = 10;
        using tPCG = Linear::PCG_PreconditionedRes<WrappedVXD, real>;
        auto pcg = tPCG([n](WrappedVXD &v)
                        { v.resize(n); });
        Eigen::MatrixXd A;
        A.resize(n, n);
        std::mt19937 gen(__seed);                        // Mersenne Twister random number engine
        std::uniform_real_distribution<> dis(-1.0, 1.0); // Uniform distribution between -1 and 1
        for (int i = 0; i < n * n; i++)
            A(i) = dis(gen);
        A = A * A.transpose();
        WrappedVXD b, x;
        b.setOnes(n);
        x.setZero(n);
        Eigen::MatrixXd M = A.diagonal().asDiagonal();
        M.setIdentity();
        Eigen::MatrixXd MIA = M.inverse() * A;

        {
            std::cout << std::setprecision(16);
            std::cout << "A" << std::endl;
            std::cout << A << std::endl;
            std::cout << "M" << std::endl;
            std::cout << M << std::endl;
            std::cout << std::defaultfloat;
            std::cout.unsetf(std::ios::floatfield);
            std::cout << std::endl;
        }

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

        std::cout << x << std::endl;
    }

    void testGMRES()
    {
        int n = 10;
        int nk = 3;
        using tGMRES = Linear::GMRES_LeftPreconditioned<WrappedVXD>;
        auto gmres = tGMRES(nk, [n](WrappedVXD &v)
                            { v.resize(n); });
        Eigen::MatrixXd A;
        A.resize(n, n);
        std::mt19937 gen(__seed);                        // Mersenne Twister random number engine
        std::uniform_real_distribution<> dis(-1.0, 1.0); // Uniform distribution between -1 and 1
        for (int i = 0; i < n * n; i++)
            A(i) = dis(gen);
        A = A * A.transpose();
        WrappedVXD b, x;
        b.setOnes(n);
        x.setZero(n);
        Eigen::MatrixXd M = A.diagonal().asDiagonal();
        // M.setIdentity();
        Eigen::MatrixXd MIA = M.inverse() * A;

        {
            std::cout << std::setprecision(16);
            std::cout << "A" << std::endl;
            std::cout << A << std::endl;
            std::cout << "M" << std::endl;
            std::cout << M << std::endl;
            std::cout << std::defaultfloat;
            std::cout.unsetf(std::ios::floatfield);
            std::cout << std::endl;
        }

        gmres.solve(
            [&](WrappedVXD &x, WrappedVXD &Ax)
            {
                Ax = A * x;
            },
            [&](WrappedVXD &x, WrappedVXD &Ax)
            {
                Ax = M.inverse() * x;
            },
            [&](WrappedVXD &a, WrappedVXD &b)
            {
                return a.dot(b);
            },
            b, x,
            5,
            [&](int iRestart, real res, real resB)
            {
                std::cout << "LGMRES: " << iRestart << ", res = " << res << ", ";
                std::cout << 0.5 * (A * x).dot(x) - b.dot(x) << std::endl;
                return false;
            });
        std::cout << x << std::endl;
    }
}

int main()
{
    std::cout << "\n\n=== FPCG: \n\n";
    DNDS::testPCG();
    std::cout << "\n\n=== LGMRES: \n\n";
    DNDS::testGMRES();
    return 0;
}