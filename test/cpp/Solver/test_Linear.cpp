/**
 * @file test_Linear.cpp
 * @brief Unit tests for GMRES and PCG iterative linear solvers in Solver/Linear.hpp.
 *
 * Tests solve Ax=b for small dense systems where the exact solution is known.
 * Uses a thin Vec wrapper around Eigen::VectorXd.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Solver/Linear.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

using namespace DNDS;

// ===================================================================
// Wrapper for Eigen::VectorXd satisfying TDATA interface
// ===================================================================
struct DVec
{
    Eigen::VectorXd data;
    DVec() = default;
    DVec(int n) : data(Eigen::VectorXd::Zero(n)) {}
    DVec &operator=(const DVec &) = default;
    void operator+=(const DVec &o) { data += o.data; }
    void operator*=(double s) { data *= s; }
    void operator*=(const DVec &o) { data.array() *= o.data.array(); }
    void setConstant(double s) { data.setConstant(s); }
    void addTo(const DVec &o, double s) { data += o.data * s; }
};

static auto dvinit(int n)
{
    return [n](DVec &v) { v = DVec(n); };
}

// ===================================================================
// GMRES: solve 3x3 SPD system
// ===================================================================

TEST_CASE("GMRES: solve 3x3 SPD system exactly")
{
    const int n = 3;
    Eigen::Matrix3d A;
    A << 4, 1, 0,
         1, 3, 1,
         0, 1, 2;
    Eigen::Vector3d b_exact(1, 2, 3);
    Eigen::Vector3d x_exact = A.ldlt().solve(b_exact);

    Linear::GMRES_LeftPreconditioned<DVec> gmres(n, dvinit(n));

    DVec b(n), x(n);
    b.data = b_exact;
    x.data.setZero();

    auto FA = [&](DVec &xin, DVec &Ax) { Ax.data = A * xin.data; };
    auto FML = [&](DVec &xin, DVec &MLx) { MLx = xin; }; // no preconditioner
    auto fDot = [](DVec &a, DVec &b) -> double { return a.data.dot(b.data); };
    auto fStop = [](int, double res, double resB) -> bool { return res < 1e-14 * resB; };

    bool converged = gmres.solve(FA, FML, fDot, b, x, 10, fStop);
    CHECK(converged);

    double err = (x.data - x_exact).norm();
    std::cout << "[GMRES/3x3] err=" << std::scientific << err << std::endl;
    CHECK(err < 1e-12);
}

TEST_CASE("GMRES: solve 10x10 random SPD system")
{
    const int n = 10;
    Eigen::MatrixXd R = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd A = R.transpose() * R + n * Eigen::MatrixXd::Identity(n, n); // SPD
    Eigen::VectorXd b_exact = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x_exact = A.ldlt().solve(b_exact);

    Linear::GMRES_LeftPreconditioned<DVec> gmres(n, dvinit(n));

    DVec b(n), x(n);
    b.data = b_exact;
    x.data.setZero();

    auto FA = [&](DVec &xin, DVec &Ax) { Ax.data = A * xin.data; };
    auto FML = [&](DVec &xin, DVec &MLx) { MLx = xin; };
    auto fDot = [](DVec &a, DVec &b) -> double { return a.data.dot(b.data); };
    auto fStop = [](int, double res, double resB) -> bool { return res < 1e-12 * resB; };

    bool converged = gmres.solve(FA, FML, fDot, b, x, 20, fStop);
    CHECK(converged);

    double err = (x.data - x_exact).norm();
    std::cout << "[GMRES/10x10] err=" << std::scientific << err << std::endl;
    CHECK(err < 1e-10);
}

TEST_CASE("GMRES: diagonal preconditioner improves convergence")
{
    const int n = 10;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++)
        A(i, i) = (i + 1) * 10.0; // condition number = 100
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    Eigen::VectorXd b_exact = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x_exact = A.fullPivLu().solve(b_exact);

    // Without preconditioner
    Linear::GMRES_LeftPreconditioned<DVec> gmres1(3, dvinit(n));
    DVec b1(n), x1(n);
    b1.data = b_exact;
    int restarts_npc = 0;
    auto fStop1 = [&](int iRestart, double res, double resB) -> bool {
        restarts_npc = iRestart;
        return res < 1e-12 * resB;
    };
    gmres1.solve(
        [&](DVec &xin, DVec &Ax) { Ax.data = A * xin.data; },
        [](DVec &xin, DVec &MLx) { MLx = xin; },
        [](DVec &a, DVec &b) -> double { return a.data.dot(b.data); },
        b1, x1, 50, fStop1);

    // With diagonal preconditioner
    Linear::GMRES_LeftPreconditioned<DVec> gmres2(3, dvinit(n));
    DVec b2(n), x2(n);
    b2.data = b_exact;
    int restarts_pc = 0;
    auto fStop2 = [&](int iRestart, double res, double resB) -> bool {
        restarts_pc = iRestart;
        return res < 1e-12 * resB;
    };
    Eigen::VectorXd diagInv = A.diagonal().cwiseInverse();
    gmres2.solve(
        [&](DVec &xin, DVec &Ax) { Ax.data = A * xin.data; },
        [&](DVec &xin, DVec &MLx) { MLx.data = diagInv.asDiagonal() * xin.data; },
        [](DVec &a, DVec &b) -> double { return a.data.dot(b.data); },
        b2, x2, 50, fStop2);

    double err2 = (x2.data - x_exact).norm();
    CHECK(err2 < 1e-10);
    // Preconditioned should converge in fewer restarts (or same)
    CHECK(restarts_pc <= restarts_npc);
}

// ===================================================================
// PCG: solve SPD system
// ===================================================================

TEST_CASE("PCG: solve 5x5 SPD system")
{
    const int n = 5;
    Eigen::MatrixXd R = Eigen::MatrixXd::Random(n, n);
    R << 1, 0.1, 0, 0, 0,
         0.1, 2, 0.2, 0, 0,
         0, 0.2, 3, 0.3, 0,
         0, 0, 0.3, 4, 0.1,
         0, 0, 0, 0.1, 5;
    Eigen::MatrixXd A = (R + R.transpose()) / 2.0; // symmetric
    Eigen::VectorXd b_exact = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x_exact = A.ldlt().solve(b_exact);

    using TScalar = double;
    Linear::PCG_PreconditionedRes<DVec, TScalar> pcg(dvinit(n));

    DVec x(n);
    x.data.setZero();

    auto FA = [&](DVec &p, DVec &Ap) { Ap.data = A * p.data; };
    auto FM = [&](DVec &z, DVec &r) { r = z; }; // no preconditioning
    auto FResPrec = [&](DVec &xin, DVec &z) { z.data = b_exact - A * xin.data; };
    auto fDot = [](DVec &a, DVec &b) -> TScalar { return a.data.dot(b.data); };
    auto fStop = [](int, TScalar zrDot, TScalar) -> bool { return std::abs(zrDot) < 1e-24; };

    bool converged = pcg.solve(FA, FM, FResPrec, fDot, x, 100, fStop);
    CHECK(converged);

    double err = (x.data - x_exact).norm();
    std::cout << "[PCG/5x5] err=" << std::scientific << err << std::endl;
    CHECK(err < 1e-10);
}
