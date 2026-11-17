// Qualification-only t=0 projection on the periodic IV10_20 Cartesian lattice.
// The production Cauchy spatial rate certifies every iterate and the checkpoint.
#include "Cauchy/Core/Simulation.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <Eigen/Dense>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace C = DNDS::Cauchy;
namespace
{
    using Real = DNDS::real;
    using Grid = Eigen::MatrixXd;
    using Spectrum = Eigen::MatrixXcd;
    constexpr int N = 20;
    constexpr Real h = .5;
    constexpr Real volume = h * h;
    constexpr int maxIterations = 8;

    class PeriodicVortex
    {
        C::Domain &_domain;
        std::vector<std::array<int, 2>> _xy;
        std::vector<std::array<Real, 4>> _initial;

    public:
        explicit PeriodicVortex(C::Domain &domain) : _domain(domain)
        {
            auto metrics = domain.Discretization().Metrics().AcquireRead();
            auto mean = domain.Dofs().CoupledField().Owned().Read();
            const auto view = mean.View();
            Eigen::MatrixXi membership = Eigen::MatrixXi::Zero(N, N);
            for (DNDS::index iCell = 0; iCell < view.RowCount(); ++iCell)
            {
                const auto x = metrics.HostView().GetCellBary(iCell);
                std::array<int, 2> ij{};
                for (int d = 0; d < 2; ++d)
                {
                    const int image = int(std::round(x[d] / h - .5));
                    DNDS_check_throw_info(std::abs(x[d] - (image + .5) * h) < 1e-9,
                                          "expected the periodic IV10_20 cell centers");
                    ij[d] = (image % N + N) % N;
                }
                DNDS_check_throw_info(std::abs(metrics.HostView().GetCellVol(iCell) - volume) < 1e-9,
                                      "expected the uniform IV10_20 cell volumes");
                _xy.push_back(ij);
                _initial.push_back({view(iCell, 0), view(iCell, 1), view(iCell, 2), view(iCell, 3)});
                ++membership(ij[0], ij[1]);
            }
            MPI_Allreduce(MPI_IN_PLACE, membership.data(), N * N, MPI_INT, MPI_SUM, _domain.Communicator());
            DNDS_check_throw_info((membership.array() == 1).all(), "each IV10_20 lattice cell must have exactly one owner");
        }
        /// Integrated continuity -> code-time rate on the uniquely owned grid.
        Grid Rate()
        {
            auto read = _domain.Residual().CoupledField().Owned().Read();
            const auto rate = read.View();
            Grid g = Grid::Zero(N, N);
            for (DNDS::index i = 0; i < rate.RowCount(); ++i)
                g(_xy[i][0], _xy[i][1]) = rate(i, 3) / volume;
            MPI_Allreduce(MPI_IN_PLACE, g.data(), N * N, MPI_DOUBLE, MPI_SUM, _domain.Communicator());
            DNDS_check_throw_info(g.allFinite(), "nonfinite native spatial continuity residual");
            return g;
        }
        /// With p constant, Rusanov's pressure dissipation vanishes and the
        /// unlimited reconstruction/central face-average divergence is linear.
        Grid Impulse(int direction)
        {
            {
                auto write = _domain.Dofs().CoupledField().Owned().Write();
                const auto q = write.View();
                for (DNDS::index i = 0; i < q.RowCount(); ++i)
                    for (int j = 0; j < 4; ++j)
                        q(i, j) = j == direction && _xy[i][0] == 0 && _xy[i][1] == 0 ? 1 : 0;
            }
            return Rate();
        }
        void SetCorrection(const Grid &dx, const Grid &dy)
        {
            auto write = _domain.Dofs().CoupledField().Owned().Write();
            const auto q = write.View();
            for (DNDS::index i = 0; i < q.RowCount(); ++i)
            {
                q(i, 0) = _initial[i][0] + dx(_xy[i][0], _xy[i][1]);
                q(i, 1) = _initial[i][1] + dy(_xy[i][0], _xy[i][1]);
                q(i, 2) = _initial[i][2];
                q(i, 3) = _initial[i][3];
            }
        }
    };

    Real RMS(const Grid &g) { return g.norm() / N; }

    int Project(const DNDS::MPIInfo &mpi, const char *configPath, const char *checkpoint)
    {
        DNDS_check_throw_info(!std::filesystem::exists(checkpoint), "choose a fresh checkpoint path");
        C::Json input;
        std::ifstream(configPath) >> input;
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        DNDS_check_throw_info(config.Domains().size() == 1, "expected one vortex domain");
        const auto &settings = config.Domains().front().settings;
        DNDS_check_throw_info(settings.numerics.flux == "rusanov" && settings.numerics.limiter == "none" &&
                                  settings.numerics.reconstruction == "variational" && settings.numerics.cfv.maxOrder == 3 &&
                                  settings.execution.device == "cpu" && !settings.numerics.positivity,
                              "projection requires the CPU unlimited P3 Rusanov qualification operator");
        C::Simulation simulation(config, mpi.comm);
        auto &domain = simulation.GetDomain(0);
        DNDS_check_throw_info(domain.Equation().IsACM() && domain.Dofs().ComponentNumber() == 4 &&
                                  domain.Time() == 0 && !domain.Equation().Viscous(),
                              "expected the inviscid flow4 initial state");
        PeriodicVortex fixture(domain);
        const Real target = .1 * settings.implicit.algebraicRateTolerance;
        const Real initialRMS = RMS(fixture.Rate());
        Spectrum f(N, N);
        for (int k = 0; k < N; ++k)
            for (int x = 0; x < N; ++x)
                f(k, x) = std::polar(1., -2 * std::acos(-1.) * k * x / N);
        const auto forward = [&](const Grid &g) -> Spectrum
        { return f * g.cast<std::complex<Real>>() * f.transpose(); };
        const auto inverse = [&](const Spectrum &g) -> Grid
        { return (f.adjoint() * g * f.conjugate()).real() / (N * N); };
        const Spectrum sx = forward(fixture.Impulse(0)), sy = forward(fixture.Impulse(1));
        const Grid norm2 = sx.cwiseAbs2() + sy.cwiseAbs2();
        const Real cutoff = 1e-16 * norm2.maxCoeff(); // Squared relative singular-value cutoff of 1e-8.
        Grid dx = Grid::Zero(N, N), dy = dx;
        fixture.SetCorrection(dx, dy);
        Grid rate = fixture.Rate();
        for (int iteration = 0; iteration < maxIterations && RMS(rate) > target; ++iteration)
        {
            // Least-L2 change to (u,v) solving D delta = -g on each resolvable
            // Fourier mode. D is calibrated once. At variable analytic p the
            // wavespeed changes with velocity; accept only an actual decrease
            // in the full native residual, including its discarded null modes.
            const Spectrum rhs = forward(rate);
            Spectrum ux = Spectrum::Zero(N, N), uy = ux;
            for (int k = 0; k < N; ++k)
                for (int l = 0; l < N; ++l)
                    if ((k || l) && norm2(k, l) > cutoff)
                    {
                        ux(k, l) = -std::conj(sx(k, l)) * rhs(k, l) / norm2(k, l);
                        uy(k, l) = -std::conj(sy(k, l)) * rhs(k, l) / norm2(k, l);
                    }
            const Grid stepX = inverse(ux), stepY = inverse(uy);
            bool improved = false;
            for (Real weight = 1; weight >= 1. / 64; weight *= .5)
            {
                fixture.SetCorrection(dx + weight * stepX, dy + weight * stepY);
                Grid trial = fixture.Rate();
                if (RMS(trial) < RMS(rate))
                {
                    dx += weight * stepX;
                    dy += weight * stepY;
                    rate = std::move(trial);
                    improved = true;
                    break;
                }
            }
            if (!mpi.rank)
                std::cout << C::Json{{"iteration", iteration + 1}, {"continuity_rms", RMS(rate)}, {"improved", improved}}.dump() << std::endl;
            if (!improved)
                break;
        }
        if (!mpi.rank)
        {
            Spectrum null = forward(rate);
            for (int k = 0; k < N; ++k)
                for (int l = 0; l < N; ++l)
                    if ((k || l) && norm2(k, l) > cutoff)
                        null(k, l) = 0;
            std::cout << C::Json{{"initial_rms", initialRMS}, {"final_rms", RMS(rate)}, {"target", target}, {"velocity_change_rms", std::hypot(RMS(dx), RMS(dy))}, {"mean_du", dx.mean()}, {"mean_dv", dy.mean()}, {"velocity_change_max", (dx.array().square() + dy.array().square()).sqrt().maxCoeff()}, {"unresolved_modes", (norm2.array() <= cutoff).count()}, {"unresolved_continuity_rms", RMS(inverse(null))}, {"time", domain.Time()}}.dump() << std::endl;
        }
        if (RMS(rate) > target)
            return 2;
        simulation.WriteCheckpoint(checkpoint, false);
        return 0;
    }
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    int result = 1;
    try
    {
        const DNDS::MPI::SolverFailureScope failures(mpi.comm);
        DNDS_check_throw_info(argc == 3, "usage: cauchy_experiment_acm_vortex_projection CONFIG NEW_CHECKPOINT.h5");
        result = Project(mpi, argv[1], argv[2]);
    }
    catch (const std::exception &error)
    {
        DNDS::MPI::AbortCaught(mpi.comm, error.what());
    }
    catch (...)
    {
        DNDS::MPI::AbortCaught(mpi.comm, "unknown vortex projection failure");
    }
    DNDS::MPI::Finalize();
    return result;
}
