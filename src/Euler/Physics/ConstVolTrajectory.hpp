#pragma once

#include "PhysicsProperties.hpp"

#include "cantera/zerodim.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace DNDS::Euler::Reactive0D
{
    struct ConstVolOptions
    {
        int maxNewtonIterations = 80;
        double newtonTolerance = 1e-12;
        double canteraTemperatureFloor = 200.0;
    };

    struct ConstVolCase
    {
        Eigen::VectorXd U;
        double dtCode = 0;
        int nSteps = 0;
        int outputEvery = 1;
        double U0 = 1.0;
        double rho0 = 1.0;
        double L0 = 1.0;
    };

    struct StateSample
    {
        double tCode = 0;
        double tPhys = 0;
        double T = 0;
        double p = 0;
        std::vector<double> Y;
    };

    inline void validateCaseScales(const ConstVolCase &c)
    {
        DNDS_check_throw_info(std::isfinite(c.U0) && c.U0 > 0 && std::isfinite(c.rho0) && c.rho0 > 0 &&
                                  std::isfinite(c.L0) && c.L0 > 0,
                              fmt::format("ConstVolTrajectory requires positive finite scales [U0,rho0,L0]=[{:.3e},{:.3e},{:.3e}]",
                                          c.U0, c.rho0, c.L0));
    }

    template <int dim>
    inline void validateStateSize(const ConstVolCase &c, int nSpecies)
    {
        int nRequired = dim + 2 + nSpecies - 1;
        DNDS_check_throw_info(c.U.size() >= nRequired,
                              fmt::format("ConstVolTrajectory state has {} entries; expected at least {} for dim={} and nSpecies={}",
                                          c.U.size(), nRequired, dim, nSpecies));
    }

    inline void validateTrajectoryControls(const ConstVolCase &c)
    {
        DNDS_check_throw_info(std::isfinite(c.dtCode) && c.dtCode >= 0,
                              fmt::format("ConstVolTrajectory requires nonnegative finite dtCode, got {:.3e}", c.dtCode));
        DNDS_check_throw_info(c.nSteps >= 0,
                              fmt::format("ConstVolTrajectory requires nonnegative nSteps, got {}", c.nSteps));
        DNDS_check_throw_info(c.outputEvery > 0,
                              fmt::format("ConstVolTrajectory requires positive outputEvery, got {}", c.outputEvery));
    }

    inline std::vector<double> massFractions(const Chemistry::ChemicalSource &chem, const Eigen::VectorXd &U)
    {
        int Ns = chem.nSpecies();
        int Ns1 = Ns - 1;
        int Isp = static_cast<int>(U.size()) - Ns1;
        auto Yv = chem.massFractions(U[0], U.data() + Isp, Ns1);
        std::vector<double> Y(Ns);
        for (int k = 0; k < Ns; ++k)
            Y[k] = Yv[k];
        return Y;
    }

    template <EulerModel model, int dim>
    StateSample sampleState(const ConstVolCase &c, PhysicsProperties<model> &phys,
                            const Chemistry::ChemicalSource &chem, const Eigen::VectorXd &U, int step)
    {
        validateCaseScales(c);
        auto Y = massFractions(chem, U);
        Chemistry::ConstSpeciesBufferView Yv{Y.data(), static_cast<int>(Y.size())};
        double TCode = phys.template temperature<dim>(U);
        double TPhys = phys.toPhysT(TCode);
        double pCode = U[0] * phys.toCode(chem.mixtureR(Yv)) * TCode;
        return {step * c.dtCode, step * c.dtCode * c.L0 / c.U0, TPhys, phys.toPhysP(pCode), std::move(Y)};
    }

    template <EulerModel model, int dim>
    void stepImplicit(const ConstVolCase &c, PhysicsProperties<model> &phys, Chemistry::ChemicalSource &chem,
                      Eigen::VectorXd &U, const ConstVolOptions &options = ConstVolOptions{})
    {
        DNDS_check_throw_info(options.maxNewtonIterations > 0,
                              fmt::format("ConstVolTrajectory requires positive maxNewtonIterations, got {}",
                                          options.maxNewtonIterations));
        DNDS_check_throw_info(std::isfinite(options.newtonTolerance) && options.newtonTolerance > 0,
                              fmt::format("ConstVolTrajectory requires positive finite newtonTolerance, got {:.3e}",
                                          options.newtonTolerance));
        DNDS_check_throw_info(std::isfinite(options.canteraTemperatureFloor) && options.canteraTemperatureFloor > 0,
                              fmt::format("ConstVolTrajectory requires positive finite canteraTemperatureFloor, got {:.3e}",
                                          options.canteraTemperatureFloor));
        int Ns = chem.nSpecies();
        validateCaseScales(c);
        validateStateSize<dim>(c, Ns);
        int Ns1 = Ns - 1;
        int nVars = static_cast<int>(U.size());
        int Isp = nVars - Ns1;
        int I4 = Isp - 1;
        auto MW = chem.molecularWeights();
        double invS0 = c.L0 / (c.rho0 * c.U0);

        Eigen::VectorXd Uk = U;
        bool converged = false;
        double finalStepNorm = 0.0;
        for (int iter = 0; iter < options.maxNewtonIterations; ++iter)
        {
            auto Y = massFractions(chem, Uk);
            Chemistry::ConstSpeciesBufferView Yv{Y.data(), Ns};
            double TCode = phys.template temperature<dim>(Uk);
            double TPhys = phys.toPhysT(TCode);
            double pPhys = phys.toPhysP(Uk[0] * phys.toCode(chem.mixtureR(Yv)) * TCode);

            std::vector<double> omega(Ns);
            double TCantera = std::max(TPhys, options.canteraTemperatureFloor);
            chem.productionRates(TCantera, pPhys, Yv, Chemistry::SpeciesBufferView{omega.data(), Ns});

            Eigen::VectorXd ret = Eigen::VectorXd::Zero(nVars);
            for (int k = 0; k < Ns1; ++k)
                ret[Isp + k] = omega[k] * MW[k] * invS0;

            std::vector<double> jbuf(Ns * nVars, 0.0);
            chem.productionRatesAndJacobian(TCantera, pPhys, Uk[0], Uk[I4], 0.0, 0.0, 0.0, I4,
                                            c.U0, c.rho0, Yv, Chemistry::SpeciesBufferView{omega.data(), Ns},
                                            Chemistry::JacobianBufferView{jbuf.data(), Ns, nVars, Ns});

            Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(nVars, nVars);
            for (int k = 0; k < Ns1; ++k)
                for (int j = 0; j < nVars; ++j)
                    jac(Isp + k, j) = MW[k] * jbuf[k + j * Ns] * invS0;

            Eigen::VectorXd F = Uk - U - c.dtCode * ret;
            Eigen::MatrixXd Jn = Eigen::MatrixXd::Identity(nVars, nVars) - c.dtCode * jac;
            for (int r = 0; r <= I4; ++r)
            {
                Jn.row(r) = Eigen::VectorXd::Unit(nVars, r);
                F[r] = 0;
            }

            Eigen::VectorXd dU = Jn.partialPivLu().solve(-F);
            finalStepNorm = dU.lpNorm<Eigen::Infinity>();
            DNDS_check_throw_info(std::isfinite(finalStepNorm), "ConstVolTrajectory Newton produced non-finite update");
            Uk += dU;
            for (int k = Isp; k < nVars; ++k)
                Uk[k] = std::max(Uk[k], 1e-30);
            if (finalStepNorm < options.newtonTolerance)
            {
                converged = true;
                break;
            }
        }

        DNDS_check_throw_info(converged,
                              fmt::format("ConstVolTrajectory Newton failed to converge in {} iterations, final step norm={:.3e}",
                                          options.maxNewtonIterations, finalStepNorm));
        U = Uk;
    }

    template <EulerModel model, int dim>
    std::vector<StateSample> runDNDSRTrajectory(const ConstVolCase &c, PhysicsProperties<model> &phys,
                                                Chemistry::ChemicalSource &chem,
                                                const ConstVolOptions &options = ConstVolOptions{})
    {
        validateCaseScales(c);
        validateStateSize<dim>(c, chem.nSpecies());
        validateTrajectoryControls(c);
        std::vector<StateSample> out;
        Eigen::VectorXd U = c.U;
        out.push_back(sampleState<model, dim>(c, phys, chem, U, 0));
        for (int step = 1; step <= c.nSteps; ++step)
        {
            stepImplicit<model, dim>(c, phys, chem, U, options);
            if (step % c.outputEvery == 0 || step == c.nSteps)
                out.push_back(sampleState<model, dim>(c, phys, chem, U, step));
        }
        return out;
    }

    inline std::vector<StateSample> runCanteraTrajectory(const ConstVolCase &c, const std::string &mechPath,
                                                         const StateSample &initial,
                                                         const std::vector<StateSample> &times)
    {
        validateCaseScales(c);
        DNDS_check_throw_info(c.U.size() > 0, "ConstVolTrajectory Cantera reference requires a nonempty state vector");
        DNDS_check_throw_info(!times.empty(), "ConstVolTrajectory Cantera reference requires nonempty sample times");
        auto sol = Cantera::newSolution(mechPath, "", "none");
        auto gas = sol->thermo();
        gas->setMassFractions_NoNorm(initial.Y.data());
        gas->setState_TD(initial.T, c.rho0 * c.U[0]);
        auto reactor = Cantera::newReactorBase("IdealGasReactor", sol);
        Cantera::ReactorNet net(reactor);

        std::vector<StateSample> out;
        out.reserve(times.size());
        double lastTime = -std::numeric_limits<double>::infinity();
        for (const auto &target : times)
        {
            DNDS_check_throw_info(std::isfinite(target.tPhys) && target.tPhys >= lastTime,
                                  "ConstVolTrajectory Cantera reference sample times must be monotone");
            if (target.tPhys > 0)
                net.advance(target.tPhys);
            auto &th = *reactor->phase()->thermo();
            std::vector<double> Y(th.nSpecies());
            th.getMassFractions(Y.data());
            out.push_back({target.tCode, target.tPhys, th.temperature(), th.pressure(), std::move(Y)});
            lastTime = target.tPhys;
        }
        return out;
    }

    inline double ignitionTime(const std::vector<StateSample> &hist, double threshold)
    {
        DNDS_check_throw_info(!hist.empty(), "ConstVolTrajectory ignitionTime requires a nonempty history");
        for (size_t i = 1; i < hist.size(); ++i)
        {
            if (hist[i - 1].T <= threshold && hist[i].T >= threshold)
            {
                double a = (threshold - hist[i - 1].T) / std::max(hist[i].T - hist[i - 1].T, 1e-300);
                return hist[i - 1].tPhys + a * (hist[i].tPhys - hist[i - 1].tPhys);
            }
        }
        return hist.back().tPhys;
    }
}
