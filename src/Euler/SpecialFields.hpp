#pragma once

#include "EulerEvaluator.hpp"

namespace DNDS::Euler::SpecialFields
{
    template <EulerModel model = NS>
    auto IsentropicVortex10(
        EulerEvaluator<model> &eval,
        const Geom::tPoint &x,
        real t, int cnVars, 
        real chi)
    {
        typename EulerEvaluator<model>::TU ret;
        ret.resize(cnVars);
        
        real xyc = 5;
        real gamma = eval.settings.idealGasProperty.gamma;
        Geom::tPoint pPhysics = x;
        pPhysics[0] = float_mod(pPhysics[0] - t, 10);
        pPhysics[1] = float_mod(pPhysics[1] - t, 10);
        real r = std::sqrt(sqr(pPhysics(0) - xyc) + sqr(pPhysics(1) - xyc));
        real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r));
        real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - xyc);
        real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - xyc);
        real T = dT + 1;
        real ux = dux + 1;
        real uy = duy + 1;
        real S = 1;
        real rho = std::pow(T / S, 1 / (gamma - 1));
        real p = T * rho;

        real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

        ret.setZero();
        ret(0) = rho;
        ret(1) = rho * ux;
        ret(2) = rho * uy;
        ret(EulerEvaluator<model>::dim + 1) = E;
        return ret;
    }

    template <EulerModel model = NS>
    auto IsentropicVortex30(
        EulerEvaluator<model> &eval,
        const Geom::tPoint &x,
        real t, int cnVars)
    {
        typename EulerEvaluator<model>::TU ret;
        ret.resize(cnVars);

        real chi = 5;
        real xyc = 5;
        real gamma = eval.settings.idealGasProperty.gamma;
        Geom::tPoint pPhysics = x;
        pPhysics[0] = float_mod(pPhysics[0] - t + 10, 30) - 10;
        pPhysics[1] = float_mod(pPhysics[1] - t + 10, 30) - 10;
        real r = std::sqrt(sqr(pPhysics(0) - xyc) + sqr(pPhysics(1) - xyc));
        real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r));
        real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - xyc);
        real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - xyc);
        real T = dT + 1;
        real ux = dux + 1;
        real uy = duy + 1;
        real S = 1;
        real rho = std::pow(T / S, 1 / (gamma - 1));
        real p = T * rho;

        real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

        ret.setZero();
        ret(0) = rho;
        ret(1) = rho * ux;
        ret(2) = rho * uy;
        ret(EulerEvaluator<model>::dim + 1) = E;
        return ret;
    }

    template <EulerModel model = NS>
    auto IsentropicVortexCent(
        EulerEvaluator<model> &eval,
        const Geom::tPoint &x,
        real t, int cnVars)
    {
        typename EulerEvaluator<model>::TU ret;
        ret.resize(cnVars);

        real chi = 5;
        real xyc = 0; // center is origin
        real gamma = eval.settings.idealGasProperty.gamma;
        Geom::tPoint pPhysics = x;
        // pPhysics[0] = float_mod(pPhysics[0] - t, 10);
        // pPhysics[1] = float_mod(pPhysics[1] - t, 10);
        real r = std::sqrt(sqr(pPhysics(0) - xyc) + sqr(pPhysics(1) - xyc));
        real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r));
        real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - xyc);
        real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - xyc);
        real T = dT + 1;
        real ux = dux + 0; // no translation
        real uy = duy + 0;
        real S = 1;
        real rho = std::pow(T / S, 1 / (gamma - 1));
        real p = T * rho;

        real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

        ret.setZero();
        ret(0) = rho;
        ret(1) = rho * ux;
        ret(2) = rho * uy;
        ret(EulerEvaluator<model>::dim + 1) = E;
        return ret;
    }
}