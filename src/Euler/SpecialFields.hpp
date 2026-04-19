/** @file SpecialFields.hpp
 *  @brief Analytic isentropic-vortex solutions for inviscid accuracy verification.
 *
 *  Provides three variants of the classical isentropic vortex used as exact
 *  solutions for the Euler equations:
 *  - IsentropicVortex10 — moving vortex on a [0,10]^2 periodic domain.
 *  - IsentropicVortex30 — moving vortex on a [-10,20]^2 periodic domain (period 30).
 *  - IsentropicVortexCent — stationary vortex centered at the origin (no periodicity).
 *
 *  All three share the same isentropic vortex formula:
 *  - Temperature perturbation: dT = -(gamma-1)/(8*gamma*pi^2) * chi^2 * exp(1-r^2)
 *  - Velocity perturbation (solid-body rotation):
 *    - du_x = -chi/(2*pi) * exp((1-r^2)/2) * (y - y_c)
 *    - du_y = +chi/(2*pi) * exp((1-r^2)/2) * (x - x_c)
 *  - T = 1 + dT,  rho = T^(1/(gamma-1)),  p = T * rho
 *
 *  The returned state vector is in conservative variables: (rho, rho*u, rho*v, [rho*w,] E).
 */
#pragma once

#include "EulerEvaluator.hpp"

namespace DNDS::Euler::SpecialFields
{
    /**
     * @brief Analytic isentropic vortex on a [0,10] x [0,10] periodic domain.
     *
     * Background flow (u,v) = (1,1), vortex center at (5,5). The domain wraps
     * with period 10 in both x and y via float_mod, so the vortex convects
     * diagonally and re-enters from the opposite corner. Used for inviscid
     * accuracy verification (spatial and temporal order-of-accuracy studies).
     *
     * @tparam model  EulerModel selecting the equation set (default NS = 2-D Navier-Stokes).
     * @param eval    EulerEvaluator providing gas properties (gamma).
     * @param x       Physical-space coordinates at which to evaluate the solution.
     * @param t       Current physical time (the vortex translates at speed (1,1)*t).
     * @param cnVars  Number of conservative variables in the returned vector.
     * @param chi     Vortex strength parameter (standard test value = 5).
     * @return Conservative state vector (rho, rho*u_x, rho*u_y, ..., E) of size cnVars.
     */
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

    /**
     * @brief Analytic isentropic vortex on a [-10,20] x [-10,20] periodic domain (period 30).
     *
     * Same vortex formula as IsentropicVortex10 but on a larger domain with
     * period 30. Background flow (u,v) = (1,1), vortex center at (5,5),
     * chi = 5 (hardcoded). Useful when more space around the vortex core is
     * needed to reduce periodic-image interactions in convergence studies.
     *
     * @tparam model  EulerModel selecting the equation set (default NS).
     * @param eval    EulerEvaluator providing gas properties (gamma).
     * @param x       Physical-space coordinates at which to evaluate the solution.
     * @param t       Current physical time (vortex translates at speed (1,1)*t).
     * @param cnVars  Number of conservative variables in the returned vector.
     * @return Conservative state vector (rho, rho*u_x, rho*u_y, ..., E) of size cnVars.
     */
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

    /**
     * @brief Stationary isentropic vortex centered at the origin with zero background flow.
     *
     * Same vortex formula as IsentropicVortex10 but with (u_bg, v_bg) = (0,0) and
     * the vortex center at the origin (0,0). No periodic wrapping is applied, so
     * the computational domain should be large enough that the vortex decays
     * before reaching the boundaries. chi = 5 (hardcoded).
     *
     * Used for steady-state vortex preservation tests to verify that the
     * numerical scheme maintains the stationary vortex without spurious
     * dissipation or distortion.
     *
     * @tparam model  EulerModel selecting the equation set (default NS).
     * @param eval    EulerEvaluator providing gas properties (gamma).
     * @param x       Physical-space coordinates at which to evaluate the solution.
     * @param t       Current physical time (unused since the vortex is stationary).
     * @param cnVars  Number of conservative variables in the returned vector.
     * @return Conservative state vector (rho, rho*u_x, rho*u_y, ..., E) of size cnVars.
     */
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