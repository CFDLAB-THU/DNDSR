#pragma once
#include "Cauchy/Boundary/ACMBoundary.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "Physics/Incompressible/RANSPoint.hpp"
#include "Physics/RANS/PointKernels.hpp"

namespace DNDS::Cauchy::ACM
{
    template <int C, class Row>
    struct DenseBlockView
    {
        Row elements;
        bool rowMajor;
        DNDS_DEVICE_CALLABLE real &operator()(int i, int j) const { return elements[Algebra::DenseBlockEntry(i, j, C, rowMajor)]; }
    };
    template <int C, class Row>
    DNDS_DEVICE_CALLABLE auto DenseBlock(Row elements, bool rowMajor) { return DenseBlockView<C, Row>{elements, rowMajor}; }

    /// Fill the integrated, frozen pseudo-time and physical-time correction
    /// block for [u,v,w,p,scalars...]. Physical pressure has zero mass.
    /// During dual time, a primitive transported scalar q carries pseudo-mass
    /// q dp/dTau / beta², matching the continuity pseudo-mass in the Kirabo
    /// ACM formulation. Its physical derivative is still dq/dt.
    template <int C, class State, class Block>
    DNDS_DEVICE_CALLABLE void FillACMTemporalBlock(const State &state, real volume, real inversePseudoTime, real physicalInverseDt,
                                                   real alpha, real beta2, Block &&matrix)
    {
        static_assert(C >= 4 && C <= 6);
        const real pseudo = volume * inversePseudoTime, mass = volume * physicalInverseDt;
        for (int j = 0; j < C; ++j)
            for (int i = 0; i < C; ++i)
                matrix(i, j) = i == j ? pseudo + mass : 0;
        const real pressurePseudo = pseudo / beta2;
        matrix(3, 3) = pressurePseudo;
        for (int i = 0; i < 3; ++i)
            matrix(i, 3) = pressurePseudo * (1 + alpha) * state[i];
        if (physicalInverseDt > 0)
            for (int i = 4; i < C; ++i)
                matrix(i, 3) = pressurePseudo * state[i];
    }

    /// Kirabo's centered numerical linearization of the mean-state face flux.
    /// The observer composes any exterior-state or periodic mapping, so its
    /// derivative maps stored unknowns directly into the owning cell frame.
    /// Weight includes face measure and stage residual coefficient, with a
    /// positive sign because the integrated spatial residual is -div(F).
    template <int C, class Evaluate, class Block>
    DNDS_DEVICE_CALLABLE NumericalStatus AddACMFluxDerivative(const std::array<real, C> &base, real weight,
                                                              const Evaluate &evaluate, Block &&matrix)
    {
        for (int j = 0; j < C; ++j)
        {
            const real step = 1e-7 * std::max(real(1), std::abs(base[j]));
            auto plus = base, minus = base;
            plus[j] += step;
            minus[j] -= step;
            std::array<real, C> fluxPlus{}, fluxMinus{};
            if (const auto status = evaluate(plus, fluxPlus); !status)
                return status;
            if (const auto status = evaluate(minus, fluxMinus); !status)
                return status;
            const real scale = weight / (plus[j] - minus[j]);
            for (int i = 0; i < C; ++i)
            {
                const real entry = scale * (fluxPlus[i] - fluxMinus[i]);
                if (!std::isfinite(entry))
                    return {NumericalError::InvalidState};
                matrix(i, j) += entry;
            }
        }
        return {};
    }
    /// First-order matrix row in stored coordinates. The rank-local state view
    /// maps logical components; offsets and canonical incidence slots identify
    /// disjoint destination blocks. CPU rows and CUDA threads share the math.
    template <int C, class Request>
    DNDS_DEVICE_CALLABLE NumericalStatus AssembleRow(const Request &request, const FiniteVolumeStencilView &stencil, index cell)
    {
        using State = std::array<real, C>;
        const auto &metric = stencil.metric;
        const auto read = [&](index row)
        {
            State state{};
            for (int j = 0; j < C; ++j)
                state[j] = request.input(row, j);
            return state;
        };
        const real volume = metric.GetCellVol(cell), tau = request.tau(cell, 0);
        if (!(std::isfinite(tau) && tau >= 0 && volume > 0))
            return NumericalStatus{NumericalError::InvalidState}.At(cell);
        auto diagonal = DenseBlock<C>(request.diagonal[cell], request.offDiagonal.rowMajorBlocks);
        const auto own = read(cell);
        Eigen::Matrix<real, C, 3> gradient = Eigen::Matrix<real, C, 3>::Zero();
        FillACMTemporalBlock<C>(own, volume, tau, request.mass, request.physics.alpha, request.physics.beta2, diagonal);
        for (index slot = 0; slot < request.offDiagonal.BlockNumber(cell); ++slot)
            for (int k = 0; k < C * C; ++k)
                request.offDiagonal(cell, slot)[k] = 0;
        for (index entry = stencil.jacobianOffsets(cell, 0); entry < stencil.jacobianOffsets(cell + 1, 0); ++entry)
        {
            const auto sample = stencil.JacobianSample(cell, entry);
            const auto &bc = request.conditions[sample.condition];
            const bool boundary = sample.other == UnInitIndex;
            const auto remote = boundary ? State{} : read(sample.other);
            const real weight = request.residualScale * metric.GetFaceArea(sample.face);
            // Freeze the viscous spectral coefficient from aux preparation.
            // Only differential transported variables diffuse. The pressure
            // equation still participates in the complete inviscid flux.
            const real viscous = boundary && bc.kind == BoundaryKind::SlipWall ? 0 : .5 * request.radius(sample.auxiliaryFace, 1);
            real wallOmega = 0;
            if constexpr (C == 6)
                if (boundary && bc.kind == BoundaryKind::NoSlipWall)
                {
                    const real distance = request.wallMean(cell, 0);
                    if (!(distance > 0))
                        return NumericalStatus{NumericalError::InvalidState}.At(cell, sample.face);
                    wallOmega = request.physics.kinematicViscosity * Physics::RANS::kWallOmegaCoeff / sqr(distance);
                }
            const auto exterior = [&](const State &left, State &right) -> NumericalStatus
            {
                if (boundary)
                {
                    if (!ACM::Exterior(request.physics, bc.kind, left, bc.acmState, bc.velocity, sample.normal,
                                       right, C, wallOmega))
                        return {NumericalError::UnsupportedRequest};
                }
                else if (sample.changesFrame)
                {
                    const Geom::tPoint velocity{right[0], right[1], right[2]};
                    const auto rotated = (Eigen::Map<const Geom::tGPoint>(sample.rotation.data()) * velocity).eval();
                    for (int j = 0; j < 3; ++j)
                        right[j] = rotated[j];
                }
                return {};
            };
            const auto flux = [&](const State &left, State right, State &result) -> NumericalStatus
            {
                if (const auto status = exterior(left, right); !status)
                    return status;
                ACM::NumericalFlux(request.physics, left, right, sample.normal, result, C);
                for (int j = 0; j < C; ++j)
                    if (j != 3)
                        result[j] -= viscous * (right[j] - left[j]);
                // The former isotropic 2*rho*k/3 stress is absent from the ACM residual.
                return {};
            };
            if constexpr (C > 4)
            {
                // Mean-state, first-order Jacobian. Freeze the velocity
                // and scalar gradients of this same discretization while
                // evaluating the closure's positive destruction diagonal.
                // Subtract the cell mean to retain exact constant gradients.
                auto right = remote;
                if (const auto status = exterior(own, right); !status)
                    return status.At(cell, sample.face);
                const real gradScale = .5 * metric.GetFaceArea(sample.face) / volume;
                for (int j = 0; j < C; ++j)
                    for (int d = 0; d < 3; ++d)
                        gradient(j, d) += gradScale * (right[j] - own[j]) * sample.normal[d];
            }
            const auto leftFlux = [&](const State &q, State &out)
            { return flux(q, remote, out); };
            if (const auto status = AddACMFluxDerivative<C>(own, weight, leftFlux, diagonal); !status)
                return status.At(cell, sample.face);
            if (boundary)
                continue;
            const auto rightFlux = [&](const State &q, State &out)
            { return flux(own, q, out); };
            if (sample.other == cell)
            {
                if (const auto status = AddACMFluxDerivative<C>(remote, weight, rightFlux, diagonal); !status)
                    return status.At(cell, sample.face);
            }
            else
            {
                if (sample.jacobianColumn == UnInitIndex)
                    return NumericalStatus{NumericalError::InvalidInput}.At(cell, sample.face);
                auto block = DenseBlock<C>(request.offDiagonal(cell, sample.jacobianColumn), request.offDiagonal.rowMajorBlocks);
                if (const auto status = AddACMFluxDerivative<C>(remote, weight, rightFlux, block); !status)
                    return status.At(cell, sample.face);
            }
        }
        if constexpr (C > 4)
        {
            const auto point = Physics::Incompressible::RANSPoint(own, gradient, C - 4,
                                                                  request.physics.kinematicViscosity, request.wallMean(cell, 0), request.lengths(cell, 0));
            const auto source = Physics::RANS::TrySource(request.rans, point, 1);
            if (!source)
                return source.status.At(cell);
            for (int j = 4; j < C; ++j)
                diagonal(j, j) += request.residualScale * volume * source.value[j - 4];
        }
        for (int j = 0; j < C; ++j)
            for (int i = 0; i < C; ++i)
                if (!request.active[i] || !request.active[j])
                {
                    diagonal(i, j) = real(i == j);
                    for (index slot = 0; slot < request.offDiagonal.BlockNumber(cell); ++slot)
                        request.offDiagonal.Element(cell, slot, i, j) = 0;
                }
        for (int j = 0; j < C * C; ++j)
            if (!std::isfinite(request.diagonal(cell, j)))
                return NumericalStatus{NumericalError::InvalidState}.At(cell);
        return {};
    }
}
