#include "Cauchy/Implicit/CPU/FiniteVolumeJacobianKernels.hpp"
namespace DNDS::Cauchy::CPU
{
    void EvaluateEquationJacobianBlock(const EquationJacobianView &snapshot, index iRow, index iBlock, Eigen::Ref<Eigen::MatrixXd> output)
    {
        output.setZero();
        for (const auto &record : snapshot.geometry.Points(iRow, iBlock))
            AddEquationJacobianPoint(snapshot, ExpandJacobianPoint(snapshot.geometry.Point(iRow, record)), output, snapshot.means);
    }
    void ApplyEquationJacobianBlock(const EquationJacobianView &snapshot, index iRow, index iBlock, JacobianInput input, Eigen::Ref<Eigen::VectorXd> output)
    {
        // Geometry classification and component selection are frozen at
        // preparation, not rediscovered inside each component product.
        // Input retains SoA/AoSoA stride; only the immutable snapshot's
        // mean/snapshot.pressure columns and caller scratch are contiguous.
        if (snapshot.sixComponentProduct && snapshot.sameComponentFrames)
        {
            const FrozenFluxDirection6 direction{input.data(), 6, input.innerStride()};
            const FrozenFluxProduct6 product{output.data(), 6};
            for (index j = 0; j < 6; ++j)
                product[j] = 0;
            for (const auto &record : snapshot.geometry.Points(iRow, iBlock))
            {
                const auto point = ExpandJacobianPoint(snapshot.geometry.Point(iRow, record));
                AddFrozenFluxProduct6({snapshot.means[point.other].data(), 6},
                                      {snapshot.pressureDerivative[point.other].data(), 6}, snapshot.pressure(point.other, 0),
                                      point.normal, snapshot.frame.GridVelocity(point.location), .5 * snapshot.scale * point.weight,
                                      snapshot.radius(point.auxiliary, 0) + snapshot.radius(point.auxiliary, 1), direction, product);
            }
            return;
        }
        output.setZero();
        for (const auto &record : snapshot.geometry.Points(iRow, iBlock))
        {
            const auto point = ExpandJacobianPoint(snapshot.geometry.Point(iRow, record));
            FlowState flow{}, direction{};
            real dp = 0;
            for (index j = 0; j < snapshot.components; ++j)
                if (snapshot.active[j])
                    dp += snapshot.pressureDerivative(point.other, j) * input[j];
            for (int j = 0; j < 5; ++j)
            {
                flow[j] = snapshot.means(point.other, snapshot.flowFirst + j);
                direction[j] = snapshot.active[snapshot.flowFirst + j] ? input[snapshot.flowFirst + j] : 0;
            }
            if (point.changesFrame)
            {
                flow = FiniteVolumeDetail::RotateState(flow, point.rotation);
                direction = FiniteVolumeDetail::RotateState(direction, point.rotation);
            }
            const auto grid = snapshot.frame.GridVelocity(point.location);
            auto derivative = PrepareCompressibleFluxDerivative(flow, direction, snapshot.pressure(point.other, 0), dp, point.normal, grid);
            // The former isotropic 2*rho*k/3 stress and work are absent from the residual.
            real speed = 0, deltaSpeed = 0;
            for (int d = 0; d < 3; ++d)
            {
                const real velocity = flow[1 + d] / flow[0];
                speed += (velocity - grid[d]) * point.normal[d];
                deltaSpeed += (direction[1 + d] - velocity * direction[0]) / flow[0] * point.normal[d];
            }
            const real factor = .5 * snapshot.scale * point.weight;
            const real lambda = snapshot.radius(point.auxiliary, 0) + snapshot.radius(point.auxiliary, 1);
            for (index j = 0; j < snapshot.components; ++j)
                if (snapshot.active[j])
                {
                    const auto k = j - snapshot.flowFirst;
                    const real value = k >= 0 && k < 5 ? derivative[k] - lambda * direction[k]
                                                       : PrepareAdvectedScalarFluxDerivative(snapshot.means(point.other, j), input[j], speed, deltaSpeed) - lambda * input[j];
                    output[j] += factor * value;
                }
        }
    }
}
