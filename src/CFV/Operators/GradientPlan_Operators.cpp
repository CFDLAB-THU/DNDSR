#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "CFV/Detail/GradientKernels.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"

namespace DNDS::CFV
{
    index GradientPlan::NumBoundaryQuadrature() const
    {
        return _data->boundaryCount;
    }

    BoundaryQuadraturePoint GradientPlan::GetBoundaryQuadrature(index iPoint) const
    {
        _data->metrics.Validate();
        DNDS_check_throw_info(Device() == Runtime::Device::Host(), "gradient boundary coordinates are CPU setup data");
        DNDS_check_throw_info(iPoint >= 0 && iPoint < NumBoundaryQuadrature(), "gradient boundary point out of range");
        return _data->boundary[iPoint];
    }

    Runtime::Array<real> GradientPlan::BuildBoundaryValues(index components) const
    {
        DNDS_check_throw_info(components > 0, "gradient boundary components must be positive");
        return Runtime::Array<real>(Runtime::RowLayout::Dense(NumBoundaryQuadrature(), components), Device());
    }

    GradientFaceCharacteristicTransform GradientPlan::BuildFaceTransform(index first, index count) const
    {
        namespace R = Runtime;
        _data->metrics.Validate();
        DNDS_check_throw_info(first >= 0 && count >= 0 && count <= 64,
                              "invalid gradient characteristic component range");
        if (count == 0)
            return {first, 0, {}, {}, _data->identity};
        DNDS_check_throw_info(Device() == R::Device::Host(), "build gradient characteristics on CPU, then place the finalized transform");
        GradientFaceCharacteristicTransform result{
            first,
            count,
            R::Array<real>(R::RowLayout::Dense(_data->metrics.NumFaceProc(), count * count)),
            R::Array<real>(R::RowLayout::Dense(_data->metrics.NumFaceProc(), count * count)),
            _data->identity};
        for (auto *array : {&result.forward, &result.inverse})
        {
            auto write = array->Write();
            const auto view = write.View<R::TableRows>();
            for (index iFace = 0; iFace < view.RowCount(); ++iFace)
                Eigen::Map<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>>(view[iFace].data(), count, count).setIdentity();
        }
        FinalizeFaceTransform(result);
        return result;
    }

    void GradientPlan::FinalizeFaceTransform(GradientFaceCharacteristicTransform &transform, bool checkInverses) const
    {
        namespace R = Runtime;
        _data->metrics.Validate();
        transform.forwardVersion = transform.inverseVersion = 0;
        DNDS_check_throw_info(transform.validation == CharacteristicValidation::Frozen, "queued characteristics do not use host version finalization");
        transform.forwardIdentity = transform.inverseIdentity = 0;
        DNDS_check_throw_info(transform.plan == _data->identity && transform.first >= 0 &&
                                  transform.count >= 0 && transform.count <= 64,
                              "gradient characteristic transform belongs to another plan or has an invalid range");
        const index count = transform.count;
        if (count == 0)
            return;
        const auto valid = [&](const R::Array<real> &array)
        {
            return array.GetDevice() == R::Device::Host() && array.Layout().Kind() == R::RowLayoutKind::Table &&
                   array.RowCount() == _data->metrics.NumFaceProc() && array.Layout().Width() == count * count;
        };
        DNDS_check_throw_info(valid(transform.forward) && valid(transform.inverse),
                              "gradient characteristic matrix shape or placement differs");
        if (checkInverses)
        {
            const auto forward = transform.forward.Read(), inverse = transform.inverse.Read();
            const auto left = forward.View<R::TableRows>(), right = inverse.View<R::TableRows>();
            Eigen::MatrixXd product(count, count);
            for (index iFace = 0; iFace < _data->metrics.NumFaceProc(); ++iFace)
            {
                const Eigen::Map<const Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>> L(left[iFace].data(), count, count);
                const Eigen::Map<const Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>> Rm(right[iFace].data(), count, count);
                product.noalias() = L * Rm;
                DNDS_check_throw_info(L.allFinite() && Rm.allFinite() &&
                                          product.isApprox(Eigen::MatrixXd::Identity(count, count), 1e-10),
                                      "gradient characteristic matrices must be finite mutual inverses");
            }
        }
        transform.forwardVersion = transform.forward.Version();
        transform.inverseVersion = transform.inverse.Version();
        transform.forwardIdentity = transform.forward.Identity();
        transform.inverseIdentity = transform.inverse.Identity();
    }

    void GradientPlan::Evaluate(const FVField &mean, const Runtime::Array<real> &boundaryValues, FVField &gradient,
                                const std::vector<index> &periodicVectorStarts, ReconstructionRegion region) const
    {
        namespace R = Runtime;
        if (Device() != R::Device::Host())
        {
#ifndef DNDS_USE_CUDA
            DNDS_check_throw_info(false, "CUDA gradients are not built");
#else
            auto workspace = BuildWorkspace(mean.Width(), periodicVectorStarts);
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            Evaluate(engine, stream, mean, boundaryValues, gradient, workspace, region);
            engine.Close();
#endif
            return;
        }
        auto geometry = _data->metrics.AcquireRead();
        ValidateEvaluation(mean, boundaryValues, gradient, region);
        Detail::ValidateVectorBlocks(mean.Width(), periodicVectorStarts);
        auto owned = mean.Owned().Read(), boundary = boundaryValues.Read();
        std::optional<R::Array<real>::ReadLease> ghost;
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{owned.View<R::UniformTableRows>(), {}};
        if (region != ReconstructionRegion::Core)
        {
            ghost.emplace(mean.Ghost().Read());
            u.ghost = ghost->View<R::UniformTableRows>();
        }
        auto outWrite = gradient.Owned().Write();
        auto offsets = _data->offsets.Read(), boundaryIds = _data->boundaryIds.Read();
        auto weights = _data->weights.Read();
        auto codes = _data->metrics.FacePeriodicCodes().Read();
        auto parameters = _data->metrics.PeriodicParameters().Read();
        const auto topology = _data->metrics.AcquireTopologyRead();
        const auto offset = offsets.View<R::FixedRows<1>>(), bnd = boundaryIds.View<R::FixedRows<1>>();
        const auto w = weights.View<R::FixedRows<3>>();
        const auto output = outWrite.View<R::UniformTableRows>();
        auto selectionRead = (region == ReconstructionRegion::HaloDependent ? _data->dependent : _data->core).Read();
        const auto selectionRows = selectionRead.View<R::FixedRows<1>>();
        const Detail::DeviceWorkSelection selection{selectionRows.DynamicView(),
                                                    region == ReconstructionRegion::All ? _data->metrics.NumCell() : selectionRows.RowCount(), region != ReconstructionRegion::All};
        const Detail::GradientKernelView plan{offset, bnd, w, topology.View(), {codes.View<R::FixedRows<1>, false>(), parameters.View<R::FixedRows<1>, false>()}, _data->metrics.Dimension()};
        Detail::EvaluateGradient(plan, selection, {{periodicVectorStarts.data(), index(periodicVectorStarts.size())}}, u, boundary.View<R::UniformTableRows>(), output);
    }

}
