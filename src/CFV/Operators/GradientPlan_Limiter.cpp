#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Detail/GradientKernels.hpp"
#include "CFV/Detail/CharacteristicKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    namespace
    {
        void ValidateTransform(const GradientFaceCharacteristicTransform &transform, R::Device device, index faces)
        {
            const auto valid = [&](const R::Array<real> &array)
            {
                return array.GetDevice() == device && array.Layout().Kind() == R::RowLayoutKind::Table &&
                       array.RowCount() == faces && array.Layout().Width() == transform.count * transform.count;
            };
            DNDS_check_throw_info(valid(transform.forward) && valid(transform.inverse),
                                  "gradient characteristic matrix shape or placement differs");
            const bool queued = transform.validation == CharacteristicValidation::Queued;
            DNDS_check_throw_info((queued || (transform.validation == CharacteristicValidation::Frozen && transform.forwardVersion != 0 && transform.forwardVersion == transform.forward.Version() && transform.inverseVersion == transform.inverse.Version())) &&
                                      transform.forwardIdentity == transform.forward.Identity() && transform.inverseIdentity == transform.inverse.Identity(),
                                  "gradient characteristic matrices changed; finalize them before limiting");
        }
    }
    GradientFaceCharacteristicTransform GradientPlan::BuildQueuedFaceTransform(index first, index count) const
    {
        _data->metrics.Validate();
        DNDS_check_throw_info(first >= 0 && count >= 0 && count <= 64,
                              "queued gradient characteristics require a valid component range");
        if (Device() == R::Device::Host())
        {
            auto result = BuildFaceTransform(first, count);
            result.validation = CharacteristicValidation::Queued;
            return result;
        }
        GradientFaceCharacteristicTransform result{first, count, {}, {}, _data->identity};
        result.validation = CharacteristicValidation::Queued;
        if (!count)
            return result;
        result.forward = R::Array<real>(R::RowLayout::Dense(_data->metrics.NumFaceProc(), count * count), Device());
        result.inverse = R::Array<real>(R::RowLayout::Dense(_data->metrics.NumFaceProc(), count * count), Device());
        result.forwardIdentity = result.forward.Identity();
        result.inverseIdentity = result.inverse.Identity();
        return result;
    }
    void GradientPlan::SubmitValidateFaceTransform(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                   const GradientFaceCharacteristicTransform &transform, GradientWorkspace &workspace) const
    {
        _data->metrics.Validate();
        ValidateWorkspace(workspace, workspace.components);
        DNDS_check_throw_info(Device().space == R::MemorySpace::CUDA && stream.Context().device == Device() && workspace.withLimiter &&
                                  transform.plan == _data->identity && transform.validation == CharacteristicValidation::Queued &&
                                  transform.first >= 0 && transform.first <= workspace.components && transform.count >= 0 &&
                                  transform.count <= 64 && transform.count <= workspace.components - transform.first,
                              "queued gradient characteristic publication belongs to another plan/range");
        if (!transform.count)
            return;
        ValidateTransform(transform, Device(), _data->metrics.NumFaceProc());
        DNDS_check_throw_info(workspace.deviceStatus.GetDevice() == Device() && workspace.deviceStatus.RowCount() == 1 &&
                                  workspace.deviceStatus.Layout().Kind() == R::RowLayoutKind::Table && workspace.deviceStatus.Layout().Width() == 1,
                              "gradient device validation status shape differs");
        for (const auto *array : {&transform.forward, &transform.inverse})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(array->Storage(), workspace.deviceStatus.Storage()), "characteristic matrices alias validation status");
#ifdef DNDS_USE_CUDA
        R::SubmitArrays(engine, stream, [first = transform.first, count = transform.count](R::ExecutionContext c, auto left, auto right, auto status)
                        { Detail::LaunchValidateCharacteristics(c, {left, right, first, count}, status); }, R::Read(transform.forward), R::Read(transform.inverse), R::ReadWrite(workspace.deviceStatus));
#else
        DNDS_check_throw_info(false, "queued CUDA characteristic validation unavailable");
#endif
    }
    void GradientPlan::CheckStatus(const GradientWorkspace &workspace) const
    {
        ValidateWorkspace(workspace, workspace.components);
        if (Device() == R::Device::Host() || !workspace.withLimiter)
            return;
        DNDS_check_throw_info(workspace.deviceStatus.GetDevice() == Device() && workspace.deviceStatus.RowCount() == 1 &&
                                  workspace.deviceStatus.Layout().Width() == 1,
                              "gradient device validation status shape differs");
        const auto host = workspace.deviceStatus.To(R::Device::Host());
        DNDS_check_throw_info(host.Read().View()(0, 0) == 0, "device gradient limiter validation failed; do not publish the queued result");
    }
    GradientFaceCharacteristicTransform GradientPlan::PlaceFaceTransform(const GradientFaceCharacteristicTransform &transform,
                                                                         R::ExecutionContext context) const
    {
        _data->metrics.Validate();
        DNDS_check_throw_info(context.device == Device() && transform.plan == _data->identity &&
                                  transform.first >= 0 && transform.count >= 0 && transform.count <= 64,
                              "gradient transform placement has incompatible origin/control");
        if (!transform.count)
            return transform;
        if (transform.forward.GetDevice() == Device())
        {
            ValidateTransform(transform, Device(), _data->metrics.NumFaceProc());
            return transform;
        }
        ValidateTransform(transform, R::Device::Host(), _data->metrics.NumFaceProc());
        auto placed = transform;
        placed.forward = transform.forward.To(Device(), context);
        placed.inverse = transform.inverse.To(Device(), context);
        placed.forwardIdentity = placed.forward.Identity();
        placed.inverseIdentity = placed.inverse.Identity();
        placed.forwardVersion = placed.forward.Version();
        placed.inverseVersion = placed.inverse.Version();
        return placed;
    }
    void GradientPlan::ValidateLimiter(const FVField &input, const FVField &output, GradientFaceCharacteristicTransformRef transform,
                                       ReconstructionRegion region, LimiterIndicatorRef indicator, real threshold, real centerWeight) const
    {
        _data->metrics.Validate();
        _data->partition.Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "unknown gradient limiter work region");
        DNDS_check_throw_info(std::isfinite(threshold) && std::isfinite(centerWeight) && centerWeight > 0, "invalid gradient limiter control");
        const index dim = _data->metrics.Dimension(), width = input.Width();
        DNDS_check_throw_info(width > 0 && width % dim == 0 && output.Width() == width, "gradient limiter field width differs");
        const index components = width / dim;
        _data->metrics.ValidateField(input, width);
        _data->metrics.ValidateField(output, width);
        for (const auto *array : {&input.Owned(), &input.Ghost()})
            DNDS_check_throw_info(output.Owned().Identity() != array->Identity() &&
                                      !Algebra::Detail::Overlaps(output.Owned().Storage(), array->Storage()),
                                  "gradient limiter output overlaps its input");
        const auto disjoint = [&](const R::Array<real> &array)
        {
            DNDS_check_throw_info(output.Owned().Identity() != array.Identity() &&
                                      !Algebra::Detail::Overlaps(output.Owned().Storage(), array.Storage()),
                                  "gradient limiter output overlaps an auxiliary input");
        };
        if (indicator)
        {
            const auto &array = indicator->get();
            DNDS_check_throw_info(array.GetDevice() == Device() && array.Layout().Kind() == R::RowLayoutKind::Table &&
                                      array.RowCount() == _data->metrics.NumCell() && array.Layout().Width() == 1,
                                  "gradient limiter indicator shape or placement differs");
            disjoint(array);
        }
        if (transform)
        {
            const auto &value = transform->get();
            DNDS_check_throw_info(value.plan == _data->identity && value.first >= 0 && value.first <= components &&
                                      value.count >= 0 && value.count <= 64 && value.count <= components - value.first,
                                  "invalid gradient characteristic transform range/plan");
            if (value.count)
            {
                ValidateTransform(value, Device(), _data->metrics.NumFaceProc());
                disjoint(value.forward);
                disjoint(value.inverse);
            }
        }
        if (Device().space == R::MemorySpace::CUDA)
            for (const auto *array : {&input.Owned(), &input.Ghost(), &output.Owned()})
                DNDS_check_throw_info(array->Layout().Kind() == R::RowLayoutKind::Table, "CUDA gradient limiter requires combined AoS fields");
    }
    void GradientPlan::LimitWBAP3(const FVField &input, FVField &output, const std::vector<index> &periodicVectorStarts,
                                  GradientFaceCharacteristicTransformRef transform, ReconstructionRegion region,
                                  LimiterIndicatorRef indicator, real threshold, real centerWeight) const
    {
        ValidateLimiter(input, output, transform, region, indicator, threshold, centerWeight);
        // Synchronous CPU convenience API preserves its input diagnostic.
        // Queued producers cannot be inspected on the controller. CPU leaves
        // assert this precondition; device validation reports through status.
        if (indicator && Device() == R::Device::Host())
        {
            const auto read = indicator->get().Read();
            const auto values = read.View<R::FixedRows<1>>();
            OMP::ForSetupRows(0, values.RowCount(), [&](index iCell)
                              { DNDS_check_throw_info(std::isfinite(values(iCell, 0)), "gradient limiter indicator must be finite"); });
        }
        auto workspace = BuildWorkspace(input.Width() / _data->metrics.Dimension(), periodicVectorStarts, true);
        R::EagerEngine engine;
        auto stream = engine.CreateCPUStream();
        if (Device() != R::Device::Host())
        {
#ifdef DNDS_USE_CUDA
            stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
#else
            DNDS_check_throw_info(false, "CUDA gradient limiter is not built");
#endif
        }
        LimitWBAP3(engine, stream, input, output, workspace, transform, region, indicator, threshold, centerWeight);
        engine.Close();
        CheckStatus(workspace);
    }
    void GradientPlan::LimitWBAP3(R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &input, FVField &output,
                                  GradientWorkspace &workspace, GradientFaceCharacteristicTransformRef transform, ReconstructionRegion region,
                                  LimiterIndicatorRef indicator, real threshold, real centerWeight) const
    {
        ValidateLimiter(input, output, transform, region, indicator, threshold, centerWeight);
        const index dimension = _data->metrics.Dimension();
        ValidateWorkspace(workspace, input.Width() / dimension);
        DNDS_check_throw_info(stream.Context().device == Device() && workspace.withLimiter &&
                                  workspace.limiterScratch.GetDevice() == Device() && workspace.limiterScratch.Layout().Kind() == R::RowLayoutKind::Table &&
                                  (Device() != R::Device::Host() || (workspace.limiterScratch.RowCount() > 0 && workspace.limiterScratch.Layout().Width() >= 6 * input.Width())),
                              "gradient limiter workspace/stream is incompatible");
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(Device() == R::Device::Host(), "CUDA gradient limiter is not built");
#endif
        const index first = transform ? transform->get().first : 0, count = transform ? transform->get().count : 0;
        const auto &forward = count ? transform->get().forward : workspace.empty;
        const auto &inverse = count ? transform->get().inverse : workspace.empty;
        const auto &activation = indicator ? indicator->get() : workspace.empty;
        if (Device().space == R::MemorySpace::CUDA)
            DNDS_check_throw_info(workspace.deviceStatus.GetDevice() == Device() && workspace.deviceStatus.RowCount() == 1 &&
                                      workspace.deviceStatus.Layout().Kind() == R::RowLayoutKind::Table && workspace.deviceStatus.Layout().Width() == 1,
                                  "gradient limiter device status shape differs");
        for (const auto *array : std::initializer_list<const R::Array<real> *>{&input.Owned(), &input.Ghost(), &output.Owned(), &forward, &inverse, &activation})
            DNDS_check_throw_info(workspace.limiterScratch.Identity() != array->Identity() &&
                                      !Algebra::Detail::Overlaps(workspace.limiterScratch.Storage(), array->Storage()),
                                  "gradient limiter scratch aliases an operand");
        auto geometry = std::make_shared<FiniteVolumeTopologyRead>(_data->metrics.AcquireTopologyRead());
        const auto &selected = _data->regionRows.Get(region);
        R::SubmitArrays(engine, stream, [geometry, shape = selected.shape, dimension, first, count, threshold, centerWeight, activated = bool(indicator)](R::ExecutionContext context, auto offsets, auto codes, auto parameters, auto rows, auto starts, auto owned, auto ghost, auto forward, auto inverse, auto activation, auto output, auto scratch, auto status)
                        {
                            const Detail::GradientLimiterView plan{offsets, geometry->View(), {codes, parameters}, int(dimension)};
                            const Runtime::RowSelection selection{shape, rows};
                            const Detail::GradientLimiterValues values{{owned, ghost}, forward, inverse, activation, first, count, threshold, centerWeight, activated};
                            if (context.device == R::Device::Host())
                                Detail::LimitGradient(plan, selection, {{starts.data, starts.RowCount()}}, values, output, scratch);
#ifdef DNDS_USE_CUDA
                            else
                            {
                                Detail::LaunchGradientLimiter(context, plan, selection, {starts.DynamicView()}, values, output, status);
                            }
#endif
                        },
                        R::ReadAs<R::FixedRows<1>>(_data->offsets), R::ReadAs<R::FixedRows<1>>(_data->metrics.FacePeriodicCodes()), R::ReadAs<R::FixedRows<1>>(_data->metrics.PeriodicParameters()), R::ReadAs<R::FixedRows<1>>(selected.indices), R::ReadAs<R::FixedRows<1>>(workspace.periodicVectorStarts), R::ReadAs<R::UniformTableRows>(input.Owned()), R::ReadAs<R::UniformTableRows>(region == ReconstructionRegion::Core ? workspace.empty : input.Ghost()), R::ReadAs<R::UniformTableRows>(forward), R::ReadAs<R::UniformTableRows>(inverse), R::ReadAs<R::UniformTableRows>(activation), R::WriteAs<R::UniformTableRows>(output.Owned()), R::WriteAs<R::UniformTableRows>(workspace.limiterScratch), R::ReadWrite(workspace.deviceStatus));
    }
}
