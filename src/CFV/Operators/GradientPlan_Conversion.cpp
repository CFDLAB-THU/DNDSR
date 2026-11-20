#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Detail/GradientKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    void GradientPlan::ValidateConversion(const FVField &source, const FVField &destination,
                                          const VariationalReconstructionPlan &reconstruction, bool toCoefficients, bool ghost) const
    {
        const index dimension = _data->metrics.Dimension(), modes = reconstruction.NRecDOF();
        const index gradientWidth = toCoefficients ? source.Width() : destination.Width();
        const index coefficientWidth = toCoefficients ? destination.Width() : source.Width();
        DNDS_check_throw_info(gradientWidth > 0 && gradientWidth % dimension == 0 && modes >= dimension &&
                                  coefficientWidth == (gradientWidth / dimension) * modes && reconstruction.Space().Same(_data->metrics.Space()) &&
                                  (!toCoefficients || !ghost),
                              "linear gradient conversion has incompatible shape or row space");
        _data->metrics.ValidateField(source, source.Width());
        _data->metrics.ValidateField(destination, destination.Width());
        const auto &input = ghost ? source.Ghost() : source.Owned();
        const auto &output = ghost ? destination.Ghost() : destination.Owned();
        DNDS_check_throw_info(input.Identity() != output.Identity() && !Algebra::Detail::Overlaps(input.Storage(), output.Storage()),
                              "linear gradient conversion output overlaps input");
        if (Device().space == R::MemorySpace::CUDA)
            DNDS_check_throw_info(input.Layout().Kind() == R::RowLayoutKind::Table && output.Layout().Kind() == R::RowLayoutKind::Table,
                                  "CUDA linear gradient conversion requires AoS tables");
    }
    void GradientPlan::ConvertFields(const FVField &source, FVField &destination, const VariationalReconstructionPlan &reconstruction,
                                     bool toCoefficients, bool ghost) const
    {
        ValidateConversion(source, destination, reconstruction, toCoefficients, ghost);
        if (Device() != R::Device::Host())
        {
#ifndef DNDS_USE_CUDA
            DNDS_check_throw_info(false, "CUDA gradient conversion is not built");
#else
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            ConvertFields(engine, stream, source, destination, reconstruction, toCoefficients, ghost);
            engine.Close();
#endif
            return;
        }
        const auto geometry = _data->metrics.AcquireRead();
        const auto input = (ghost ? source.Ghost() : source.Owned()).Read();
        auto output = (ghost ? destination.Ghost() : destination.Owned()).Write();
        Detail::ConvertGradient(geometry.View().cells, input.View<R::UniformTableRows>(), output.View<R::UniformTableRows>(),
                                ghost ? _data->metrics.NumCell() : 0, _data->metrics.Dimension(), reconstruction.NRecDOF(), toCoefficients);
    }
    void GradientPlan::ConvertFields(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                     const FVField &source, FVField &destination, const VariationalReconstructionPlan &reconstruction,
                                     bool toCoefficients, bool ghost, index firstComponent, index componentCount) const
    {
        ValidateConversion(source, destination, reconstruction, toCoefficients, ghost);
        const index components = (toCoefficients ? source.Width() : destination.Width()) / _data->metrics.Dimension();
        if (componentCount == -1)
            componentCount = components;
        DNDS_check_throw_info(firstComponent >= 0 && firstComponent <= components && componentCount >= 0 && componentCount <= components - firstComponent, "gradient conversion component range");
        DNDS_check_throw_info(stream.Context().device == Device(), "gradient conversion stream placement differs");
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(Device() == R::Device::Host(), "CUDA gradient conversion is not built");
#endif
        auto geometry = std::make_shared<Geom::MeshReadAccess>(_data->metrics.AcquireMeshRead());
        const int dimension = _data->metrics.Dimension();
        const index firstCell = ghost ? _data->metrics.NumCell() : 0, modes = reconstruction.NRecDOF();
        auto &rows = ghost ? destination.Ghost() : destination.Owned();
        const auto outputOperand = componentCount == components ? R::WriteAs<R::UniformTableRows>(rows)
                                                                : R::ReadWriteAs<R::UniformTableRows>(rows);
        R::SubmitArrays(engine, stream, [geometry, dimension, firstCell, modes, toCoefficients, firstComponent, componentCount](R::ExecutionContext context, auto cells, auto input, auto output)
                        {
                            if (context.device == R::Device::Host())
                                Detail::ConvertGradient(cells, input, output, firstCell, dimension, modes, toCoefficients, firstComponent, componentCount);
#ifdef DNDS_USE_CUDA
                            else
                                Detail::LaunchGradientConversion(context, cells, input, output, firstCell, dimension, modes, toCoefficients, firstComponent, componentCount);
#endif
                        },
                        R::ReadAs<R::FixedRows<8>>(_data->metrics.CellMetrics()), R::ReadAs<R::UniformTableRows>(ghost ? source.Ghost() : source.Owned()), outputOperand);
    }
    void GradientPlan::ConvertToCoefficients(const FVField &gradient, const VariationalReconstructionPlan &reconstruction, FVField &coefficients) const
    {
        ConvertFields(gradient, coefficients, reconstruction, true, false);
    }
    void GradientPlan::ConvertToCoefficients(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                             const FVField &gradient, const VariationalReconstructionPlan &reconstruction, FVField &coefficients) const
    {
        ConvertFields(engine, stream, gradient, coefficients, reconstruction, true, false);
    }
    void GradientPlan::SetLinearCoefficients(R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &gradient,
                                             const VariationalReconstructionPlan &plan, FVField &coefficients,
                                             index firstComponent, index componentCount) const
    {
        ConvertFields(engine, stream, gradient, coefficients, plan, true, false, firstComponent, componentCount);
    }
    void GradientPlan::ExtractLinearGradients(const FVField &coefficients, const VariationalReconstructionPlan &reconstruction, FVField &gradient, bool ghost) const
    {
        ConvertFields(coefficients, gradient, reconstruction, false, ghost);
    }
    void GradientPlan::ExtractLinearGradients(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                              const FVField &coefficients, const VariationalReconstructionPlan &reconstruction, FVField &gradient, bool ghost) const
    {
        ConvertFields(engine, stream, coefficients, gradient, reconstruction, false, ghost);
    }
}
