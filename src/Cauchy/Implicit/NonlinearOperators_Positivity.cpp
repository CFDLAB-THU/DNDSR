#include "Cauchy/Implicit/NonlinearOperators_Internal.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace NonlinearOperatorDetail;
    AlphaStatistics NonlinearOperators::PreparePseudoTime(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                          const EquationDofs &mean, EquationDofs &defect, const R::Array<real> &spectral,
                                                          R::Array<real> &inverse, OptionalRef<R::Array<real>> alpha, NonlinearPseudoTimeControls controls) const
    {
        _data->Check(engine, stream, mean);
        _data->CheckPair(mean, defect);
        _data->CheckScalar(spectral);
        _data->CheckScalar(inverse);
        if (alpha)
            _data->CheckScalar(*alpha);
        DNDS_check_throw_info(!alpha || controls.positivity, "deferred nonlinear alpha requires positivity enabled");
        DNDS_check_throw_info(std::isfinite(controls.cfl) && controls.cfl > 0 && std::isfinite(controls.forecastScale) && controls.forecastScale >= 0 &&
                                  std::isfinite(controls.relax) && controls.relax >= 0 && controls.relax <= 1 && controls.compressionPower > 0,
                              "invalid nonlinear pseudo-time controls");
        const auto &smoothScale = _data->geometry.CellSmoothScaleFactors();
        if (mean.Schema().AccumulationKind() == PhysicalAccumulationKind::ConstantDensityVelocityPressure)
        {
            DNDS_check_throw_info(!controls.positivity && !alpha, "ACM pseudo-time uses uncompressed spectral steps");
            const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
            R::SubmitArrays(engine, stream, [geometry, cfl = controls.cfl](R::ExecutionContext context, auto spectrum, auto scale, auto step)
                            {
                if (context.device == R::Device::Host())
                    CPU::PrepareSpectralPseudoTime(geometry->HostView(), spectrum, scale, step, cfl);
                else
                {
#ifdef DNDS_USE_CUDA
                    CUDA::PrepareSpectralPseudoTime(context, geometry->View().cells, Table(spectrum), Table(scale), Table(step), cfl);
#else
                    Unavailable{}();
#endif
                } }, R::ReadAs<R::UniformTableRows>(spectral), R::ReadAs<R::UniformTableRows>(smoothScale), R::WriteAs<R::UniformTableRows>(inverse));
            // CPU assembly takes immediate read access. Device assembly imports
            // this lane order and validates the inverse step in its own kernel.
            if (stream.Context().device == R::Device::Host())
                engine.Synchronize();
            return {};
        }
        Inputs inputs;
        Outputs outputs;
        const auto u = AddInput(mean, inputs), d = AddOutput(defect, outputs, true);
        const index spectralSlot = index(inputs.size()), inverseSlot = index(outputs.size());
        inputs.push_back(R::ReadAs<R::UniformTableRows>(spectral));
        const index scaleSlot = index(inputs.size());
        inputs.push_back(R::ReadAs<R::UniformTableRows>(smoothScale));
        outputs.push_back(R::WriteAs<R::UniformTableRows>(inverse));
        const index alphaSlot = alpha ? index(outputs.size()) : -1;
        if (alpha)
            outputs.push_back(R::WriteAs<R::UniformTableRows>(*alpha));
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto retained = _data;
        const auto host = [retained, geometry, u, d, spectralSlot, scaleSlot, inverseSlot, alphaSlot, controls](auto input, auto output, Result &result)
        {
            const auto mean = u.Components<const real>(input);
            const auto defect = d.Components<real>(output);
            const auto statistics = CPU::PreparePseudoTime({geometry->HostView(), mean.View(), defect.View(), input[spectralSlot], input[scaleSlot], output[inverseSlot],
                                                            alphaSlot >= 0 ? std::optional(output[alphaSlot]) : std::nullopt, retained->policy.View().thermal,
                                                            controls.cfl, controls.forecastScale, controls.relax, real(controls.compressionPower), controls.positivity, controls.compressRHS});
            result.status = statistics.status;
            if (statistics)
                result.alpha = statistics.value;
        };
#ifdef DNDS_USE_CUDA
        const auto device = [retained, geometry, u, d, spectralSlot, scaleSlot, inverseSlot, alphaSlot, controls](auto context, auto input, auto output, auto scratch, auto summary)
        {
            const CUDA::PseudoTimeParameters parameters{retained->policy.View().thermal, controls.cfl, controls.forecastScale, controls.relax,
                                                        controls.compressionPower, controls.positivity, controls.compressRHS, alphaSlot >= 0};
            CUDA::PreparePseudoTime(context, geometry->View().cells, Table(input[u.firstSlot]), Table(output[d.firstSlot]), Table(input[spectralSlot]), Table(input[scaleSlot]),
                                    Table(output[inverseSlot]), alphaSlot >= 0 ? Table(output[alphaSlot]) : CUDA::LinearWriteRows{}, parameters, scratch, summary);
        };
#else
        const Unavailable device;
#endif
        return _data->Reduce(engine, stream, mean.ComponentNumber(), std::move(inputs), std::move(outputs), host, device).alpha;
    }
    void NonlinearOperators::SubmitAgreedRHSAlpha(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                  const R::Array<real> &alpha, R::Array<real> &inverse, EquationDofs &defect, real minimum, AlphaReduction reduction, bool compressRHS) const
    {
        _data->Check(engine, stream, defect);
        _data->CheckScalar(alpha);
        _data->CheckScalar(inverse);
        DNDS_check_throw_info(std::isfinite(minimum) && minimum >= 0 && minimum <= 1 && int(reduction) >= 0 && int(reduction) <= 2,
                              "invalid agreed nonlinear alpha policy");
        Inputs inputs{R::ReadAs<R::UniformTableRows>(alpha)};
        Outputs outputs{R::ReadWriteAs<R::UniformTableRows>(inverse)};
        const auto d = AddOutput(defect, outputs, true);
        R::SubmitArrayBatch(engine, stream, [d, minimum, reduction, compressRHS](R::ExecutionContext context, auto input, auto output)
                            {
            if (context.device == R::Device::Host())
            {
                const auto defect = d.Components<real>(output);
                CPU::ApplyAgreedRHSAlpha(input[0], output[0], defect.View(), minimum, reduction, compressRHS);
            }
            else
            {
#ifdef DNDS_USE_CUDA
                CUDA::ApplyAgreedRHSAlpha(context, Table(input[0]), Table(output[0]), Table(output[d.firstSlot]), minimum, reduction, compressRHS);
#else
                Unavailable{}();
#endif
            } }, std::move(inputs), std::move(outputs));
    }
    NonlinearCorrectionResult NonlinearOperators::ApplyCorrection(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                                  const EquationDofs &mean, const EquationDofs &increment, EquationDofs &candidate,
                                                                  SpanView<const uint8_t> active, NonlinearCorrectionControls controls) const
    {
        _data->Check(engine, stream, mean);
        _data->Check(engine, stream, increment);
        _data->CheckPair(mean, candidate);
        _data->CheckPair(increment, candidate);
        DNDS_check_throw_info(std::isfinite(controls.scale) && controls.scale > 0 && controls.scale <= 1 &&
                                  std::isfinite(controls.relax) && controls.relax >= 0 && controls.relax <= 1 && controls.compressionPower > 0,
                              "invalid nonlinear increment controls");
        Inputs inputs;
        Outputs outputs;
        const auto u = AddInput(mean, inputs), d = AddInput(increment, inputs), y = AddOutput(candidate, outputs);
        const auto selection = Selection(active, mean.ComponentNumber());
        const auto retained = _data;
        struct HostCorrectionScratch
        {
            std::vector<real> species;
            std::vector<CPU::CorrectionResult> partials;
        };
        std::shared_ptr<HostCorrectionScratch> hostScratch;
        if (stream.Context().device == R::Device::Host())
        {
            hostScratch = std::make_shared<HostCorrectionScratch>();
            const index partitions = std::min<index>(OMP::DistributedThreads(), std::max<index>(1, mean.Space().OwnedRows()));
            hostScratch->species.resize(size_t(partitions * retained->independent));
            hostScratch->partials.resize(size_t(partitions));
        }
        const auto host = [retained, u, d, y, selection, controls, hostScratch](auto input, auto output, Result &result)
        {
            const auto mean = u.Components<const real>(input), increment = d.Components<const real>(input);
            const auto candidate = y.Components<real>(output);
            const auto view = retained->policy.View();
            const auto correction = CPU::ApplyCorrection({view.thermal, view.rans, selection.HostView(),
                                                          index(u.columns.size()), retained->flowFirst, retained->speciesFirst, retained->ransFirst, int(retained->independent),
                                                          controls.scale, controls.relax, real(controls.compressionPower), controls.positivity, controls.ransRelaxation},
                                                         mean.View(), increment.View(), candidate.View(),
                                                         {{hostScratch->species.data(), index(hostScratch->species.size())},
                                                          {hostScratch->partials.data(), index(hostScratch->partials.size())}});
            result.status = correction.status;
            if (correction)
            {
                result.alpha = {correction.value.limited, correction.value.minimum};
                result.frozen = correction.value.frozen;
            }
        };
#ifdef DNDS_USE_CUDA
        const auto device = [retained, u, d, y, selection, controls](auto context, auto input, auto output, auto scratch, auto summary)
        {
            const CUDA::CorrectionParameters parameters{retained->policy.View(), DeviceSelection(selection),
                                                        controls.scale, controls.relax, controls.compressionPower, controls.positivity, controls.ransRelaxation};
            CUDA::ApplyCorrection(context, Table(input[u.firstSlot]), Table(input[d.firstSlot]), Table(output[y.firstSlot]), parameters, scratch, summary);
        };
#else
        const Unavailable device;
#endif
        const auto result = _data->Reduce(engine, stream, mean.ComponentNumber(), std::move(inputs), std::move(outputs), host, device);
        return {result.alpha, result.frozen};
    }
}
