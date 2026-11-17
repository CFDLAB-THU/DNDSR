#pragma once
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Discretization/CPU/ModalPositivityKernels.hpp"
#include "Cauchy/Discretization/CUDA/ModalPositivityKernels.hpp"
#include "CFV/Math/TaylorBasis.hpp"
#include "DNDS/Core/OMP.hpp"
#include <charconv>
#include <cstdlib>
#include <cstring>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct FiniteVolume::PositivityWorkspace::State
    {
        std::shared_ptr<const Data> owner;
        EquationDofs prototype;
        ThermalPolicy thermal;
        ModalPositivityPolicy policy;
        FiniteVolumeStencilStorage::ReadAccess stencil;
        std::optional<CFV::ReconstructionBasisRead> hostBasis;
        std::optional<CFV::ReconstructionResidentRead> residentBasis;
        std::optional<R::Array<index>::ReadLease> volumeOffsets;
        std::optional<R::Array<real>::ReadLease> volumePoints, volumeBasis;
        ModalPositivityGeometry geometry;
        R::Array<index> degree;
        R::Array<real> beta;
        std::vector<CPU::ModalPositivityWorkerScratch> cpuWorkers;
        CUDA::ModalPositivityLaunch launch;
        struct DeviceStorage
        {
            R::Array<NumericalStatus> partialStatus, status;
            R::Array<index> partialCount, count;
            R::Array<real> partialMinimum, minimum;
        };
        std::optional<DeviceStorage> device;
        bool pending = false;
        index limited = 0;
        real minimum = 1;

        static index EnvironmentInteger(const char *name, index fallback)
        {
            const char *text = std::getenv(name);
            if (!text)
                return fallback;
            index value = 0;
            const char *last = text + std::strlen(text);
            const auto parsed = std::from_chars(text, last, value);
            DNDS_check_throw_info(parsed.ec == std::errc{} && parsed.ptr == last && value >= 0,
                                  std::string("invalid modal PP launch control: ") + name);
            return value;
        }
        State(std::shared_ptr<const Data> data, const EquationDofs &input,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
            : owner(std::move(data)), prototype(input.NumericAlias()),
              thermal(owner->equation, owner->settings.numerics.densityFloor, owner->settings.numerics.temperatureFloor,
                      owner->settings.numerics.sensibleEnergyFloor, owner->settings.numerics.pressureFloor,
                      workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}),
              stencil(owner->stencil.Read({owner->metrics.Device(), nullptr})),
              degree(R::RowLayout::Dense(owner->metrics.NumCell(), 1), owner->metrics.Device()),
              beta(degree.Layout(), owner->metrics.Device())
        {
            DNDS_check_throw_info(owner->quadrature.positivity && owner->reconstruction,
                                  "modal PP requires prepared cell-local geometry and a variational plan");
            volumeOffsets.emplace(owner->quadrature.positivity->volumeOffsets.Read({owner->metrics.Device(), nullptr}));
            volumePoints.emplace(owner->quadrature.positivity->volumePoints.Read({owner->metrics.Device(), nullptr}));
            volumeBasis.emplace(owner->quadrature.positivity->volumeBasis.Read({owner->metrics.Device(), nullptr}));
            const auto &numerics = owner->settings.numerics;
            policy.bounds = thermal.View();
            policy.components = int(input.ComponentNumber());
            policy.modes = owner->reconstruction->NRecDOF();
            policy.maxOrder = numerics.cfv.maxOrder;
            policy.betaPower = numerics.positivityBetaPower;
            policy.betaCutoff = numerics.positivityBetaCutoff;
            policy.meanBase = numerics.positivityCompressToMean;
            policy.clippedPolicy = numerics.positivityClippedReferencePolicy == "backtracking"
                                       ? ReconstructionClippedReferencePolicy::Backtracking
                                       : ReconstructionClippedReferencePolicy::FallbackToBase;
            for (int order = 1; order <= policy.maxOrder; ++order)
                policy.modeEnd[order] = CFV::Math::PolynomialSize(owner->metrics.Dimension(), order) - 1;
            if (numerics.positivityReconstruction == "Segment")
            {
                DNDS_check_throw_info(!owner->equation.HasMixture() && !owner->equation.HasRANS() && input.ComponentNumber() == 5,
                                      "Segment reconstruction PP is a flow5 compatibility policy");
                policy.segment = true;
                policy.segmentBounds = owner->FixedFlowThermalBounds();
            }
            if (owner->equation.HasRANS())
            {
                policy.rans = ResolveRANSReconstructionPolicy(owner->equation);
                index first = 0;
                for (const auto &block : owner->equation.Schema().blocks)
                {
                    if (block.name == "rans")
                        policy.ransFirst = first;
                    first += index(block.components.size());
                }
                DNDS_assert(policy.ransFirst >= 0);
            }
            geometry.stencil = stencil.KernelDescriptor();
            geometry.volumeOffsets = volumeOffsets->View<R::FixedRows<1>>();
            geometry.volumePoints = volumePoints->View<R::FixedRows<3>>();
            geometry.volumeBasis = volumeBasis->View<R::TableRows>();
            geometry.volumeCachedDerivatives = owner->quadrature.positivity->cachedDerivatives;
            geometry.maximumCellPoints = owner->quadrature.positivity->maximumCellPoints;
            const auto deviceLocation = owner->metrics.Device();
            if (deviceLocation == R::Device::Host())
            {
                hostBasis.emplace(owner->reconstruction->AcquireBasisRead());
                const auto &view = hostBasis->EvaluationView();
                geometry.basis = {view.cached, view.moments, view.geometry.cells,
                                  CFV::Math::TaylorBasis(view.dimension, view.degree).KernelParameters(), view.cachedDerivatives, 1};
                const index threads = std::min<index>(OMP::DistributedThreads(), std::max<index>(1, owner->metrics.NumCell()));
                for (index iWorker = 0; iWorker < threads; ++iWorker)
                    cpuWorkers.emplace_back(policy.components, policy.modes, geometry.maximumCellPoints, policy.meanBase);
            }
            else
            {
#ifdef DNDS_USE_CUDA
                DNDS_check_throw_info(input.PhysicalFieldNumber() == 1 && !owner->equation.HasMixture() &&
                                          input.ComponentNumber() >= 5 && input.ComponentNumber() <= 7,
                                      "CUDA modal PP requires combined perfect-gas flow5/SA/2EQ fields");
                residentBasis.emplace(owner->reconstruction->AcquireResidentRead());
                geometry.basis = residentBasis->PointBasisView();
                if (const char *mapping = std::getenv("DNDS_CAUCHY_CUDA_MODAL_PP_MAPPING"))
                {
                    const std::string value(mapping);
                    DNDS_check_throw_info(value == "warpPerCell" || value == "blockPerCell", "unknown modal PP CUDA mapping");
                    launch.mapping = value == "blockPerCell" ? CUDA::ModalPositivityMapping::BlockPerCell : CUDA::ModalPositivityMapping::WarpPerCell;
                }
                const index threads = EnvironmentInteger("DNDS_CAUCHY_CUDA_MODAL_PP_THREADS", owner->settings.execution.kernels.polynomialPPThreads ? owner->settings.execution.kernels.polynomialPPThreads : 128);
                DNDS_check_throw_info(threads <= INT_MAX, "modal PP block size overflows int");
                launch.threads = int(threads);
                launch.blocks = EnvironmentInteger("DNDS_CAUCHY_CUDA_MODAL_PP_BLOCKS", 0);
                CUDA::PrepareModalPositivityLaunch(deviceLocation, policy, geometry.maximumCellPoints, launch);
                const index partials = CUDA::ModalPositivitySummaryRows(owner->metrics.NumCell(), launch);
                device.emplace(DeviceStorage{
                    R::Array<NumericalStatus>(R::RowLayout::Dense(partials, 1), deviceLocation),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), deviceLocation),
                    R::Array<index>(R::RowLayout::Dense(partials, 1), deviceLocation),
                    R::Array<index>(R::RowLayout::Dense(1, 1), deviceLocation),
                    R::Array<real>(R::RowLayout::Dense(partials, 1), deviceLocation),
                    R::Array<real>(R::RowLayout::Dense(1, 1), deviceLocation)});
                device->partialStatus.SetName("Cauchy/modalPP/statusPartials", DNDS_ALLOCATION_SITE());
                device->status.SetName("Cauchy/modalPP/statusResult", DNDS_ALLOCATION_SITE());
                device->partialCount.SetName("Cauchy/modalPP/countPartials", DNDS_ALLOCATION_SITE());
                device->count.SetName("Cauchy/modalPP/count", DNDS_ALLOCATION_SITE());
                device->partialMinimum.SetName("Cauchy/modalPP/minimumPartials", DNDS_ALLOCATION_SITE());
                device->minimum.SetName("Cauchy/modalPP/minimum", DNDS_ALLOCATION_SITE());
#else
                DNDS_check_throw_info(false, "CUDA modal PP is unavailable in this build");
#endif
            }
            geometry.ValidateDevice(deviceLocation);
            degree.SetName("Cauchy/positivity/degree", DNDS_ALLOCATION_SITE());
            beta.SetName("Cauchy/positivity/beta", DNDS_ALLOCATION_SITE());
        }
    };
}
