#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include <optional>
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    void RHSGeometry::ValidateDevice(R::Device device) const
    {
        R::ValidateViewDevices(device, stencil, faceOffsets, cells, quadrature);
        if (acm.RowCount())
            R::ValidateViewDevices(device, acm, acmBC);
        else
            conditions.ValidateDevice(device);
        if (hasBasis)
            basis.ValidateDevice(device);
        if (hasWall)
            R::ValidateViewDevices(device, wallMean, wallFaces);
    }
    struct RHSGeometryPublication::Data
    {
        RHSGeometry host;
        R::Array<RHSGeometry> storage;
        std::optional<R::Array<RHSGeometry>::ReadLease> read;
        int multiprocessors = 0;
        Data(RHSGeometry geometry, R::ExecutionContext context) : host(geometry)
        {
            host.ValidateDevice(context.device);
#ifdef DNDS_USE_CUDA
            cudaDeviceProp properties;
            DNDS_check_throw_info(cudaGetDeviceProperties(&properties, context.device.ordinal) == cudaSuccess, "RHS placement device query failed");
            multiprocessors = properties.multiProcessorCount;
#endif
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && host.stencil.jacobianOffsets.RowCount() > 0 &&
                                      host.stencil.jacobianOffsets.RowCount() - 1 <= host.cells.RowCount(),
                                  "invalid CUDA RHS geometry publication");
            R::Array<RHSGeometry> staging(R::RowLayout::Dense(1, 1));
            {
                auto write = staging.Write();
                write.View<R::FixedRows<1>>()(0, 0) = host;
            }
            storage = staging.To(context.device, context);
            storage.SetName("Cauchy/RHSGeometryArguments", DNDS_ALLOCATION_SITE());
            read.emplace(storage.Read(context));
        }
    };
    RHSGeometryPublication::RHSGeometryPublication(RHSGeometry geometry, R::ExecutionContext context)
        : _data(std::make_shared<Data>(geometry, context)) {}
    const RHSGeometry &RHSGeometryPublication::HostDescriptor() const { return _data->host; }
    R::ArrayView<const RHSGeometry, R::AccessBoundsChecks, R::FixedRows<1>> RHSGeometryPublication::View() const
    {
        return _data->read->View<R::FixedRows<1>>();
    }
    R::Device RHSGeometryPublication::Device() const { return _data->storage.GetDevice(); }
    int RHSGeometryPublication::MultiprocessorCount() const { return _data->multiprocessors; }
    void RHSKernelParameters::Validate(int components) const
    {
        physics.Validate(components);
        DNDS_check_throw_info(thermal.flowFirst == 0 && thermal.components == components && thermal.independent == 0 &&
                                  thermal.speciesEnergies[0].empty() && thermal.speciesEnergies[1].empty(),
                              "invalid perfect-gas RHS policy");
    }
    void RHSKernelControls::Validate() const
    {
        DNDS_check_throw_info(std::isfinite(time) && components >= (mechanical ? 4 : 5) && components <= (mechanical ? 6 : 7) && (dimension == 2 || dimension == 3), "invalid CUDA RHS shape/profile");
        DNDS_check_throw_info(solver >= Riemann::Solver::Rusanov && solver <= Riemann::Solver::AUSMPlusUp2, "unknown CUDA Riemann solver");
        DNDS_check_throw_info(direct || firstOrder || (modes > 0 && modes <= 19), "RHS polynomial degree/shape is missing");
        DNDS_check_throw_info(!primitiveGradient || components == 5, "primitive viscous gradients require pure flow5");
        DNDS_check_throw_info(!produceLow || components == 5, "low-order flux blending applies to flow5 only");
        using P = DNDS::Math::ComputePrecision;
        DNDS_check_throw_info(blockSize == 64 || blockSize == 128 || blockSize == 256 || blockSize == 512, "invalid CUDA RHS block size");
        const bool fp64 = polynomialPrecision == P::FP64 && dissipationPrecision == P::FP64 && fluxPrecision == P::FP64;
        if (mechanical)
        {
            DNDS_check_throw_info(fp64 && !produceLow && !positivity && !primitiveGradient && !faceFallback &&
                                      !frame.moving && mapping == RHSFaceMapping::Thread && !doubleBuffer,
                                  "unsupported CUDA ACM spatial kernel policy");
            return;
        }
        DNDS_check_throw_info(!doubleBuffer, "buffered-face RHS experiment has been retired");
        DNDS_check_throw_info(mapping == (fp64 ? RHSFaceMapping::CTASegments : RHSFaceMapping::Thread),
                              "FP64 CUDA RHS requires CTA segments; reduced-precision comparisons require thread mapping");
        DNDS_check_throw_info(reduction >= RHSFaceReduction::Serial && reduction <= RHSFaceReduction::Segmented && gridBlocks >= 0,
                              "invalid CUDA RHS mapping/reduction/grid selector");
        DNDS_check_throw_info(mapping != RHSFaceMapping::CTASegments || blockSize >= 128,
                              "segmented RHS uses its own shared staging and 128/256/512 threads");
        P selected = P::FP64;
        for (const auto precision : {polynomialPrecision, dissipationPrecision, fluxPrecision})
        {
            DNDS_check_throw_info(precision >= P::FP64 && precision <= P::FloatFloat, "unknown CUDA RHS arithmetic");
#ifndef DNDS_USE_CCCL_COMPUTE
            DNDS_check_throw_info(precision != P::FloatFloat, "float-float RHS requires the CCCL compute build option");
#endif
            if (precision != P::FP64)
            {
                DNDS_check_throw_info(selected == P::FP64 || selected == precision,
                                      "one face launch requires one common reduced arithmetic type");
                selected = precision;
            }
        }
        DNDS_check_throw_info(selected == P::FP64 || components == 5, "reduced RHS arithmetic is specialized for flow5");
        DNDS_check_throw_info(fluxPrecision == P::FP64 || (!primitiveGradient && dissipationPrecision == P::FP64),
                              "full reduced flux uses conservative gradients and includes its own dissipation");
        DNDS_check_throw_info(dissipationPrecision == P::FP64 || (solver >= Riemann::Solver::Roe && solver <= Riemann::Solver::Roe_M9),
                              "reduced dissipation currently requires the Roe family");
    }
}
