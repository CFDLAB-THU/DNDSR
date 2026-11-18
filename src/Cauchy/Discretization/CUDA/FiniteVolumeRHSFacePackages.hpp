#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Core/Span.hpp"
#include <memory>

namespace DNDS::Cauchy::CUDA
{
    enum class RHSFaceMapping : uint8_t
    {
        Thread,
        WarpQuadrature,
        WarpSegments,
        CTASegments
    };
    enum class RHSFaceReduction : uint8_t
    {
        Serial,
        Segmented
    };
    struct RHSKernelControls;
    struct RHSFaceStageShape
    {
        int components = 5, reconstruction = 0, viscous = 0, width = 0;
        DNDS_DEVICE_CALLABLE int ViscousFirst() const { return components + reconstruction; }
        DNDS_DEVICE_CALLABLE int AuxFirst() const { return components + reconstruction + viscous; }
    };
    /// Every segment is a complete face. lanes is its bounded point tile;
    /// larger quadrature rules revisit those lanes in increasing point tiles.
    struct RHSFaceSegment
    {
        index auxiliaryFace = 0, points = 0;
        int firstLane = 0, lanes = 0;
    };
    struct RHSFacePackage
    {
        index firstSegment = 0;
        int faces = 0, tiles = 0;
    };
    struct RHSFacePackageView
    {
        Runtime::ArrayView<const RHSFacePackage, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> packages;
        Runtime::ArrayView<const RHSFaceSegment, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> segments;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> selection;
        RHSFaceStageShape shape;
        RHSFaceMapping mapping = RHSFaceMapping::Thread;
        int blockSize = 0, maximumFaces = 0;
        bool produceLow = false;
        size_t sharedBytes = 0;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, packages, segments, selection);
        }
    };
    /** @brief Setup-only complete-face packing for a selected ready work set.
     * @details Preserves selection order and exact selected-array identity.
     * CTASegments packs tightly and may cross warps, never CTAs. Shared scratch
     * and staging fit 48 KiB or reject
     * explicitly. No shadow mesh/state or point-to-face field is retained.
     * Prepare may download selected face IDs once; no hot-path allocation,
     * upload or immediate access is hidden in View. Retain the owner through
     * every queued use of the borrowed package descriptor.
     */
    class RHSFacePackages
    {
        struct Data;
        std::shared_ptr<const Data> _data;
        explicit RHSFacePackages(std::shared_ptr<const Data> data) : _data(std::move(data)) {}

    public:
        /// Central shared-field shape for the CTA-segment kernel. It depends
        /// on semantic operands, not their values or static instantiation.
        static RHSFaceStageShape StageShape(const RHSKernelControls &controls);
        static size_t LaneScratchBytes(const RHSKernelControls &controls);
        static RHSFacePackages Prepare(const Runtime::Array<index> &selection, SpanView<const index> quadratureCountsByAux,
                                       const RHSKernelControls &controls, Runtime::ExecutionContext context);
        RHSFacePackageView View() const;
    };
}
