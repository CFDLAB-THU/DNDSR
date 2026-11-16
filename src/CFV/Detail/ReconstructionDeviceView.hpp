#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"

namespace DNDS::CFV::Detail
{
    /// Immutable numeric publication; all pointers have the placement of the
    /// resident plan. No host owner, Eigen object, map or functor reaches kernels.
    struct ReconstructionDeviceView
    {
        Runtime::ArrayView<const index> offsets, neighbors, faces, boundaryOffsets;
        Runtime::ArrayView<const uint8_t> rotationActive, faceSide;
        Runtime::ArrayView<const real> matrixAInvB, vectorAInvB, secondary, boundaryInjection, rotation, localInverse;
        Runtime::ArrayView<const float> matrixAInvB32, vectorAInvB32, boundaryInjection32;
        Runtime::ArrayView<const index> core, dependent;
        index owned = 0, n = 0, maxRowEntries = 0;
        real relax = 1, wbapCenterWeight = 10;
        bool rotated = false;
        bool quantized = false;
        bool raw = false, inverseTriangle = false, faceCoupling = false;
        /// One host launch check for every nested allocation; never called by
        /// point/row arithmetic. The placed plan retains all borrowed storage.
        void ValidateDevice(Runtime::Device expected) const
        {
            Runtime::ValidateViewDevices(expected, offsets, neighbors, faces, boundaryOffsets, rotationActive,
                                         matrixAInvB, vectorAInvB, secondary, boundaryInjection, rotation, core, dependent,
                                         matrixAInvB32, vectorAInvB32, boundaryInjection32, localInverse, faceSide);
        }
    };
    /// Sweep-only projection, excluding WBAP/sampling data from kernel arguments.
    struct RawReconstructionSweepView
    {
        Runtime::ArrayView<const index> offsets, neighbors, faces, boundaryOffsets;
        Runtime::ArrayView<const uint8_t> rotationActive, faceSide;
        Runtime::ArrayView<const real> matrices, vectors, injection, inverse, rotation;
        index n;
        real relax;
        bool rotated, triangle, faceCoupling;
        explicit RawReconstructionSweepView(const ReconstructionDeviceView &p)
            : offsets(p.offsets), neighbors(p.neighbors), faces(p.faces), boundaryOffsets(p.boundaryOffsets),
              rotationActive(p.rotationActive), faceSide(p.faceSide), matrices(p.matrixAInvB), vectors(p.vectorAInvB),
              injection(p.boundaryInjection), inverse(p.localInverse), rotation(p.rotation), n(p.n), relax(p.relax),
              rotated(p.rotated), triangle(p.inverseTriangle), faceCoupling(p.faceCoupling) {}
    };
    struct DeviceWorkSelection
    {
        Runtime::ArrayView<const index> rows;
        index count = 0;
        bool indirect = false;
        void ValidateDevice(Runtime::Device expected) const
        {
            if (indirect)
                rows.ValidateDevice(expected);
        }
        DNDS_DEVICE_CALLABLE index Cell(index i) const { return indirect ? rows(i, 0) : i; }
    };
    struct DeviceCharacteristicView
    {
        Runtime::ArrayView<const real> forward, inverse;
        index first = 0, count = 0;
        void ValidateDevice(Runtime::Device expected) const
        {
            if (count)
                Runtime::ValidateViewDevices(expected, forward, inverse);
        }
    };
    struct DevicePeriodicBlocks
    {
        Runtime::ArrayView<const index> starts;
        void ValidateDevice(Runtime::Device expected) const { starts.ValidateDevice(expected); }
        DNDS_DEVICE_CALLABLE index First(index component) const
        {
            for (index i = 0; i < starts.RowCount(); ++i)
            {
                const auto first = starts(i, 0);
                if (component >= first && component < first + 3)
                    return first;
            }
            return -1;
        }
    };
}
