#pragma once
#include "CFV/Detail/CellRegionRows.hpp"

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Math/LimitedVR.hpp"

namespace DNDS::CFV
{
    /// Immutable face-incidence stencil for scalar LVR gates. No EOS, target
    /// construction, communication or retained iteration history is hidden here.
    /// Copies share immutable metadata buffers and the metric/mesh publication.
    class LimitedVRPlan
    {
        FiniteVolumePlan _metrics;
        Geom::EntityWorkPartition _partition;
        Runtime::Array<index> _offsets, _neighbors, _boundary;
        Detail::ReconstructionRegionRows _regionRows;
        Runtime::Array<real> _empty;
        explicit LimitedVRPlan(FiniteVolumePlan metrics)
            : _metrics(std::move(metrics)), _partition(_metrics.CellWorkPartition()),
              _regionRows(_partition, _metrics.NumCell()), _empty(Runtime::RowLayout::Dense(0, 0), _metrics.Device()) {}
        void ValidateRegion(ReconstructionRegion region) const;
        void ValidateSeeds(const FVField &, const FVField &, const Runtime::Array<real> &,
                           const FVField &, ReconstructionRegion) const;
        void ValidateMaximum(const FVField &, const FVField &, std::array<real, 3>, ReconstructionRegion) const;

    public:
        static LimitedVRPlan Prepare(FiniteVolumePlan metrics);
        Runtime::Device Device() const { return _metrics.Device(); }
        /// Place immutable incidence/selection buffers, sharing the supplied
        /// resident metric plan instead of allocating duplicate metric tables.
        LimitedVRPlan Placed(FiniteVolumePlan metrics, Runtime::ExecutionContext context) const;
        /// Mean scalars are [p,rho]; base gates [shock,contact,mean-contact
        /// support]. Previous is an owned table of [shock,contact] from the last
        /// RHS input. Output is [pressure seed,density seed,0]. Only v3 adds the
        /// density seed. Merge base gates after expanding each seed's own depth.
        /// No MPI: caller completes base/input halos before dependent/All work.
        void BuildExtremumSeeds(const FVField &meanScalars, const FVField &baseGates,
                                const Runtime::Array<real> &previous, FVField &output,
                                const LimitedVRSettings &settings, ReconstructionRegion region = ReconstructionRegion::All) const;
        /// Retained queued operator, shared by CPU inline/worker and CUDA.
        /// Core declares owned reads only; the caller orders any required halo.
        void BuildExtremumSeeds(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                const FVField &meanScalars, const FVField &baseGates,
                                const Runtime::Array<real> &previous, FVField &output,
                                const LimitedVRSettings &, ReconstructionRegion region = ReconstructionRegion::All) const;
        /// One synchronous, out-of-place interior-face maximum round on three
        /// scalar gate columns. Inactive columns copy unchanged. Correction
        /// seeds exclude boundary-adjacent receivers; extra shock layers do not.
        /// Saturation filtering uses caps[column]*(1-1e-12), preserving each
        /// local unsaturated value. Input halos must be current for dependent/
        /// All; Core reads owned only. Caller finishes both phases before copy
        /// or the next exchange. Distinct output; no allocation in cell loops.
        void NeighborMaximum(const FVField &input, FVField &output, std::array<bool, 3> active,
                             bool excludeBoundary, bool saturatedOnly = false, std::array<real, 3> caps = {1, 1, 1},
                             ReconstructionRegion region = ReconstructionRegion::All) const;
        void NeighborMaximum(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                             const FVField &input, FVField &output, std::array<bool, 3> active,
                             bool excludeBoundary, bool saturatedOnly = false, std::array<real, 3> caps = {1, 1, 1},
                             ReconstructionRegion region = ReconstructionRegion::All) const;
    };
}
