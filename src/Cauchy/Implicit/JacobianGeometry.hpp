#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy
{
    /// Transient centered incidence arithmetic. Never stored as an Array:
    /// topology, center metrics and periodic maps remain shared publications.
    struct JacobianPointRecord
    {
        index other, auxiliary, sample;
        Vector3 normal, location;
        std::array<real, 9> rotation;
        real weight;
        bool changesFrame;
    };
    static_assert(std::is_trivially_copyable_v<JacobianPointRecord>);

    struct JacobianPointValue
    {
        index other, auxiliary, sample;
        Vector3 normal, location;
        Eigen::Matrix3d rotation;
        real weight;
        bool changesFrame;
    };
    inline JacobianPointValue ExpandJacobianPoint(const JacobianPointRecord &point)
    {
        return {point.other, point.auxiliary, point.sample, point.normal,
                point.location, Eigen::Map<const Eigen::Matrix3d>(point.rotation.data()),
                point.weight, point.changesFrame};
    }

    /** @brief Sparse block-to-incidence routing over the canonical FV stencil.
     * @details Each stored integer identifies an actual cell-face incidence;
     * repeated periodic images are preserved even for one sparse column. The
     * large stencil descriptor is published once and borrowed by array view,
     * avoiding both repeated geometry and oversized CUDA kernel arguments.
     */
    struct JacobianGeometryView
    {
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::CSRRows> columns;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> blockOffsets;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::CSRRows> incidences;
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        DNDS_DEVICE_CALLABLE auto Points(index iCell, index iBlock) const
        {
            DNDS_HD_assert(iBlock >= 0 && iBlock < columns.RowSize(iCell));
            return incidences[blockOffsets(iCell, 0) + iBlock];
        }
        DNDS_DEVICE_CALLABLE JacobianPointRecord Point(index iCell, index incidence) const
        {
            const auto &shared = stencil(0, 0);
            const auto sample = shared.JacobianSample(iCell, incidence);
            return {sample.other, sample.auxiliaryFace, incidence, sample.normal,
                    sample.ownPoint, sample.rotation, shared.metric.GetFaceArea(sample.face), sample.changesFrame};
        }
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, columns, blockOffsets, incidences, stencil);
        }
    };

    /** @brief Rank-local sparse routing, retaining shared canonical geometry.
     * @details Numerical snapshots keep this owner after their FV plan is
     * released. Only integer routing is copied to a device. symbolicTopology
     * is used synchronously during setup, in the same local numbering as the
     * retained stencil; it need not be retained after construction. No metric
     * or rotation field is downloaded or re-expanded for symbolic setup.
     */
    class JacobianGeometryStorage
    {
        Runtime::Array<index> _columns, _blockOffsets, _incidences;
        FiniteVolumeStencilStorage::ReadAccess _stencil;
        void Build(const std::vector<std::vector<index>> &, Geom::CellFaceIncidenceView,
                   Runtime::ExecutionContext);

    public:
        bool sameComponentFrames = true;
        JacobianGeometryStorage(FiniteVolumeStencilStorage::ReadAccess stencil,
                                const std::vector<std::vector<index>> &columns);
        JacobianGeometryStorage(FiniteVolumeStencilStorage::ReadAccess stencil,
                                const std::vector<std::vector<index>> &columns,
                                Geom::CellFaceIncidenceView symbolicTopology,
                                bool sameFrames, Runtime::ExecutionContext context);
        class ReadAccess
        {
            Runtime::Array<index>::ReadLease _columns, _blockOffsets, _incidences;
            FiniteVolumeStencilStorage::ReadAccess _stencil;

        public:
            explicit ReadAccess(const JacobianGeometryStorage &storage, Runtime::ExecutionContext context)
                : _columns(storage._columns.Read(context)), _blockOffsets(storage._blockOffsets.Read(context)),
                  _incidences(storage._incidences.Read(context)), _stencil(storage._stencil) {}
            JacobianGeometryView View() const
            {
                return {_columns.View<Runtime::CSRRows>(), _blockOffsets.View<Runtime::FixedRows<1>>(),
                        _incidences.View<Runtime::CSRRows>(), _stencil.KernelDescriptor()};
            }
        };
        /// Retain through asynchronous use of the borrowed geometry views.
        ReadAccess Read(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const { return ReadAccess(*this, context); }
        std::vector<std::vector<index>> CopyColumns() const;
    };
}
