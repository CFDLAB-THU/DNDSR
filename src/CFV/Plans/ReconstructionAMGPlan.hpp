#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"

namespace DNDS::CFV
{
    /** @brief Explicit single-device AMG exploration of the projected VR system.
     * @details The operator is I-A^+B, including accumulated repeated neighbor
     * columns. One modal-block hierarchy is reused for all physical components.
     * The initial profile rejects rotated periodicity and LVR; it does not
     * silently replace their coupled operators by independent scalar systems.
     * PublicUpload uses the provider C API; NativeStorage additionally borrows
     * the prepared fine numeric allocation. Both currently synchronize at the
     * provider boundary and copy provider vectors. This is not a graph API.
     */
    enum class ReconstructionAMGStorage
    {
        PublicUpload,
        NativeStorage
    };

    struct ReconstructionAMGWorkspace
    {
        Runtime::Array<real> defect, packedDefect, packedCorrection;
        index components = 0;
    };

    class ReconstructionAMGPlan
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        /** @brief Assemble the exact fixed-point matrix from a host VR plan.
         * @details Direct setup creates one row-major native modal matrix;
         * no second native GPU sweep representation is retained. Raw storage
         * is normalized with the existing filtered inverse, preserving free
         * modes. Matrix setup and hierarchy creation are synchronous. The
         * source must have sweep capability and a single-rank cell space.
         */
        static ReconstructionAMGPlan Prepare(const VariationalReconstructionPlan &source,
                                             Runtime::ExecutionContext context,
                                             ReconstructionAMGStorage storage = ReconstructionAMGStorage::PublicUpload,
                                             Algebra::AmgXSettings settings = {});
        ReconstructionAMGWorkspace BuildWorkspace(index components) const;
        ReconstructionBoundaryValues PlaceBoundaryValues(const ReconstructionBoundaryValues &) const;
        /** @brief One fixed zero-start AMG defect correction into distinct output.
         * @details Evaluates the exact projected residual g-A*x, then applies
         * the same frozen modal hierarchy separately to each component. Native
         * combined coefficient rows remain [component][mode]; explicit device
         * gather/scatter is counted as vector adaptation, not zero-copy. Each
         * workspace/provider is serially used by its caller. Input halos may
         * be stale because this profile accepts only local neighbor columns;
         * output ghosts are not refreshed. No convergence controller is added.
         */
        void SubmitDefectCorrection(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                    const FVField &mean, const FVField &input,
                                    const ReconstructionBoundaryValues &, FVField &output,
                                    ReconstructionAMGWorkspace &) const;
        Algebra::AmgXMemorySnapshot MemorySnapshot() const;
        /// Exact native matrix values, including one required provider sentinel.
        /// Read-only access supports independent matrix-action/oracle checks.
        const Runtime::Array<real> &MatrixValues() const;
        const Runtime::Array<int> &RowOffsets() const;
        const Runtime::Array<int> &Columns() const;
    };
}
