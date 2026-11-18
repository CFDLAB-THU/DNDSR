#pragma once
#include "Cauchy/Discretization/EquationPhysicsProfile.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp"

namespace DNDS::Cauchy::CPU
{
    template <class View>
    bool CompactFluxTable(const View &v, index width)
    {
        return (!v.rows && !v.data) || (v.device == Runtime::Device::Host() &&
                                        v.kind == Runtime::RowLayoutKind::Table && !v.sizes && !v.offsets &&
                                        v.width == width && v.stride == width);
    }
    /// The fixed catalog requires the actual parent pitch for every projected
    /// operand. A segmented or independently laid out coefficient field uses
    /// the same numerical body with its original dynamic descriptors.
    inline bool CompactFluxOperands(const EquationRHSRequest &r, ReconstructionForm form)
    {
        if (!CompactFluxTable(r.faceValues, r.width) || (r.produceLow && !CompactFluxTable(r.lowFaceValues, r.width)))
            return false;
        const auto compactPair = [&](const auto &pair, index w)
        { return CompactFluxTable(pair.owned, w) && CompactFluxTable(pair.ghost, w); };
        if (r.primitiveGradient && !compactPair(r.primitiveGradientValues, 5 * r.dimension))
            return false;
        if (form == ReconstructionForm::DirectGradient)
        {
            for (index i = 0; i < r.directLocations.size(); ++i)
                if (r.directLocations[i].physicalField != 0 || r.directLocations[i].firstComponent != r.logicalFirst[i])
                    return false;
            return r.directViews.size() == 1 && compactPair(r.directViews[0], r.width * r.dimension) &&
                   (r.directViscousViews.empty() || (r.directViscousViews.size() == 1 && compactPair(r.directViscousViews[0], r.width * r.dimension)));
        }
        if (form == ReconstructionForm::Modal)
        {
            const auto compact = [&](SpanView<const EquationBlockReadView> blocks)
            {
                if (blocks.empty() || blocks.size() != r.meanBlocks.size())
                    return false;
                for (index i = 0; i < blocks.size(); ++i)
                    if (blocks[i].owned.first != r.logicalFirst[i] * r.modes ||
                        blocks[i].ghost.first != r.logicalFirst[i] * r.modes ||
                        blocks[i].owned.parent.data != blocks[0].owned.parent.data ||
                        blocks[i].ghost.parent.data != blocks[0].ghost.parent.data ||
                        !CompactFluxTable(blocks[i].owned.parent, r.width * r.modes) ||
                        !CompactFluxTable(blocks[i].ghost.parent, r.width * r.modes))
                        return false;
                return true;
            };
            return compact(r.recBlocks) && (r.viscousRecBlocks.empty() || compact(r.viscousRecBlocks));
        }
        return true;
    }
    template <int W, class T, bool Checked, class Layout>
    auto FluxTable(Runtime::ArrayView<T, Checked, Layout> v)
    {
        if constexpr (W <= 0)
            return v;
        else
            return Runtime::ArrayView<T, Checked, Runtime::FixedRows<W, W>>{
                v.data, nullptr, v.rows, W, W, nullptr, v.kind, v.device};
    }

    /// One worker's face or cell range. All view descriptors borrow the
    /// enclosing operator's host leases, including through the face/gather join.
    struct FluxRowsRequest
    {
        const EquationRHSRequest &request;
        const EquationSystem &equation;
        const BoundaryPointOperator &boundary;
        const Riemann::RiemannFlux &flux;
        PointEvaluator &worker;
        EquationLayoutView<const real> owned, remote;
        EquationLayoutView<real> destination, low;
        CFV::ReconstructionRegion region;
        EquationRHSPointScratch &scratch;
        index first, last;
        bool faces;
    };
    using FluxRowsKernel = NumericalResult<index> (*)(const FluxRowsRequest &) noexcept;
}
