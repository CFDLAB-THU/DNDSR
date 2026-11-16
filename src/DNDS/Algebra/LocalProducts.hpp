#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/Core/OMP.hpp"
#include <vector>

namespace DNDS::Algebra::Detail
{
    /// Same logical component scaling for combined and separately stored DOFs.
    /// Caller validates scale width; the empty case performs no scale loads.
    template <class Source, class Probe>
    real AccumulateLocalProduct(Source source, Probe probe, index rows, index components,
                                const std::vector<real> &inverseScales)
    {
        real product = 0;
        for (index iRow = 0; iRow < rows; ++iRow)
            for (index component = 0; component < components; ++component)
                product += inverseScales.empty() ? source(iRow, component) * probe(iRow, component)
                                                 : (source(iRow, component) * inverseScales[component]) *
                                                       (probe(iRow, component) * inverseScales[component]);
        return product;
    }
    /// Private CPU batch-product arithmetic over already acquired views.
    /// Ownership/schema validation stays in each field/equation adapter. Keep
    /// row -> logical component -> probe accumulation order for reproducibility;
    /// no layout flattening, communication, allocation or per-element dispatch.
    template <class Source, class Probe>
    void AccumulateLocalProducts(Source source, const std::vector<Probe> &probes,
                                 index rows, index components, std::vector<real> &products,
                                 const std::vector<real> &inverseScales = {})
    {
        // Distinct probes are independent outputs; preserving each probe's
        // row/component order gives identical sums at every CPU team size.
        OMP::ForRows(0, index(probes.size()), [&](index iProbe)
                     {
        for (index iRow = 0; iRow < rows; ++iRow)
            for (index component = 0; component < components; ++component)
                    products[iProbe] += inverseScales.empty() ? source(iRow, component) * probes[iProbe](iRow, component)
                                                              : (source(iRow, component) * inverseScales[component]) *
                                                                    (probes[iProbe](iRow, component) * inverseScales[component]); });
    }
}

namespace DNDS::Algebra
{
    /** @brief Reserved local scalar reductions for queued field operations.
     * @details Product slots share bounded tree scratch, so submissions must
     * use one ordered lane (or explicit cross-lane dependencies). Inputs and
     * scratch are retained by eager operation records. No halo or MPI is
     * performed. Scales are setup-sized metadata; values remain resident.
     * Products compute sum((x/s)*(y/s)) with the supplied inverse scales.
     * Finite checks return one for all-finite, including an empty rank.
     * CPU partitions contiguous rows and merges in partition order (OMP=1
     * retains canonical row/component order); CUDA uses a fixed two-level
     * reduction tree, with no floating-point atomic adds. A per-workspace
     * completion counter elects the last block to perform the tail in the same
     * kernel; the numerical order is independent of block arrival order.
     * Cross-backend sums
     * agree numerically, not necessarily bitwise.
     */
    class LocalProductWorkspace
    {
        index _rows, _components;
        Runtime::Device _device;
        Runtime::Array<real> _scales, _partial, _result;
        Runtime::Array<unsigned int> _completed;
        std::vector<real> _inverseScales;
        void Submit(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                    const Runtime::Array<real> &, const Runtime::Array<real> &, index slot, bool finite);

    public:
        LocalProductWorkspace(index rows, index components, index capacity, Runtime::Device device);
        /// Synchronous configuration boundary; serialize with all submissions.
        /// Identical scales reuse their resident allocation without uploading.
        void SetInverseScales(const std::vector<real> &, Runtime::ExecutionContext);
        void SubmitProduct(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                           const Runtime::Array<real> &x, const Runtime::Array<real> &y, index slot);
        void SubmitFinite(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                          const Runtime::Array<real> &x, index slot = 0);
        /// Explicit host observation after the caller drains the producing
        /// lane. Copies only the small result buffer; does not synchronize MPI.
        std::vector<real> ReadResults(index count, Runtime::ExecutionContext) const;
    };

    namespace Detail
    {
        using ProductInput = Runtime::ArrayView<const real>;
        using ProductOutput = Runtime::ArrayView<real>;
        using ProductCompletion = Runtime::ArrayView<unsigned int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
        /// Host launcher; all inputs and scratch are borrowed whole-array views.
        void ReduceLocalProductCUDA(Runtime::ExecutionContext, ProductInput x, ProductInput y,
                                    ProductInput inverseScales, ProductOutput partial, ProductOutput result, ProductCompletion completed,
                                    index slot, bool finite);
    }
}
