#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"

namespace DNDS::CFV::CPU
{
    namespace R = Runtime;
    /// Setup-only projected operator assembly over preallocated disjoint rows.
    /// The host wrapper supplies sorted unique local columns, full/triangle
    /// inverse interpretation and already validated owned incidence. No owner,
    /// allocation, lease or MPI call enters this arithmetic kernel.
    struct ReconstructionAMGSetup
    {
        R::ArrayView<const int> rowOffsets, columns;
        R::ArrayView<const index> offsets, neighbors, faces, boundaryOffsets;
        R::ArrayView<const real> matrices, vectors, injection, inverse;
        R::ArrayView<const uint8_t> faceSides;
        R::ArrayView<real> output, normalizedVectors, normalizedInjection;
        int modes = 0;
        bool raw = false, triangle = false, faceCoupling = false;
    };
    void AssembleReconstructionAMG(const ReconstructionAMGSetup &);
}
