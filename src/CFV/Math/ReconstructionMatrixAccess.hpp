#pragma once
#include "DNDS/Core/Defines.hpp"

namespace DNDS::CFV::Math
{
    /// Logical row-packed lower-triangle entry, independent of Array layout.
    DNDS_DEVICE_CALLABLE inline index SymmetricTriangleEntry(index i, index j)
    {
        const index high = i > j ? i : j, low = i > j ? j : i;
        return high * (high + 1) / 2 + low;
    }
    /// One full oriented cross block per face, or one directed block per
    /// incidence. Reverse-face lookup transposes the full block, never assumes
    /// that a generic B itself is symmetric. Setup owns the orientation choice.
    template <class MatrixRows, class FaceRows, class SideRows>
    DNDS_DEVICE_CALLABLE real ReconstructionCouplingEntry(MatrixRows matrices, FaceRows faces, SideRows sides,
                                                          index entry, index i, index j, index n, bool faceCoupling)
    {
        if (!faceCoupling)
            return matrices(entry, i + n * j);
        return matrices(faces(entry, 0), sides(entry, 0) ? j + n * i : i + n * j);
    }
    template <class MatrixRows>
    DNDS_DEVICE_CALLABLE real ReconstructionInverseEntry(MatrixRows inverse, index cell,
                                                         index i, index j, index n, bool triangle)
    {
        return inverse(cell, triangle ? SymmetricTriangleEntry(i, j) : i + n * j);
    }
}
