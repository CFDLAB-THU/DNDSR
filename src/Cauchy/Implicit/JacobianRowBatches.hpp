#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"

namespace DNDS::Cauchy
{
    /// Host-side assembly schedule. The retained FV stencil owns the optional
    /// index lists on the execution device. Whole-matrix preparation, source
    /// addition and publication each occur once around these disjoint flux rows.
    class JacobianRowBatches
    {
        std::array<Runtime::RowSelection, 2> _rows;
        unsigned _size = 1;

    public:
        JacobianRowBatches(const FiniteVolumeStencilView &stencil, index owned, bool split)
        {
            using Region = CFV::ReconstructionRegion;
            if (split)
            {
                _rows[0] = stencil.RHSRows(Region::Core, false, owned);
                _rows[1] = stencil.RHSRows(Region::HaloDependent, false, owned);
                if (_rows[0].RowCount() && _rows[1].RowCount())
                {
                    _size = 2;
                    return;
                }
            }
            _rows[0] = stencil.RHSRows(Region::All, false, owned);
        }
        bool Split() const { return _size == 2; }
        auto begin() const { return _rows.begin(); }
        auto end() const { return begin() + _size; }
    };
}
