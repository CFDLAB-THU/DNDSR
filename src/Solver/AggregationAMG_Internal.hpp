#pragma once
#include "Solver/AggregationAMG.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Solver
{
    namespace AMGDetail
    {
        namespace R = Runtime;
        template <class T>
        R::Array<T> Table(const std::string &name, index rows, index width)
        {
            R::Array<T> array(R::RowLayout::Dense(rows, width));
            array.SetName(name, DNDS_ALLOCATION_SITE());
            return array;
        }

        template <class T>
        void RequireTable(const R::Array<T> &a, index rows, index width)
        {
            const auto layout = a.Layout();
            DNDS_check_throw_info(a.GetDevice() == R::Device::Host() && layout.Kind() == R::RowLayoutKind::Table &&
                                      a.RowCount() == rows && layout.Width() == width,
                                  "aggregation AMG host table shape or residency mismatch");
        }

        struct Level
        {
            R::Array<int> offsets, columns, group, memberOffsets, members, edgeSlot;
            R::Array<real> blocks, diagonal, inverse, rhs, x, residual;
        };

        template <class Body>
        void Rows(index count, Body &&body)
        {
            if (count < 64)
                for (index i = 0; i < count; ++i)
                    body(i);
            else
                OMP::ForRows(0, count, body);
        }
    }

    struct AggregationAMG::Data
    {
        int width;
        AggregationAMGSettings settings;
        std::vector<AMGDetail::Level> levels;
        uint64_t valuesEpoch = 0, diagonalEpoch = 0;
        bool ready = false;
        void Cycle(index depth, const Runtime::Array<real> &rhs, Runtime::Array<real> &correction);
    };
}
