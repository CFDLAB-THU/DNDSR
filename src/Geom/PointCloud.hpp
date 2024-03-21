#pragma once
#include "DNDS/Defines.hpp"
#include "Geometric.hpp"

namespace DNDS::Geom
{
    struct PointCloudKDTree
    {
        using coord_t = real; //!< The type of each coordinate

        std::vector<tPoint> pts;

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return pts.size(); }

        inline real kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            if (dim == 0)
                return pts[idx].x();
            else if (dim == 1)
                return pts[idx].y();
            else
                return pts[idx].z();
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const
        {
            return false;
        }
    };

    struct PointCloudFunctional
    {
        using coord_t = real; //!< The type of each coordinate
        using t_f = std::function<tPoint(size_t)>;

        t_f _fun;
        index _size;

        PointCloudFunctional(const t_f &fun, index size) : _fun(fun), _size(size)
        {
        }

        index size() const { return _size; };

        tPoint operator[](index i) const
        {
            DNDS_assert(i >= 0 && i < size());
            return _fun(i);
        }

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return _size; }

        inline real kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            return this->operator[](idx)(dim);
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const
        {
            return false;
        }
    };
}
