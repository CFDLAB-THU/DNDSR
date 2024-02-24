#pragma once

#include "../ArrayTransformer.hpp"

namespace DNDS
{
    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    class ArrayEigenVector : public ParArray<real, _vec_size, _row_max, _align>
    {
    public:
        using t_base = ParArray<real, _vec_size, _row_max, _align>;
        using t_base::t_base;

        using t_EigenVector = Eigen::Matrix<real, RowSize_To_EigenSize(_vec_size), 1>;
        using t_EigenMap = Eigen::Map<t_EigenVector>; // default no buffer align and stride

        using t_copy = t_EigenVector;

    public:
        t_EigenMap operator[](index i)
        {
            return t_EigenMap(t_base::operator[](i), t_base::RowSize(i)); // need static dispatch?
        }

        using t_base::ReadSerializer;
        using t_base::WriteSerializer; //! because no extra data than Array<>
    };
}
