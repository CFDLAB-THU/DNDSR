#pragma once
#include "Solver/SmallBlockInverse.hpp"

namespace DNDS::Direct::ScaledBlock
{
    // A row scales by its signed diagonal when that normalization is finite.
    // Otherwise use its largest magnitude. A structural zero row stays zero
    // in the scaled problem. Dividing by divisors avoids overflowing 1/d.
    enum InverseStatus
    {
        Success,
        NonfiniteInput,
        SVDNoConvergence,
        NonfiniteOutput
    };

    template <int N>
    DNDS_DEVICE_CALLABLE InverseStatus InvertBlock(const double *block, double *output)
    {
        Direct::SmallDenseMatrix<N> matrix, inverse;
        double divisor[N];
        bool any = false;
        for (int i = 0; i < N; ++i)
        {
            double largest = 0;
            for (int j = 0; j < N; ++j)
            {
                const double x = block[i * N + j];
                if (!std::isfinite(x))
                    return NonfiniteInput;
                largest = std::fmax(largest, std::fabs(x));
            }
            any = any || largest > 0;
            const double d = block[i * N + i];
            divisor[i] = d != 0 && std::isfinite(1 / d) && std::isfinite(largest / d) ? d : (largest > 0 ? largest : 1);
            for (int j = 0; j < N; ++j)
                matrix(i, j) = block[i * N + j] / divisor[i];
        }
        if (!any)
        {
            for (int k = 0; k < N * N; ++k)
                output[k] = 0;
            return Success;
        }
        int truncated = 0;
        if (!Direct::TrySmallLUInverse<N, false>(matrix, inverse) &&
            !Direct::SmallSVDInverse<N, false>(matrix, inverse, truncated))
            return SVDNoConvergence;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                const double x = inverse(i, j) / divisor[j];
                if (!std::isfinite(x))
                    return NonfiniteOutput;
                output[i * N + j] = x;
            }
        return Success;
    }
}
