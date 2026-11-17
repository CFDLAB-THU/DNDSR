#include "CFV/CPU/ReconstructionAMGKernels.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV::CPU
{
    void AssembleReconstructionAMG(const ReconstructionAMGSetup &request)
    {
        const auto ptr = request.rowOffsets, col = request.columns;
        const auto offsets = request.offsets, neighbors = request.neighbors, faces = request.faces, boundary = request.boundaryOffsets;
        const auto matrices = request.matrices, vector = request.vectors, injection = request.injection, inverse = request.inverse;
        const auto sides = request.faceSides;
        const auto a = request.output, v = request.normalizedVectors, b = request.normalizedInjection;
        const index n = request.modes, rows = ptr.RowCount() - 1;
        OMP::ForSetupRows(0, rows, [&](index iCell)
                          {
            const index first = ptr(iCell, 0), end = ptr(iCell + 1, 0);
            for (index entry = first; entry < end; ++entry)
            {
                for (index k = 0; k < n * n; ++k) a(entry * n * n + k, 0) = 0;
            }
            const auto ordinal = [&](index other)
            {
                index position = first;
                while (position < end && col(position, 0) != other) ++position;
                DNDS_assert(position < end);
                return position;
            };
            const index diagonal = ordinal(iCell);
            for (index i = 0; i < n; ++i) a(diagonal * n * n + i * n + i, 0) = 1;
            const auto inv = [&](index i, index j)
            { return Math::ReconstructionInverseEntry(inverse, iCell, i, j, n, request.triangle); };
            for (index entry = offsets(iCell, 0); entry < offsets(iCell + 1, 0); ++entry)
            {
                const index other = neighbors(entry, 0);
                if (other != UnInitIndex && (!request.raw || other != iCell))
                {
                    const index target = ordinal(other);
                    for (index i = 0; i < n; ++i)
                        for (index j = 0; j < n; ++j)
                        {
                            real value = 0;
                            if (request.raw)
                                for (index k = 0; k < n; ++k)
                                    value += inv(i, k) * Math::ReconstructionCouplingEntry(matrices, faces, sides, entry, k, j, n, request.faceCoupling);
                            else
                                value = matrices(entry, i + n * j);
                            a(target * n * n + i * n + j, 0) -= value;
                        }
                }
                for (index i = 0; i < n; ++i)
                {
                    real value = vector(entry, i);
                    if (request.raw)
                    {
                        value = 0;
                        for (index k = 0; k < n; ++k) value += inv(i, k) * vector(entry, k);
                    }
                    v(entry, i) = value;
                }
                for (index iPoint = boundary(entry, 0); iPoint < boundary(entry + 1, 0); ++iPoint)
                    for (index i = 0; i < n; ++i)
                    {
                        real value = injection(iPoint, i);
                        if (request.raw)
                        {
                            value = 0;
                            for (index k = 0; k < n; ++k) value += inv(i, k) * injection(iPoint, k);
                        }
                        b(iPoint, i) = value;
                    }
            } });
    }
}
