#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    GradientPlan GradientPlan::Prepare(Geom::UnstructuredMesh &mesh, FiniteVolumePlan metrics,
                                       GradientMethod method, real relativeCutoff,
                                       Diagnostics::SetupProgress progress, LeastSquaresWeights leastSquares,
                                       const std::vector<int> &extrapolatedZones)
    {
        auto setupStage = progress.Stage("FV/gradient");
        namespace R = Runtime;
        using namespace Geom;
        DNDS_check_throw_info(method == GradientMethod::GreenGauss || method == GradientMethod::LeastSquares || method == GradientMethod::GGMP,
                              "unknown FV gradient method");
        DNDS_check_throw_info(std::isfinite(relativeCutoff) && relativeCutoff >= 0, "invalid gradient cutoff");
        DNDS_check_throw_info(std::isfinite(leastSquares.areaPower) && std::isfinite(leastSquares.distancePower),
                              "least-squares weight exponents must be finite");
        auto metricRead = metrics.AcquireRead();
        auto meshRead = mesh.AcquireRead(MeshReadRequirements::CellFlux());
        DNDS_check_throw_info(mesh.Entities(EntityKind::Cell).SameOrder(metrics.Cells()), "gradient metrics and mesh orders differ");
        const auto view = meshRead.HostView();
        const auto geometry = metricRead.HostView();
        const int dim = metrics.Dimension();
        auto data = std::make_shared<Data>(metrics);
        const auto usesBoundaryValue = [&](index iFace)
        {
            if (method != GradientMethod::LeastSquares || extrapolatedZones.empty())
                return true;
            const int zone = view.GetFaceZone(iFace);
            return std::find(extrapolatedZones.begin(), extrapolatedZones.end(), zone) == extrapolatedZones.end();
        };
        // Prefix each cell's incidence and physical-boundary ranges before
        // filling. Per-cell SVD setup then writes disjoint preallocated rows;
        // neither boundary ordering nor incidence summation depends on OMP.
        data->offsets = R::Array<index>(R::RowLayout::Dense(metrics.NumCell() + 1, 1));
        R::Array<index> boundaryOffsets(R::RowLayout::Dense(metrics.NumCell() + 1, 1));
        auto offsetWrite = data->offsets.Write(), boundaryOffsetWrite = boundaryOffsets.Write();
        const auto offsets = offsetWrite.View<R::FixedRows<1>>(), boundaryStarts = boundaryOffsetWrite.View<R::FixedRows<1>>();
        std::vector<index> entryCounts(static_cast<size_t>(metrics.NumCell()));
        std::vector<index> boundaryCounts(static_cast<size_t>(metrics.NumCell()));
        OMP::ForSetupRows(0, metrics.NumCell(), [&](index iCell)
                          {
            index boundaries = 0;
            for (index slot = 0; slot < view.CellFaces(iCell).size(); ++slot)
            {
                const index face = view.CellFaces(iCell)[slot];
                if (view.CellFaceOther(iCell, face, rowsize(slot)) == UnInitIndex)
                    ++boundaries;
            }
            entryCounts[size_t(iCell)] = index(view.CellFaces(iCell).size());
            boundaryCounts[size_t(iCell)] = boundaries; });
        const index entries = OMP::ExclusivePrefixRows(
            metrics.NumCell(), [&](index row) noexcept
            { return entryCounts[size_t(row)]; },
            [&](index row, index value) noexcept
            { offsets(row, 0) = value; });
        const index boundaryCount = OMP::ExclusivePrefixRows(
            metrics.NumCell(), [&](index row) noexcept
            { return boundaryCounts[size_t(row)]; },
            [&](index row, index value) noexcept
            { boundaryStarts(row, 0) = value; });
        data->boundaryIds = R::Array<index>(R::RowLayout::Dense(entries, 1));
        data->weights = R::Array<real>(R::RowLayout::Dense(entries, 3));
        data->offsets.SetName("CFV/gradient/offsets", DNDS_ALLOCATION_SITE());
        data->boundaryIds.SetName("CFV/gradient/boundaryIds", DNDS_ALLOCATION_SITE());
        data->weights.SetName("CFV/gradient/weights", DNDS_ALLOCATION_SITE());
        data->boundary.resize(size_t(boundaryCount));
        data->boundaryCount = boundaryCount;
        const auto storeRows = [](const std::vector<index> &rows)
        {
            R::Array<index> result(R::RowLayout::Dense(index(rows.size()), 1));
            auto write = result.Write();
            const auto view = write.View<R::FixedRows<1>>();
            OMP::ForRows(0, index(rows.size()), [&](index iRow)
                         { view(iRow, 0) = rows[iRow]; });
            return result;
        };
        data->core = storeRows(data->partition.Core());
        data->dependent = storeRows(data->partition.HaloDependent());
        auto boundaryWrite = data->boundaryIds.Write();
        auto weightWrite = data->weights.Write();
        const auto boundaries = boundaryWrite.View<R::FixedRows<1>>();
        const auto weights = weightWrite.View<R::FixedRows<3>>();
        OMP::ForSetupRows(0, metrics.NumCell(), [&](index iCell)
                          {
            index next = offsets(iCell, 0), nextBoundary = boundaryStarts(iCell, 0);
            const auto center = geometry.GetCellBary(iCell);
            const index count = offsets(iCell + 1, 0) - next;
            Eigen::MatrixXd design(count, dim), initial(dim, count);
            Eigen::VectorXd lengths(count);
            Eigen::MatrixXd correction = Eigen::MatrixXd::Identity(dim, dim);
            index local = 0;
            for (index slot = 0; slot < view.CellFaces(iCell).size(); ++slot)
            {
                const index face = view.CellFaces(iCell)[slot];
                const index other = view.CellFaceOther(iCell, face, rowsize(slot));
                DNDS_check_throw_info(other == UnInitIndex || (other >= 0 && other < metrics.NumCellProc()),
                                      "gradient stencil requires a complete face-neighbor ghost mapping");
                const bool back = view.CellIsFaceBack(iCell, face, rowsize(slot));
                const int side = back ? 0 : 1;
                {
                    const tPoint point = geometry.periodic.FacePoint(face, side, geometry.GetFaceCent(face));
                    const tPoint ownNormal = (side == 0 ? geometry.GetFaceMeanNorm(face)
                        : tPoint(geometry.periodic.Rotation(face, 0, 1) * geometry.GetFaceMeanNorm(face))) * (back ? 1 : -1);
                    // Euler's mapped reference center, normalized integrated
                    // normal and separately integrated area. An axis face has
                    // zero weighted area but retains its geometric direction.
                    const real area = geometry.GetFaceArea(face);
                    const tPoint areaNormal = ownNormal * (area / geometry.GetCellVol(iCell));
                    const tPoint otherCenter = other == UnInitIndex ? point : geometry.periodic.OtherPoint(face, side, geometry.GetCellBary(other));
                    const tPoint displacement = otherCenter - center;
                    const real length = displacement.head(dim).norm();
                    DNDS_check_throw_info(length > 0 && std::isfinite(length), "degenerate FV gradient sample displacement");
                    const bool useSample = other != UnInitIndex || usesBoundaryValue(face);
                    if (method == GradientMethod::LeastSquares)
                    {
                        real scale = 0;
                        if (useSample)
                        {
                            const real areaScale = leastSquares.areaPower == 0 ? 1 : std::pow(area, .5 * leastSquares.areaPower);
                            scale = areaScale * (leastSquares.distancePower == 0 ? 1 : std::pow(length, .5 * leastSquares.distancePower));
                            DNDS_check_throw_info(std::isfinite(scale) && scale >= 0, "invalid least-squares sample weight");
                        }
                        design.row(local) = displacement.head(dim).transpose() * scale;
                        lengths[local] = scale;
                    }
                    else
                        design.row(local) = displacement.head(dim).transpose();
                    boundaries(next, 0) = UnInitIndex;
                    if (other == UnInitIndex)
                    {
                        // Preserve the public center packet order. Excluded
                        // samples have a zero design row and no numeric input.
                        if (useSample)
                            boundaries(next, 0) = nextBoundary;
                        data->boundary[nextBoundary++] = {iCell, face, -1, point, ownNormal};
                    }
                    real fraction = 1;
                    if (other != UnInitIndex)
                    {
                        const real ownDistance = std::abs((point - center).dot(ownNormal)), otherDistance = std::abs((otherCenter - point).dot(ownNormal));
                        DNDS_check_throw_info(ownDistance + otherDistance > 0, "gradient face centers have zero normal separation");
                        fraction = method == GradientMethod::GGMP ? 0.5 : ownDistance / (ownDistance + otherDistance);
                        if (method == GradientMethod::GGMP && !metrics.Axisymmetric())
                            correction.noalias() -= areaNormal.head(dim) * (point - (center + otherCenter) * 0.5).head(dim).transpose();
                    }
                    initial.col(local) = areaNormal.head(dim) * fraction;
                    ++next;
                    ++local;
                }
            }
            if (metrics.Axisymmetric() && method != GradientMethod::LeastSquares)
            {
                // IMPORTANT: boundary samples are reference-face-center values,
                // whereas axisymmetric cell means use radius-weighted barycenters.
                // A vertical face over r in [0,h] has weighted center 2h/3,
                // not its reference center h/2. The continuum radial correction
                // alone therefore leaves a tangential gradient contamination
                // of +/-0.7/6 for f=3*x-0.7*r in the first boundary cell row.
                // Use the exact first moment of THIS discrete center stencil:
                // M = sum_f w_f * (x_sample-x_cell)^T, then grad=M^+*g_raw.
                // initial already includes each GG/GGMP interpolation fraction;
                // replacing correction here applies that moment exactly once.
                // Zero-area axis faces contribute zero, and the existing full
                // SVD policy handles any unconstrained direction unchanged.
                correction.setZero();
                for (index i = 0; i < count; ++i)
                    correction.noalias() += initial.col(i) * design.row(i);
            }
            Eigen::MatrixXd inverse;
            if (method == GradientMethod::LeastSquares)
            {
                const auto diagnostics = HardEigen::EigenSVDInverse(design, inverse, relativeCutoff);
                DNDS_check_throw_info(diagnostics.rank == dim, "least-squares gradient stencil is rank deficient after boundary exclusions");
                for (index i = 0; i < count; ++i)
                    initial.col(i) = inverse.col(i) * lengths[i];
            }
            else if (method == GradientMethod::GGMP || metrics.Axisymmetric())
            {
                HardEigen::EigenSVDInverse(correction, inverse, relativeCutoff);
                initial = (inverse * initial).eval();
            }
            for (index i = 0; i < count; ++i)
                for (int d = 0; d < 3; ++d)
                    weights(next - count + i, d) = d < dim ? initial(d, i) : 0;
            DNDS_assert(next == offsets(iCell + 1, 0) && nextBoundary == boundaryStarts(iCell + 1, 0)); });
        // Prove owned-only access once, using the actual prepared entries.
        // The hot core kernel needs neither ghost views nor row-mask checks.
        const auto &core = data->partition.Core();
        OMP::ForSetupRows(0, index(core.size()), [&](index iRow)
                          {
            const index iCell = core[iRow];
            for (rowsize slot = 0; slot < rowsize(view.CellFaces(iCell).size()); ++slot)
                DNDS_check_throw_info(view.CellFaceOther(iCell, view.CellFaces(iCell)[slot], slot) == UnInitIndex ||
                                          view.CellFaceOther(iCell, view.CellFaces(iCell)[slot], slot) < metrics.NumCell(),
                                      "gradient core stencil references a ghost input"); });
        return GradientPlan(std::move(data));
    }
}
