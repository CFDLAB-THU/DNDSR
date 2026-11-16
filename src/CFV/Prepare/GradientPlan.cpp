#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    GradientPlan GradientPlan::Prepare(Geom::UnstructuredMesh &mesh, FiniteVolumePlan metrics,
                                       GradientMethod method, real relativeCutoff)
    {
        namespace R = Runtime;
        using namespace Geom;
        DNDS_check_throw_info(method == GradientMethod::GreenGauss || method == GradientMethod::LeastSquares || method == GradientMethod::GGMP,
                              "unknown FV gradient method");
        DNDS_check_throw_info(std::isfinite(relativeCutoff) && relativeCutoff >= 0, "invalid gradient cutoff");
        auto metricRead = metrics.AcquireRead();
        auto meshRead = mesh.AcquireRead(MeshReadRequirements::CellFlux());
        DNDS_check_throw_info(mesh.Entities(EntityKind::Cell).SameOrder(metrics.Cells()), "gradient metrics and mesh orders differ");
        const auto view = meshRead.HostView();
        const auto geometry = metricRead.HostView();
        const int dim = metrics.Dimension();
        auto data = std::make_shared<Data>(metrics);
        // Prefix each cell's incidence and physical-boundary ranges before
        // filling. Per-cell SVD setup then writes disjoint preallocated rows;
        // neither boundary ordering nor incidence summation depends on OMP.
        data->offsets = R::Array<index>(R::RowLayout::Dense(metrics.NumCell() + 1, 1));
        R::Array<index> boundaryOffsets(R::RowLayout::Dense(metrics.NumCell() + 1, 1));
        auto offsetWrite = data->offsets.Write(), boundaryOffsetWrite = boundaryOffsets.Write();
        const auto offsets = offsetWrite.View<R::FixedRows<1>>(), boundaryStarts = boundaryOffsetWrite.View<R::FixedRows<1>>();
        index entries = 0, boundaryCount = 0;
        for (index iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            offsets(iCell, 0) = entries;
            boundaryStarts(iCell, 0) = boundaryCount;
            for (index slot = 0; slot < view.CellFaces(iCell).size(); ++slot)
            {
                const index face = view.CellFaces(iCell)[slot];
                ++entries;
                if (view.CellFaceOther(iCell, face, rowsize(slot)) == UnInitIndex)
                    ++boundaryCount;
            }
        }
        offsets(metrics.NumCell(), 0) = entries;
        boundaryStarts(metrics.NumCell(), 0) = boundaryCount;
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
            for (index iRow = 0; iRow < index(rows.size()); ++iRow)
                view(iRow, 0) = rows[iRow];
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
            if (metrics.Axisymmetric())
                for (int d = 0; d < dim; ++d)
                    correction(1, d) += geometry.radialMoment(iCell, d);
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
                    const real quadratureScale = std::sqrt(area);
                    design.row(local) = displacement.head(dim).transpose() / length * quadratureScale;
                    lengths[local] = quadratureScale / length;
                    boundaries(next, 0) = UnInitIndex;
                    if (other == UnInitIndex)
                    {
                        boundaries(next, 0) = nextBoundary;
                        data->boundary[nextBoundary++] = {iCell, face, -1, point, ownNormal};
                    }
                    real fraction = 1;
                    if (other != UnInitIndex)
                    {
                        const real ownDistance = std::abs((point - center).dot(ownNormal)), otherDistance = std::abs((otherCenter - point).dot(ownNormal));
                        DNDS_check_throw_info(ownDistance + otherDistance > 0, "gradient face centers have zero normal separation");
                        fraction = method == GradientMethod::GGMP ? 0.5 : ownDistance / (ownDistance + otherDistance);
                        if (method == GradientMethod::GGMP)
                            correction.noalias() -= areaNormal.head(dim) * (point - (center + otherCenter) * 0.5).head(dim).transpose();
                    }
                    initial.col(local) = areaNormal.head(dim) * fraction;
                    ++next;
                    ++local;
                }
            }
            Eigen::MatrixXd inverse;
            if (method == GradientMethod::LeastSquares)
            {
                HardEigen::EigenSVDInverse(design, inverse, relativeCutoff);
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
        for (auto iCell : data->partition.Core())
            for (rowsize slot = 0; slot < rowsize(view.CellFaces(iCell).size()); ++slot)
                DNDS_check_throw_info(view.CellFaceOther(iCell, view.CellFaces(iCell)[slot], slot) == UnInitIndex ||
                                          view.CellFaceOther(iCell, view.CellFaces(iCell)[slot], slot) < metrics.NumCell(),
                                      "gradient core stencil references a ghost input");
        return GradientPlan(std::move(data));
    }
}
