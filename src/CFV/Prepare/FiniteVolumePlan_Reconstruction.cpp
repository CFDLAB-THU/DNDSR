#include "CFV/Detail/ReconstructionData.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using namespace Geom;
    using namespace Geom::Elem;
    using Math::SmallMatrix;
    using Matrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;

    namespace
    {
        /// Only requested tables acquire writers. An omitted table has no Array
        /// allocation or lease, including on ranks with zero owned cells.
        struct OptionalTableWriter
        {
            std::optional<R::Array<real>::WriteLease> lease;
            R::ArrayView<real, false, R::UniformTableRows> values;
            explicit OptionalTableWriter(std::optional<R::Array<real>> &array)
            {
                if (array)
                {
                    lease.emplace(array->Write());
                    values = lease->View<R::UniformTableRows>();
                }
            }
        };
        template <class Derived>
        void PublishMatrix(R::ArrayView<real, false, R::UniformTableRows> target, index row,
                           const Eigen::MatrixBase<Derived> &matrix, bool triangle = false)
        {
            const index n = matrix.rows();
            if (!triangle && target.ElementStride() == 1)
                Eigen::Map<Matrix>(target[row].data(), n, n) = matrix;
            else if (triangle)
                for (index i = 0; i < n; ++i)
                    for (index j = 0; j <= i; ++j)
                        target(row, i * (i + 1) / 2 + j) = matrix(i, j);
            else
                for (index j = 0; j < n; ++j)
                    for (index i = 0; i < n; ++i)
                        target(row, i + n * j) = matrix(i, j);
        }
        /// Face scratch dependencies. Full support retains the original slot
        /// order; sweep-only setup omits the two secondary change-of-base maps.
        struct FaceSystemLayout
        {
            int a = -1, b = -1, mean = -1, secondary = -1, boundaryWeight = -1, width = 0;
            FaceSystemLayout(int n, const ReconstructionCapabilities &capabilities)
            {
                if (capabilities.sweeps || capabilities.linearSystem || capabilities.localPreconditioner || capabilities.inverseDiagnostics)
                {
                    a = width;
                    width += 2 * n * n;
                    // A's self-periodic diagonal subtracts B even for an
                    // inverse-only profile; it cannot be formed from A alone.
                    b = width;
                    width += n * n;
                }
                if (capabilities.sweeps || capabilities.linearSystem)
                {
                    mean = width;
                    width += 2 * n;
                }
                if (capabilities.wbap)
                {
                    secondary = width;
                    width += 2 * n * n;
                }
                if (capabilities.sweeps || capabilities.linearSystem)
                    boundaryWeight = width++;
            }
        };
    }

    VariationalReconstructionPlan VariationalReconstructionPlan::ConstructRecCoeff(FiniteVolumePlan metrics, ReconstructionSettings settings,
                                                                                   ReconstructionCapabilities capabilities,
                                                                                   Diagnostics::SetupProgress progress)
    {
        auto completeStage = progress.Stage("VR/complete");
        auto geometry = metrics._data;
        // Capability differences change payload construction and diagnostics
        // collectives. This is required protocol agreement, not optional safety.
        int modes[]{capabilities.Mask(), -capabilities.Mask()};
        MPI_Allreduce(MPI_IN_PLACE, modes, 2, MPI_INT, MPI_MIN, geometry->communication->Communicator());
        DNDS_check_throw_info(modes[0] == -modes[1], "VR preparation capabilities differ between ranks");
        std::shared_ptr<Detail::ReconstructionData> data;
        geometry->communication->LocalStage([&]
                                            {
            metrics.Validate();
            settings.Validate();
            const auto &storage = settings.storage;
            const bool rawSweep = capabilities.sweeps && storage.Raw();
            DNDS_check_throw_info(!storage.Raw() || (!capabilities.linearSystem && !capabilities.localPreconditioner),
                                  "experimental raw VR storage requires a sweep-only capability profile");
            DNDS_check_throw_info(settings.intOrder == geometry->intOrder, "VR quadrature differs from prepared metrics");
            auto meshRead = geometry->mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            data = std::make_shared<Detail::ReconstructionData>(metrics, settings, capabilities);
            const bool needInverse = capabilities.sweeps || capabilities.localPreconditioner || capabilities.inverseDiagnostics;
            data->workPartition.emplace(metrics.CellWorkPartition());
            const int n = data->n, dim = geometry->dim, degree = settings.maxOrder;
            const Math::TaylorBasis polynomial(dim, degree);
            const FaceSystemLayout faceSystem(n, capabilities);
            const index nCell = mesh.NumCell(), nFace = mesh.NumFaceProc();
            const auto parts = geometry->mesh->NLocalParts();
            std::vector<index> partitionStarts;
            if (parts > 0)
            {
                for (int part = 0; part < parts; ++part)
                    partitionStarts.push_back(geometry->mesh->LocalPartStart(part));
                partitionStarts.push_back(geometry->mesh->LocalPartEnd(parts - 1));
            }
            else
                partitionStarts = {0, nCell};
            DNDS_check_throw_info(partitionStarts.front() == 0 && partitionStarts.back() == nCell &&
                                  std::is_sorted(partitionStarts.begin(), partitionStarts.end()), "invalid VR host subpartition ranges");
            data->hostPartitionStarts = R::MetadataBuffer<index>(partitionStarts);
            data->coreFirstPosition.resize(nCell);
            index position = 0;
            for (auto iCell : data->workPartition->Core())
                data->coreFirstPosition[iCell] = position++;
            for (auto iCell : data->workPartition->HaloDependent())
                data->coreFirstPosition[iCell] = position++;
            DNDS_check_throw_info(position == nCell, "reconstruction work partition does not cover owned cells");
            auto cellRead = geometry->cell.Read();
            const auto cell = cellRead.View<R::FixedRows<8>>();
            auto faceOffsetRead = geometry->faceOffsets.Read();
            const auto faceOffset = faceOffsetRead.View<R::FixedRows<1>>();
            auto pointRead = geometry->faceQuadrature.Read();
            const auto point = pointRead.View<R::FixedRows<7>>();
            auto originalRead = geometry->mesh->CurrentNativeState()->ReadOriginalEntityIds(EntityKind::Cell);
            const auto original = originalRead->View<R::FixedRows<1>>();

            // Count rows in parallel, then publish deterministic stable offsets.
            std::vector<index> incidenceCounts(static_cast<size_t>(nCell));
            std::vector<index> boundaryCounts(static_cast<size_t>(nCell));
            std::vector<index> cellOffsets(static_cast<size_t>(nCell + 1));
            std::vector<uint8_t> active(size_t(nFace), 0); // local faces only
            index incidences = 0, boundaryCount = 0;
            {
            auto countStage = progress.Stage("VR/count-prefix");
            OMP::ForSetupRows(0, nCell, [&](index iCell)
            {
                const auto faces = mesh.CellFaces(iCell);
                incidenceCounts[size_t(iCell)] = index(faces.size());
                index boundaries = 0;
                for (index slot = 0; slot < faces.size(); ++slot)
                {
                    const index iFace = faces[slot];
                    if (mesh.CellFaceOther(iCell, iFace, rowsize(slot)) == UnInitIndex)
                        boundaries += faceOffset(iFace + 1, 0) - faceOffset(iFace, 0);
                }
                boundaryCounts[size_t(iCell)] = boundaries;
            });
            incidences = OMP::ExclusivePrefixRows(
                nCell, [&](index row) noexcept { return incidenceCounts[size_t(row)]; },
                [&](index row, index value) noexcept { cellOffsets[size_t(row)] = value; });
            boundaryCount = OMP::ExclusivePrefixRows(
                nCell, [&](index row) noexcept { return boundaryCounts[size_t(row)]; },
                [](index, index) noexcept {});
            OMP::ForSetupRows(0, nFace, [&](index iFace)
            {
                const auto parents = mesh.FaceCells(iFace);
                active[size_t(iFace)] = (parents[0] >= 0 && parents[0] < nCell) ||
                                        (parents[1] >= 0 && parents[1] < nCell);
            });
            }
            data->moments = R::Array<real>(R::RowLayout::Dense(mesh.NumCellProc(), n + 1));
            data->moments.SetName("CFV/quadrature/cellMoments", DNDS_ALLOCATION_SITE());
            data->offsets = R::Array<index>(R::RowLayout::Dense(nCell + 1, 1));
            data->faces = R::Array<index>(R::RowLayout::Dense(incidences, 1));
            data->neighbors = R::Array<index>(R::RowLayout::Dense(incidences, 1));
            data->boundaryOffsets = R::Array<index>(R::RowLayout::Dense(incidences + 1, 1));
            // Central operation-to-storage dependency table. Do not construct
            // full tables and discard them after setup: that retains peak cost.
            if (capabilities.sweeps && !rawSweep)
            {
                data->matrixAInvB.emplace(storage.Layout(incidences, n * n), R::Device::Host(), storage.Allocation());
                data->vectorAInvB.emplace(storage.Layout(incidences, n), R::Device::Host(), storage.Allocation());
                data->boundaryInjection.emplace(storage.Layout(boundaryCount, n), R::Device::Host(), storage.Allocation());
            }
            if (capabilities.wbap)
                data->secondary.emplace(R::RowLayout::Dense(incidences, n * n));
            if (capabilities.linearSystem)
            {
                data->localMatrix.emplace(R::RowLayout::Dense(nCell, n * n));
                data->rawMatrixB.emplace(R::RowLayout::Dense(incidences, n * n));
                data->rawVectorB.emplace(R::RowLayout::Dense(incidences, n));
                data->rawBoundaryInjection.emplace(R::RowLayout::Dense(boundaryCount, n));
            }
            if (rawSweep)
            {
                data->rawMatrixB.emplace(storage.Layout(storage.faceCoupling ? nFace : incidences, n * n), R::Device::Host(), storage.Allocation());
                data->rawVectorB.emplace(storage.Layout(incidences, n), R::Device::Host(), storage.Allocation());
                data->rawBoundaryInjection.emplace(storage.Layout(boundaryCount, n), R::Device::Host(), storage.Allocation());
                if (storage.faceCoupling)
                    data->rawFaceSide.emplace(R::RowLayout::Dense(incidences, 1));
            }
            if (capabilities.localPreconditioner || rawSweep)
                data->localInverse.emplace(rawSweep
                                              ? storage.Layout(nCell, storage.inverseTriangle ? n * (n + 1) / 2 : n * n)
                                              : R::RowLayout::Dense(nCell, n * n), R::Device::Host(), storage.Allocation());
            if (capabilities.inverseDiagnostics)
                data->inverseConditions.emplace(R::RowLayout::Dense(nCell, 4));
            data->neighborRotation = R::Array<real>(R::RowLayout::Dense(incidences, 9));
            data->neighborRotationActive = R::Array<uint8_t>(R::RowLayout::Dense(incidences, 1));
            auto rotationWrite = data->neighborRotation.Write();
            auto rotationActiveWrite = data->neighborRotationActive.Write();
            const auto rotations = rotationWrite.View<R::FixedRows<9>>();
            const auto rotationActive = rotationActiveWrite.View<R::FixedRows<1>>();
            OptionalTableWriter rawMatrixWrite(data->rawMatrixB), rawVectorWrite(data->rawVectorB), rawInjectionWrite(data->rawBoundaryInjection);
            const auto rawMatrices = rawMatrixWrite.values, rawVectors = rawVectorWrite.values, rawInjections = rawInjectionWrite.values;
            OptionalTableWriter localMatrixWrite(data->localMatrix), localInverseWrite(data->localInverse);
            const auto localMatrices = localMatrixWrite.values, localInverses = localInverseWrite.values;
            std::optional<R::Array<uint8_t>::WriteLease> faceSideWrite;
            R::ArrayView<uint8_t, false, R::FixedRows<1>> faceSides;
            if (data->rawFaceSide)
            {
                faceSideWrite.emplace(data->rawFaceSide->Write());
                faceSides = faceSideWrite->View<R::FixedRows<1>>();
            }
            // A failed symmetry check is merged after the setup team; never
            // throw out of an OpenMP numerical row or silently symmetrize it.
            std::vector<uint8_t> inverseSymmetry(size_t(rawSweep && storage.inverseTriangle ? nCell : 0), 1);
            data->preparationRuntimeScratchBytes += inverseSymmetry.size();
            data->boundaryPoints = R::Array<index>(R::RowLayout::Dense(boundaryCount, 4));
            // Only requested face-system blocks exist. This is preparation
            // scratch, released before returning the immutable plan.
            std::optional<R::Array<real>> systems;
            if (faceSystem.width)
            {
                systems.emplace(R::RowLayout::Dense(nFace, faceSystem.width));
                data->preparationRuntimeScratchBytes += systems->Storage().Bytes();
            }
            auto momentWrite = data->moments.Write();
            const auto moment = momentWrite.View<R::TableRows>();
            const bool needHalfBoxes = faceSystem.width &&
                settings.functional.scaleType == ReconstructionFunctionalSettings::LengthScale::MeanAACBB;
            R::Array<real> halfBoxStorage(R::RowLayout::Dense(needHalfBoxes ? mesh.NumCellProc() : 0, 3));
            halfBoxStorage.SetName("CFV/setup/halfBoxes", DNDS_ALLOCATION_SITE());
            data->preparationRuntimeScratchBytes += halfBoxStorage.Storage().Bytes();
            auto halfBoxWrite = halfBoxStorage.Write();
            const auto halfBoxes = halfBoxWrite.View<R::FixedRows<3>>();
            {
            auto setupStage = progress.Stage("VR/moments");
            OMP::ForSetupRows(0, mesh.NumCellProc(), [&](index iCell)
            {
                SmallMatrix full;
                tSmallCoords coords;
                std::fill(moment[iCell].begin(), moment[iCell].end(), 0.);
                mesh.GetCoordsOnCell(iCell, coords);
                const tPoint bary{cell(iCell, 4), cell(iCell, 5), cell(iCell, 6)};
                if (needHalfBoxes)
                    Eigen::Map<tPoint>(halfBoxes[iCell].data()) = (coords.colwise() - bary).array().abs().rowwise().maxCoeff();
                const auto element = mesh.GetCellElement(iCell);
                const Quadrature q(element, INT_ORDER_MAX);
                const tPoint center{cell(iCell, 1), cell(iCell, 2), cell(iCell, 3)};
                for (int iG = 0; iG < q.GetNumPoints(); ++iG)
                {
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, INT_ORDER_MAX, iG);
                    const tPoint p = PPhysicsCoordD01Nj(coords, shape);
                    polynomial.Derivatives(full, p, center, cell(iCell, 7));
                    const real weight = CellJacobianDet(dim, coords, shape) * q.GetWeight(iG) * (geometry->axisymmetric ? p[1] : 1.);
                    for (int j = 0; j <= n; ++j) moment(iCell, j) += full(0, j) * weight;
                }
                const real volume = moment(iCell, 0);
                for (int j = 0; j <= n; ++j) moment(iCell, j) /= volume;
            });
            }
            OptionalTableWriter systemWrite(systems);
            const auto system = systemWrite.values;
            auto basis = [&](index iCell, const tPoint &p, SmallMatrix &out)
            {
                polynomial.Reconstruction(out, p, {cell(iCell, 1), cell(iCell, 2), cell(iCell, 3)}, cell(iCell, 7), moment[iCell], degree);
            };
            {
            auto setupStage = progress.Stage("VR/face-systems");
            OMP::ForSetupRows(0, nFace, [&](index iFace)
            {
                if (!active[size_t(iFace)]) return;
                if (!faceSystem.width) return;
                SmallMatrix left, right, weighted, aL, aR, bLR, secondaryL, secondaryR;
                tSmallCoords coords;
                if (systems) std::fill(system[iFace].begin(), system[iFace].end(), 0.);
                const auto parents = mesh.FaceCells(iFace);
                const index iL = parents[0], iR = parents[1];
                DNDS_check_throw_info(iL >= 0 && iL < mesh.NumCellProc() &&
                                          (iR == UnInitIndex || (iR >= 0 && iR < mesh.NumCellProc())), "VR stencil lacks face parent closure");
                mesh.GetCoordsOnFace(iFace, coords);
                const tPoint midpoint = PPhysicsCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(mesh.GetFaceElement(iFace).type, 1, 0));
                const auto jac = ShapeJacobianCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(mesh.GetFaceElement(iFace).type, 1, 0));
                tPoint normal = dim == 2 ? FacialJacobianToNormVec<2>(jac) : FacialJacobianToNormVec<3>(jac);
                normal.normalize();
                const tPoint baryL{cell(iL, 4), cell(iL, 5), cell(iL, 6)};
                const real distance = iR == UnInitIndex ? std::abs(2 * (midpoint - baryL).dot(normal))
                    : (Detail::OtherPointInCell(mesh, iFace, 0, {cell(iR, 4), cell(iR, 5), cell(iR, 6)}) - baryL).norm();
                DNDS_check_throw_info(distance > 0 && std::isfinite(distance), "VR face has zero barycenter distance");
                real area = 0;
                for (index iQ = faceOffset(iFace, 0); iQ < faceOffset(iFace + 1, 0); ++iQ) area += point(iQ, 6);
                if (geometry->axisymmetric)
                {
                    // Weight policies use physical face length, not r*length.
                    // In particular an axis face has zero radial measure but
                    // nonzero geometric length; no division by radius occurs.
                    area = 0;
                    const auto element = mesh.GetFaceElement(iFace);
                    const Quadrature quadrature(element, geometry->intOrder);
                    for (int iG = 0; iG < quadrature.GetNumPoints(); ++iG)
                    {
                        const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, geometry->intOrder, iG);
                        area += FacialJacobianToNormVec<2>(ShapeJacobianCoordD01Nj(coords, shape)).norm() * quadrature.GetWeight(iG);
                    }
                }
                auto functional = settings.functional;
                if (settings.HQMWeights)
                {
                    functional.dirWeightScheme = ReconstructionFunctionalSettings::DirectionalWeight::HQM_OPT;
                    functional.geomWeightScheme = ReconstructionFunctionalSettings::GeometricWeight::HQM_SD;
                    functional.geomWeightBias = 1;
                }
                // Tangential face spans for TEST_OPT; no cell-anisotropy policy
                // is implied by these surface-only principal directions.
                tPoint faceSpan = tPoint::Zero();
                if (dim == 3 && degree == 3 && functional.dirWeightScheme == ReconstructionFunctionalSettings::DirectionalWeight::TEST_OPT)
                {
                    faceSpan = Geom::Elem::GetElemNodeMajorSpan(coords);
                }
                auto weights = Math::ReconstructionOrderWeights(functional, dim, degree, area, distance, faceSpan);
                if (iR == UnInitIndex)
                {
                    const auto override = settings.boundaryZoneWeights.find(mesh.GetFaceZone(iFace));
                    const auto &boundaryWeights = override == settings.boundaryZoneWeights.end() ? settings.boundaryDerivativeWeights : override->second;
                    for (int order = 0; order <= degree; ++order)
                        weights[order] *= settings.bcWeight * boundaryWeights[order];
                }
                real diagonal[20];
                real length = distance;
                if (functional.scaleType == ReconstructionFunctionalSettings::LengthScale::MeanAACBB)
                {
                    const tPoint boxL = Eigen::Map<const tPoint>(halfBoxes[iL].data());
                    const tPoint box = iR == UnInitIndex ? boxL : (.5 * (boxL + Eigen::Map<const tPoint>(halfBoxes[iR].data()))).eval();
                    length = std::sqrt(box.head(dim).squaredNorm() / dim);
                }
                if (functional.scaleType == ReconstructionFunctionalSettings::LengthScale::CellMax)
                    length = iR == UnInitIndex ? cell(iL, 7) : std::max(cell(iL, 7), cell(iR, 7));
                Math::FunctionalWeights(diagonal, dim, degree, length * functional.scaleMultiplier, weights.data(), functional.dirWeightCombPowV);
                const tGPoint rightToLeft = Detail::OtherFrameInCell(mesh, iFace, 0);
                const bool rotated = !rightToLeft.isApprox(tGPoint::Identity(), 1e-14);
                DNDS_check_throw_info(faceSystem.a < 0 || !rotated || functional.dirWeightCombPowV == 1,
                                      "rotated periodic VR requires the Frobenius derivative metric");
                if (faceSystem.boundaryWeight >= 0) system(iFace, faceSystem.boundaryWeight) = diagonal[0];
                if (faceSystem.a >= 0) aL.setZero(n, n), aR.setZero(n, n);
                if (faceSystem.b >= 0) bLR.setZero(n, n);
                Eigen::Map<Eigen::VectorXd> bL(faceSystem.mean >= 0 ? system[iFace].data() + faceSystem.mean : nullptr, faceSystem.mean >= 0 ? n : 0);
                Eigen::Map<Eigen::VectorXd> bR(faceSystem.mean >= 0 ? system[iFace].data() + faceSystem.mean + n : nullptr, faceSystem.mean >= 0 ? n : 0);
                for (index iQ = faceOffset(iFace, 0); iQ < faceOffset(iFace + 1, 0); ++iQ)
                {
                    const tPoint p{point(iQ, 0), point(iQ, 1), point(iQ, 2)};
                    basis(iL, p, left);
                    if (faceSystem.a >= 0)
                    {
                        weighted = left;
                        for (int j = 0; j <= n; ++j) weighted.row(j) *= diagonal[j] * point(iQ, 6);
                        aL.noalias() += left.transpose() * weighted;
                    }
                    if (faceSystem.mean >= 0)
                        bL.noalias() += left.row(0).transpose() * (diagonal[0] * point(iQ, 6));
                    if (iR != UnInitIndex)
                    {
                        basis(iR, Detail::FacePointInCell(mesh, iFace, 1, p), right);
                        if (rotated) polynomial.Pullback(right, rightToLeft.transpose());
                        if (faceSystem.b >= 0) bLR.noalias() += weighted.transpose() * right;
                        if (faceSystem.a >= 0)
                        {
                            weighted = right;
                            for (int j = 0; j <= n; ++j) weighted.row(j) *= diagonal[j] * point(iQ, 6);
                            aR.noalias() += right.transpose() * weighted;
                        }
                        if (faceSystem.mean >= 0)
                            bR.noalias() += right.row(0).transpose() * (diagonal[0] * point(iQ, 6));
                    }
                }
                if (faceSystem.a >= 0)
                {
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.a, n, n) = aL;
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.a + n * n, n, n) = aR;
                }
                if (faceSystem.b >= 0)
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.b, n, n) = bLR;
                if (rawSweep && storage.faceCoupling)
                    PublishMatrix(rawMatrices, iFace, bLR);
                if (capabilities.wbap && iR != UnInitIndex)
                {
                    const tPoint centerL{cell(iL, 1), cell(iL, 2), cell(iL, 3)}, centerR{cell(iR, 1), cell(iR, 2), cell(iR, 3)};
                    basis(iR, Detail::OtherPointInCell(mesh, iFace, 1, centerL), right);
                    if (rotated) polynomial.Pullback(right, rightToLeft.transpose());
                    basis(iL, Detail::OtherPointInCell(mesh, iFace, 0, centerR), left);
                    if (rotated) polynomial.Pullback(left, rightToLeft);
                    secondaryL = right.bottomRows(n), secondaryR = left.bottomRows(n);
                    // Exact Taylor change of center/scale/frame, not a local
                    // numerical inversion: c_alpha=s^|alpha|*D^alpha/alpha!.
                    // No spectral cutoff belongs to this coordinate map.
                    for (int mode = 1; mode <= n; ++mode)
                    {
                        const auto &alpha = polynomial.MultiIndex(mode);
                        const real factorial = Geom::Base::factorials[alpha[0]] * Geom::Base::factorials[alpha[1]] * Geom::Base::factorials[alpha[2]];
                        secondaryL.row(mode - 1) *= std::pow(cell(iL, 7), polynomial.Degree(mode)) / factorial;
                        secondaryR.row(mode - 1) *= std::pow(cell(iR, 7), polynomial.Degree(mode)) / factorial;
                    }
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.secondary, n, n) = secondaryL;
                    Eigen::Map<Matrix>(system[iFace].data() + faceSystem.secondary + n * n, n, n) = secondaryR;
                }
            });
            }
            auto offsetsWrite = data->offsets.Write(), neighborsWrite = data->neighbors.Write(), facesWrite = data->faces.Write();
            auto bOffsetsWrite = data->boundaryOffsets.Write(), bPointsWrite = data->boundaryPoints.Write();
            OptionalTableWriter matrixWrite(data->matrixAInvB), vectorWrite(data->vectorAInvB), secondaryWrite(data->secondary), injectionWrite(data->boundaryInjection);
            const auto offsets = offsetsWrite.View<R::FixedRows<1>>(), neighbors = neighborsWrite.View<R::FixedRows<1>>(), faces = facesWrite.View<R::FixedRows<1>>();
            const auto bOffsets = bOffsetsWrite.View<R::FixedRows<1>>();
            const auto bPoints = bPointsWrite.View<R::FixedRows<4>>();
            const auto matrices = matrixWrite.values, vectors = vectorWrite.values, secondaries = secondaryWrite.values, injections = injectionWrite.values;
            {
            auto prefixStage = progress.Stage("VR/prefix");
            std::vector<index> entryBoundaryCounts(static_cast<size_t>(incidences));
            OMP::ForSetupRows(0, nCell, [&](index iCell)
            {
                offsets(iCell, 0) = cellOffsets[size_t(iCell)];
                index iEntry = cellOffsets[size_t(iCell)];
                const auto c2f = mesh.CellFaces(iCell);
                for (index slot = 0; slot < c2f.size(); ++slot, ++iEntry)
                {
                    const auto iFace = c2f[slot];
                    entryBoundaryCounts[size_t(iEntry)] =
                        mesh.CellFaceOther(iCell, iFace, rowsize(slot)) == UnInitIndex
                            ? faceOffset(iFace + 1, 0) - faceOffset(iFace, 0)
                            : 0;
                }
            });
            offsets(nCell, 0) = incidences;
            const index publishedBoundaryCount = OMP::ExclusivePrefixRows(
                incidences, [&](index row) noexcept { return entryBoundaryCounts[size_t(row)]; },
                [&](index row, index value) noexcept { bOffsets(row, 0) = value; });
            DNDS_check_throw_info(publishedBoundaryCount == boundaryCount, "VR setup incidence count changed");
            }
            OptionalTableWriter conditionsWrite(data->inverseConditions);
            const auto conditions = conditionsWrite.values;
            data->inverseDiagnostics.cells = nCell;
            {
            auto setupStage = progress.Stage("VR/SVD-matrices");
            OMP::ForSetupRows(0, nCell, [&](index iCell)
            {
                // Setup scratch and Eigen decompositions belong to this cell.
                SmallMatrix A, inverse, left, rawMatrixScratch(n, n);
                Eigen::VectorXd rawVectorScratch(n);
                index iEntry = offsets(iCell, 0), iBoundary = bOffsets(iEntry, 0);
                A.setZero(n, n);
                const auto c2f = mesh.CellFaces(iCell);
                for (index slot = 0; faceSystem.a >= 0 && slot < c2f.size(); ++slot)
                {
                    const index iFace = c2f[slot];
                    const int side = mesh.CellIsFaceBack(iCell, iFace, rowsize(slot)) ? 0 : 1;
                    A += Eigen::Map<const Matrix>(system[iFace].data() + faceSystem.a + side * n * n, n, n);
                    if (mesh.CellFaceOther(iCell, iFace, rowsize(slot)) == iCell &&
                        Detail::OtherFrameInCell(mesh, iFace, side).isApprox(tGPoint::Identity(), 1e-14))
                    {
                        const Eigen::Map<const Matrix> B(system[iFace].data() + faceSystem.b, n, n);
                        if (side == 0) A -= B; else A -= B.transpose();
                    }
                }
                HardEigen::SVDInverseDiagnostics spectrum{};
                if (needInverse)
                    spectrum = HardEigen::EigenSVDInverse(A, inverse, settings.svdTolerance, settings.svdAbsoluteTolerance);
                if (capabilities.linearSystem)
                    Eigen::Map<Matrix>(localMatrices[iCell].data(), n, n) = A;
                if (capabilities.localPreconditioner || rawSweep)
                {
                    if (rawSweep && storage.inverseTriangle)
                    {
                        // Relative to this inverse's own scale, with no unit
                        // floor: a zero inverse requires exactly zero asymmetry.
                        const real norm = inverse.cwiseAbs().maxCoeff();
                        const real asymmetry = (inverse - inverse.transpose()).cwiseAbs().maxCoeff();
                        inverseSymmetry[size_t(iCell)] = inverse.allFinite() &&
                            asymmetry <= 128 * n * std::numeric_limits<real>::epsilon() * norm;
                    }
                    PublishMatrix(localInverses, iCell, inverse, rawSweep && storage.inverseTriangle);
                }
                if (capabilities.inverseDiagnostics)
                {
                    conditions(iCell, 0) = spectrum.rank;
                    conditions(iCell, 1) = spectrum.rawCondition;
                    conditions(iCell, 2) = spectrum.retainedCondition;
                    conditions(iCell, 3) = spectrum.relativeCutoff;
                }
                for (index slot = 0; slot < c2f.size(); ++slot, ++iEntry)
                {
                    const index iFace = c2f[slot], other = mesh.CellFaceOther(iCell, iFace, rowsize(slot));
                    const int side = mesh.CellIsFaceBack(iCell, iFace, rowsize(slot)) ? 0 : 1;
                    faces(iEntry, 0) = iFace, neighbors(iEntry, 0) = other;
                    if (rawSweep && storage.faceCoupling)
                        faceSides(iEntry, 0) = uint8_t(side);
                    const tGPoint rotation = Detail::OtherFrameInCell(mesh, iFace, side);
                    Eigen::Map<tGPoint>(rotations[iEntry].data()) = rotation;
                    rotationActive(iEntry, 0) = !rotation.isApprox(tGPoint::Identity(), 1e-14);
                    if (capabilities.wbap)
                    {
                        auto secondary = Eigen::Map<Matrix>(secondaries[iEntry].data(), n, n);
                        secondary.setZero();
                        if (other != UnInitIndex)
                            secondary = Eigen::Map<const Matrix>(system[iFace].data() + faceSystem.secondary + side * n * n, n, n);
                    }
                    if (capabilities.sweeps || capabilities.linearSystem)
                    {
                        // Retain originals only when requested. A bounded local
                        // scratch block gives sweeps the same Map/product shape;
                        // it is never reconstructed from a truncated inverse.
                        auto rawM = Eigen::Map<Matrix>(capabilities.linearSystem ? rawMatrices[iEntry].data() : rawMatrixScratch.data(), n, n);
                        auto rawB = Eigen::Map<Eigen::VectorXd>(capabilities.linearSystem ? rawVectors[iEntry].data() : rawVectorScratch.data(), n);
                        rawM.setZero(), rawB.setZero();
                        if (other != UnInitIndex)
                        {
                            const Eigen::Map<const Matrix> B(system[iFace].data() + faceSystem.b, n, n);
                            if (other != iCell || !rotation.isApprox(tGPoint::Identity(), 1e-14))
                            {
                                if (side == 0) rawM = B; else rawM = B.transpose();
                            }
                            rawB = Eigen::Map<const Eigen::VectorXd>(system[iFace].data() + faceSystem.mean + side * n, n);
                        }
                        if (rawSweep)
                        {
                            if (!storage.faceCoupling)
                                PublishMatrix(rawMatrices, iEntry, rawM);
                            for (int i = 0; i < n; ++i)
                                rawVectors(iEntry, i) = rawB[i];
                        }
                        else if (capabilities.sweeps)
                        {
                            if (matrices.ElementStride() == 1 && vectors.ElementStride() == 1)
                            {
                                auto M = Eigen::Map<Matrix>(matrices[iEntry].data(), n, n);
                                auto b = Eigen::Map<Eigen::VectorXd>(vectors[iEntry].data(), n);
                                M.setZero(), b.setZero();
                                if (other != UnInitIndex)
                                {
                                    if (other != iCell || !rotation.isApprox(tGPoint::Identity(), 1e-14))
                                        M.noalias() = inverse * rawM;
                                    b.noalias() = inverse * rawB;
                                }
                            }
                            else
                            {
                                SmallMatrix product;
                                Eigen::VectorXd productVector(n);
                                product.setZero(n, n), productVector.setZero();
                                if (other != UnInitIndex)
                                {
                                    if (other != iCell || !rotation.isApprox(tGPoint::Identity(), 1e-14))
                                        product.noalias() = inverse * rawM;
                                    productVector.noalias() = inverse * rawB;
                                }
                                PublishMatrix(matrices, iEntry, product);
                                for (int i = 0; i < n; ++i) vectors(iEntry, i) = productVector[i];
                            }
                        }
                    }
                    if (other == UnInitIndex)
                        for (index iQ = faceOffset(iFace, 0); iQ < faceOffset(iFace + 1, 0); ++iQ, ++iBoundary)
                        {
                            if (capabilities.sweeps || capabilities.linearSystem)
                            {
                                basis(iCell, {point(iQ, 0), point(iQ, 1), point(iQ, 2)}, left);
                                auto rawInjection = Eigen::Map<Eigen::VectorXd>(capabilities.linearSystem ? rawInjections[iBoundary].data() : rawVectorScratch.data(), n);
                                rawInjection = left.row(0).transpose() * (system(iFace, faceSystem.boundaryWeight) * point(iQ, 6));
                                if (rawSweep)
                                    for (int i = 0; i < n; ++i)
                                        rawInjections(iBoundary, i) = rawInjection[i];
                                else if (capabilities.sweeps)
                                {
                                    if (injections.ElementStride() == 1)
                                        Eigen::Map<Eigen::VectorXd>(injections[iBoundary].data(), n).noalias() = inverse * rawInjection;
                                    else
                                    {
                                        const Eigen::VectorXd product = inverse * rawInjection;
                                        for (int i = 0; i < n; ++i) injections(iBoundary, i) = product[i];
                                    }
                                }
                            }
                            bPoints(iBoundary, 0) = iCell, bPoints(iBoundary, 1) = iFace;
                            bPoints(iBoundary, 2) = iQ - faceOffset(iFace, 0), bPoints(iBoundary, 3) = iQ;
                        }
                }
            });
            }
            DNDS_check_throw_info(std::find(inverseSymmetry.begin(), inverseSymmetry.end(), uint8_t(0)) == inverseSymmetry.end(),
                                  "local VR inverse is not symmetric to the packed-storage tolerance; full storage is required");
            // Deterministic controller merge after the team. These are setup
            // diagnostics, not atomics inside each inverse/face arithmetic path.
            for (index i = 0; i < incidences; ++i)
                data->hasRotatedPeriodicity |= bool(rotationActive(i, 0));
            if (capabilities.inverseDiagnostics)
                for (index iCell = 0; iCell < nCell; ++iCell)
                {
                    auto &summary = data->inverseDiagnostics;
                    const auto rank = index(conditions(iCell, 0));
                    const auto condition = conditions(iCell, 1);
                    summary.filteredCells += rank < n;
                    summary.discardedModes += n - rank;
                    summary.rankZeroCells += rank == 0;
                    ++summary.rankCounts[size_t(rank)];
                    if (condition > summary.maxCondition ||
                        (condition == summary.maxCondition && original(iCell, 0) < summary.maxConditionCell))
                    {
                        summary.maxCondition = condition;
                        summary.maxConditionCell = original(iCell, 0);
                    }
                } });
        // The matrix assembly's full derivative scratch and mutable moments
        // leases are gone. Hot sampling prepares only its requested rows and
        // never retains the full-order setup matrices.
        {
            auto setupStage = progress.Stage("VR/face-basis");
            geometry->communication->LocalStage([&]
                                                { Detail::PrepareFaceBasis(*data, *geometry); });
        }
        if (capabilities.inverseDiagnostics)
        {
            auto &summary = data->inverseDiagnostics;
            index local[]{summary.cells, summary.filteredCells, summary.discardedModes, summary.rankZeroCells}, global[4]{};
            MPI::Allreduce(local, global, 4, DNDS_MPI_INDEX, MPI_SUM, geometry->mesh->getMPI().comm);
            real condition = 0;
            MPI::Allreduce(&summary.maxCondition, &condition, 1, DNDS_MPI_REAL, MPI_MAX, geometry->mesh->getMPI().comm);
            const index localCell = summary.maxCondition == condition && summary.maxConditionCell != UnInitIndex ? summary.maxConditionCell : INT64_MAX;
            index globalCell = INT64_MAX;
            MPI::Allreduce(&localCell, &globalCell, 1, DNDS_MPI_INDEX, MPI_MIN, geometry->mesh->getMPI().comm);
            if (globalCell == INT64_MAX)
                globalCell = UnInitIndex;
            std::array<index, 20> ranks{};
            MPI::Allreduce(summary.rankCounts.data(), ranks.data(), int(ranks.size()), DNDS_MPI_INDEX, MPI_SUM, geometry->mesh->getMPI().comm);
            summary = {global[0], global[1], global[2], global[3], condition, globalCell, ranks};
        }
        return VariationalReconstructionPlan(std::move(data));
    }
}
