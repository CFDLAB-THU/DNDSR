#pragma once
#include "DNDS/Device/DeviceStorage.hpp"
#include "DNDS/Device/DeviceTransferable.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "CFV/Legacy/FiniteVolumeSettings.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "CFV/Legacy/VRDefines.hpp"
#include "Geom/Math/DiffTensors.hpp"
#include "CFV/Legacy/DOFFactory.hpp"

#include "CFV/Legacy/FiniteVolume_DeviceView.hpp"

namespace DNDS::CFV
{
    class FiniteVolume : public DeviceTransferable<FiniteVolume>
    {
        friend class DeviceTransferable<FiniteVolume>;

    public:
        MPI_int mRank{0};
        MPIInfo mpi;
        ssp<Geom::UnstructuredMesh> mesh;

    protected:
        FiniteVolumeSettings settings;
        Runtime::VersionStamp _metricsMeshStamp;
        std::weak_ptr<Geom::UnstructuredMesh> _metricsMesh;
        Runtime::VersionStamp _metricsBuildStamp;
        std::weak_ptr<Geom::UnstructuredMesh> _metricsBuildMesh;
        uint32_t _metricsStages = 0;
        uint64_t _metricsGeneration = 0;
        enum MetricsStage : uint32_t
        {
            CellAttributes = 1u << 0,
            CellVolume = 1u << 1,
            CellBarycenter = 1u << 2,
            CellCenter = 1u << 3,
            CellJacobian = 1u << 4,
            CellPoints = 1u << 5,
            CellBox = 1u << 6,
            CellInertia = 1u << 7,
            FaceAttributes = 1u << 8,
            FaceCenter = 1u << 9,
            FaceArea = 1u << 10,
            FaceJacobian = 1u << 11,
            FacePoints = 1u << 12,
            FaceNormals = 1u << 13,
            FaceMeanNormals = 1u << 14,
            CellSmoothing = 1u << 15,
            AllMetrics = (1u << 16) - 1
        };

        static Runtime::Change MeshDependencyMask()
        {
            return Runtime::Change::Topology | Runtime::Change::Geometry |
                   Runtime::Change::Distribution | Runtime::Change::Layout;
        }
        void InvalidateMetricsDependencies()
        {
            _metricsMeshStamp = Runtime::VersionStamp{};
            _metricsMesh.reset();
        }
        void PublishMetricsDependencies()
        {
            _metricsMeshStamp = mesh->versions.Stamp(MeshDependencyMask());
            _metricsMesh = mesh;
            ++_metricsGeneration;
        }
        void BeginMetricsConstruction()
        {
            InvalidateMetricsDependencies();
            _metricsBuildStamp = mesh->versions.Stamp(MeshDependencyMask());
            _metricsBuildMesh = mesh;
            _metricsStages = 0;
        }
        uint32_t MetricsPrerequisites(MetricsStage stage) const
        {
            switch (stage)
            {
            case CellBarycenter:
            case CellJacobian:
            case CellPoints:
                return CellAttributes | CellVolume;
            case CellBox:
                return CellAttributes | CellBarycenter;
            case CellInertia:
                return CellBarycenter | CellVolume;
            case FaceArea:
                return FaceAttributes | (axisSymmetric ? uint32_t(CellVolume) : 0u);
            case FaceJacobian:
            case FacePoints:
            case FaceNormals:
                return FaceAttributes;
            case FaceMeanNormals:
                return FaceAttributes | FaceCenter | CellCenter;
            case CellSmoothing:
                return CellVolume | FaceArea;
            default:
                return 0;
            }
        }
        void BeginMetricsStage(MetricsStage stage)
        {
            InvalidateMetricsDependencies();
            // Independent granular constructors keep their legacy getter use.
            // A new epoch starts an incomplete publication, never a valid full
            // bundle assembled from old same-shaped fields.
            if (_metricsBuildMesh.lock() != mesh || !_metricsBuildStamp.Valid())
                BeginMetricsConstruction();
            // Rebuilding an input invalidates every previously completed
            // dependent stage, including indirect consumers. Failed stages
            // cannot leave their old dependent bundle publishable.
            uint32_t invalid = uint32_t(stage), previous;
            do
            {
                previous = invalid;
                for (uint32_t bit = 1; bit <= CellSmoothing; bit <<= 1)
                    if (MetricsPrerequisites(MetricsStage(bit)) & invalid)
                        invalid |= bit;
            } while (previous != invalid);
            _metricsStages &= ~invalid;
            const auto required = MetricsPrerequisites(stage);
            DNDS_check_throw_info((_metricsStages & required) == required,
                                  "metric stage requires its prerequisite stages in the current construction epoch");
        }
        void EndMetricsStage(MetricsStage stage) { _metricsStages |= uint32_t(stage); }

    public:
        /// Validate once at an operator boundary, with its mesh read lease held.
        /// Geometry accessors inside numerical kernels remain plain lookups.
        void ValidateMetricsDependencies() const
        {
            DNDS_check_throw_info(_metricsMesh.lock() == mesh && _metricsMeshStamp.Valid(),
                                  "finite-volume metrics are missing or stale; rebuild ConstructMetrics");
        }
        void ValidateMeshDependencies() const { ValidateMetricsDependencies(); }
        /// Cache publication identity, including a rebuild in the same mesh epoch.
        uint64_t MetricsGeneration() const { return _metricsGeneration; }
        /// Explicit publication after the complete granular construction sequence.
        /// Shape checks alone cannot establish geometry/order correspondence.
        void FinalizeMetrics()
        {
            InvalidateMetricsDependencies();
            auto read = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            DNDS_check_throw_info(_metricsBuildMesh.lock() == mesh && _metricsBuildStamp.Valid() && _metricsStages == AllMetrics,
                                  "metrics require every construction stage in the current mesh epoch");
            auto rows = [](const auto &pair, index owned, index ghost)
            {
                DNDS_check_throw_info(pair.father && pair.son && pair.father->Size() == owned && pair.son->Size() == ghost,
                                      "metric array distribution differs from current mesh");
            };
            rows(volumeLocal, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellAtr, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellIntJacobiDet, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellBary, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellCent, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellIntPPhysics, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellAlignedHBox, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellMajorHBox, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellMajorCoord, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellInertia, mesh->NumCell(), mesh->NumCellGhost());
            rows(cellSmoothScale, mesh->NumCell(), mesh->NumCellGhost());
            rows(faceArea, mesh->NumFace(), mesh->NumFaceGhost());
            rows(faceAtr, mesh->NumFace(), mesh->NumFaceGhost());
            rows(faceIntJacobiDet, mesh->NumFace(), mesh->NumFaceGhost());
            rows(faceUnitNorm, mesh->NumFace(), mesh->NumFaceGhost());
            rows(faceMeanNorm, mesh->NumFace(), mesh->NumFaceGhost());
            rows(faceCent, mesh->NumFace(), mesh->NumFaceGhost());
            rows(faceIntPPhysics, mesh->NumFace(), mesh->NumFaceGhost());
            periodicity = Geom::Base::CFVPeriodicity(read.HostView().Periodic());
            PublishMetricsDependencies();
        }
        [[nodiscard]] const FiniteVolumeSettings &getSettings() const
        {
            return settings;
        }

        void parseSettings(FiniteVolumeSettings::json &j)
        {
            InvalidateMetricsDependencies();
            _metricsBuildStamp = Runtime::VersionStamp{};
            settings.ParseFromJson(j);
        }

    protected:
        real sumVolume{0}, minVolume{veryLargeReal}, maxVolume{0};
        real volGlobal{0};           /// @brief constructed using ConstructMetrics()
        tScalarPair volumeLocal;     /// @brief constructed using ConstructMetrics()
        tScalarPair faceArea;        /// @brief constructed using ConstructMetrics()
        tRecAtrPair cellAtr;         /// @brief constructed using ConstructMetrics()
        tRecAtrPair faceAtr;         /// @brief constructed using ConstructMetrics()
        tCoeffPair cellIntJacobiDet; /// @brief constructed using ConstructMetrics()
        tCoeffPair faceIntJacobiDet; /// @brief constructed using ConstructMetrics()
        t3VecsPair faceUnitNorm;     /// @brief constructed using ConstructMetrics()
        t3VecPair faceMeanNorm;      /// @brief constructed using ConstructMetrics()
        t3VecPair cellBary;          /// @brief constructed using ConstructMetrics()
        t3VecPair faceCent;          /// @brief constructed using ConstructMetrics()
        t3VecPair cellCent;          /// @brief constructed using ConstructMetrics()
        t3VecsPair cellIntPPhysics;  /// @brief constructed using ConstructMetrics()
        t3VecsPair faceIntPPhysics;  /// @brief constructed using ConstructMetrics()
        t3VecPair cellAlignedHBox;   /// @brief constructed using ConstructMetrics()
        t3VecPair cellMajorHBox;     /// @brief constructed using ConstructMetrics()
        t3MatPair cellMajorCoord;    /// @brief constructed using ConstructMetrics()
        t3MatPair cellInertia;       /// @brief constructed using ConstructMetrics()
        tScalarPair cellSmoothScale; /// @brief constructed using ConstructMetrics()

        auto device_array_list()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_REF(volumeLocal),
                DNDS_MAKE_1_MEMBER_REF(faceArea),
                DNDS_MAKE_1_MEMBER_REF(cellAtr),
                DNDS_MAKE_1_MEMBER_REF(faceAtr),
                DNDS_MAKE_1_MEMBER_REF(cellIntJacobiDet),
                DNDS_MAKE_1_MEMBER_REF(faceIntJacobiDet),
                DNDS_MAKE_1_MEMBER_REF(faceUnitNorm),
                DNDS_MAKE_1_MEMBER_REF(faceMeanNorm),
                DNDS_MAKE_1_MEMBER_REF(cellBary),
                DNDS_MAKE_1_MEMBER_REF(faceCent),
                DNDS_MAKE_1_MEMBER_REF(cellCent),
                DNDS_MAKE_1_MEMBER_REF(cellIntPPhysics),
                DNDS_MAKE_1_MEMBER_REF(faceIntPPhysics),
                DNDS_MAKE_1_MEMBER_REF(cellAlignedHBox),
                DNDS_MAKE_1_MEMBER_REF(cellMajorHBox),
                DNDS_MAKE_1_MEMBER_REF(cellMajorCoord),
                DNDS_MAKE_1_MEMBER_REF(cellInertia),
                DNDS_MAKE_1_MEMBER_REF(cellSmoothScale));
        }

        template <typename F>
        void for_each_device_member(F &&f)
        {
            for_each_member_list(device_array_list(), std::forward<F>(f));
        }

        Geom::Base::CFVPeriodicity periodicity;

    protected:
        int axisSymmetric = 0;
        std::set<index> axisFaces;

    public:
        int GetAxisSymmetric() const { return axisSymmetric; }
        int SetAxisSymmetric(int v)
        {
            InvalidateMetricsDependencies();
            _metricsBuildStamp = Runtime::VersionStamp{};
            _metricsStages = 0;
            return axisSymmetric = v;
        }

    public:
        FiniteVolume(MPIInfo nMpi, ssp<Geom::UnstructuredMesh> nMesh)
            : mpi(nMpi), mesh(std::move(nMesh)), settings(mesh->getDim())
        {
            mRank = mesh->mRank;
            periodicity = mesh->Periodic(); //! copy here
        }

        int getDim() const { return mesh->getDim(); }

        /**
         * @brief make pair with default MPI type, match **cell** layout
         *
         * @tparam TArrayPair ArrayPair's type
         * @tparam TOthers A list of additional resizing parameter types
         * @param aPair the pair to be constructed
         * @param name descriptive name for the pair (appears in error messages)
         * @param others additional resizing parameters
         */
        template <class TArrayPair, class... TOthers>
        void MakePairDefaultOnCell(TArrayPair &aPair, const std::string &name, TOthers... others)
        {
            aPair.InitPair(name, mpi);
            aPair.father->Resize(mesh->NumCell(), others...);
            aPair.son->Resize(mesh->NumCellGhost(), others...);
        }

        /**
         * @brief make pair with default MPI type, match **face** layout
         *
         * @tparam TArrayPair ArrayPair's type
         * @tparam TOthers A list of additional resizing parameter types
         * @param aPair the pair to be constructed
         * @param name descriptive name for the pair (appears in error messages)
         * @param others additional resizing parameters
         */
        template <class TArrayPair, class... TOthers>
        void MakePairDefaultOnFace(TArrayPair &aPair, const std::string &name, TOthers... others)
        {
            aPair.InitPair(name, mpi);
            aPair.father->Resize(mesh->NumFace(), others...);
            aPair.son->Resize(mesh->NumFaceGhost(), others...);
        }

        void SetCellAtrBasic();

        void ConstructCellVolume();
        void ConstructCellBary();
        void ConstructCellCent();
        void ConstructCellIntJacobiDet();
        void ConstructCellIntPPhysics();
        void ConstructCellAlignedHBox(); // note this is AABB
        void ConstructCellMajorHBoxCoordInertia();

        void SetFaceAtrBasic();

        void ConstructFaceArea();
        void ConstructFaceCent();
        void ConstructFaceIntJacobiDet();
        void ConstructFaceIntPPhysics();
        void ConstructFaceUnitNorm(); // on quad int points
        void ConstructFaceMeanNorm();

        void ConstructCellSmoothScale();

        template <int nVarsFixed = 1>
        void BuildUDof(tUDof<nVarsFixed> &u, int nVars, bool buildSon = true, bool buildTrans = true, Geom::MeshLoc varloc = Geom::MeshLoc::Cell)
        {
            BuildUDofOnMesh(u, "FV::BuildUDof::u", mpi, mesh, nVars, buildSon, buildTrans, varloc);
        }

        template <int nVarsFixed, int dim>
        void BuildUGradD(tUGrad<nVarsFixed, dim> &u, int nVars, bool buildSon = true, bool buildTrans = true, Geom::MeshLoc varloc = Geom::MeshLoc::Cell)
        {
            BuildUGradDOnMesh(u, "FV::BuildUGradD::u", mpi, mesh, nVars, buildSon, buildTrans, varloc);
        }

        RecAtr &GetCellAtr(index iCell)
        {
            return cellAtr(iCell, 0);
        }

        int GetCellOrder(index iCell)
        {
            return cellAtr(iCell, 0).Order;
        }

        RecAtr &GetFaceAtr(index iFace)
        {
            return faceAtr(iFace, 0);
        }

        int maxNDOF()
        {
            if (getDim() == 2)
                return Geom::Base::GetNDof<2>(settings.maxOrder);
            else
                return Geom::Base::GetNDof<3>(settings.maxOrder);
        }

        real GetCellVol(index iCell) const { return volumeLocal[iCell][0]; }
        real GetFaceArea(index iFace) const { return faceArea[iFace][0]; }

        real GetCellSmoothScaleRatio(index iCell) const
        {
            return cellSmoothScale(iCell, 0);
        }

        real GetGlobalVol() const { return volGlobal; }

        real GetCellJacobiDet(index iCell, rowsize iG) { return cellIntJacobiDet(iCell, iG); }
        real GetFaceJacobiDet(index iFace, rowsize iG) { return faceIntJacobiDet(iFace, iG); }

        Geom::Elem::Quadrature GetFaceQuad(index iFace) const
        {
            auto e = mesh->GetFaceElement(iFace);
            return Geom::Elem::Quadrature{e, faceAtr(iFace, 0).intOrder};
        }

        Geom::Elem::Quadrature GetFaceQuadO1(index iFace) const
        {
            auto e = mesh->GetFaceElement(iFace);
            return Geom::Elem::Quadrature{e, 1};
        }

        real GetFaceParamArea(index iFace) const { return std::get<1>(this->GetFaceQuadO1(iFace).GetQuadraturePointInfo(0)); }

        Geom::Elem::Quadrature GetCellQuad(index iCell) const
        {
            auto e = mesh->GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, cellAtr(iCell, 0).intOrder};
        }

        Geom::Elem::Quadrature GetCellQuadO1(index iCell) const
        {
            auto e = mesh->GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, 1};
        }

        real GetCellParamVol(index iCell) const { return std::get<1>(this->GetCellQuadO1(iCell).GetQuadraturePointInfo(0)); }

        Geom::tPoint GetCellBary(index iCell) { return cellBary[iCell]; }

        bool CellIsFaceBack(index iCell, index iFace, rowsize ic2f) const
        {
            return mesh->CellIsFaceBack(iCell, iFace, ic2f);
        }

        index CellFaceOther(index iCell, index iFace, rowsize ic2f) const
        {
            return mesh->CellFaceOther(iCell, iFace, ic2f);
        }

        Geom::tPoint GetFaceNorm(index iFace, int iG) const
        {
            if (iG >= 0)
                return faceUnitNorm(iFace, iG);
            else
                return faceMeanNorm[iFace];
        }

        Geom::tPoint GetFaceNormFromCell(index iFace, index iCell, rowsize if2c, int iG)
        {
            if (!mesh->IsPeriodic())
                return GetFaceNorm(iFace, iG);
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetFaceNorm(iFace, iG);
            DNDS_assert(if2c >= 0);
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
                return mesh->Periodic().TransVector(GetFaceNorm(iFace, iG), faceID);
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
                return mesh->Periodic().TransVectorBack(GetFaceNorm(iFace, iG), faceID);
            return GetFaceNorm(iFace, iG);
        }

        Geom::tPoint GetFaceQuadraturePPhys(index iFace, int iG)
        {
            if (iG >= 0)
                return faceIntPPhysics(iFace, iG);
            else
                return faceCent[iFace];
        }

        Geom::tPoint GetFaceQuadraturePPhysFromCell(index iFace, index iCell, rowsize if2c, int iG)
        {
            if (!mesh->IsPeriodic())
                return GetFaceQuadraturePPhys(iFace, iG);
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetFaceQuadraturePPhys(iFace, iG);
            DNDS_assert(if2c >= 0);
            if (if2c == 1 && Geom ::FaceIDIsPeriodicMain(faceID)) // I am donor
            {
                return mesh->Periodic().TransCoord(GetFaceQuadraturePPhys(iFace, iG), faceID);
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) // I am main
                return mesh->Periodic().TransCoordBack(GetFaceQuadraturePPhys(iFace, iG), faceID);
            return GetFaceQuadraturePPhys(iFace, iG);
        }

        Geom::tPoint GetFacePointFromCell(index iFace, index iCell, rowsize if2c, const Geom::tPoint &pnt)
        {
            if (!mesh->IsPeriodic())
                return pnt;
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return pnt;
            DNDS_assert(if2c >= 0);
            if (if2c == 1 && Geom ::FaceIDIsPeriodicMain(faceID)) // I am donor
            {
                return mesh->Periodic().TransCoord(pnt, faceID);
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) // I am main
                return mesh->Periodic().TransCoordBack(pnt, faceID);
            return pnt;
        }

        Geom::tPoint GetOtherCellBaryFromCell(
            index iCell, index iCellOther,
            index iFace, rowsize if2c)
        {
            if (!mesh->IsPeriodic())
                return GetCellBary(iCellOther);

            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetCellBary(iCellOther);
            DNDS_assert(if2c >= 0);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return mesh->Periodic().TransCoord(GetCellBary(iCellOther), faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return mesh->Periodic().TransCoordBack(GetCellBary(iCellOther), faceID);
            return GetCellBary(iCellOther);
        }

        Geom::tPoint GetOtherCellPointFromCell(
            index iCell, index iCellOther,
            index iFace, rowsize if2c, const Geom::tPoint &pnt)
        {
            if (!mesh->IsPeriodic())
                return pnt;

            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return pnt;
            DNDS_assert(if2c >= 0);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return mesh->Periodic().TransCoord(pnt, faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return mesh->Periodic().TransCoordBack(pnt, faceID);
            return pnt;
        }

        Geom::tGPoint GetOtherCellInertiaFromCell(
            index iCell, index iCellOther,
            index iFace, rowsize if2c)
        { // structure copy of GetOtherCellBaryFromCell
            if (!mesh->IsPeriodic())
                return cellInertia[iCellOther];

            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return cellInertia[iCellOther];
            DNDS_assert(if2c >= 0);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return mesh->Periodic().TransMat<3>(cellInertia[iCellOther], faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return mesh->Periodic().TransMatBack<3>(cellInertia[iCellOther], faceID);
            return cellInertia[iCellOther];
        }

        Geom::tPoint GetCellQuadraturePPhys(index iCell, int iG)
        {
            if (iG >= 0)
                return cellIntPPhysics(iCell, iG);
            else
                return cellCent[iCell];
        }

        real GetCellMaxLenScale(index iCell)
        {
            if (this->getDim() == 2)
                return cellMajorHBox[iCell]({0, 1}).maxCoeff() * 2;
            else
                return cellMajorHBox[iCell].maxCoeff() * 2;
        }

        real GetCellNodeMinLenScale(index iCell)
        {
            Geom::tSmallCoords coords;
            mesh->GetCoordsOnCell(iCell, coords);
            auto e = mesh->GetCellElement(iCell);
            Geom::tSmallCoords coordsV = coords(EigenAll, Eigen::seq(0, e.GetNumVertices() - 1));

            real ret = veryLargeReal;
            for (int i = 0; i < e.GetNumVertices(); i++)
                for (int j = 0; j < i; j++)
                {
                    real d = (coordsV(EigenAll, i) - coordsV(EigenAll, j)).norm();
                    ret = std::min(ret, d);
                }
            return ret;
        }

        index getArrayBytes()
        {
            index bytes = this->getDeviceArrayBytes();
            MPI::AllreduceOneIndex(bytes, MPI_SUM, mpi);
            return bytes;
        }

        // to_device(), to_host(), device() are provided by
        // DeviceTransferable<FiniteVolume> via for_each_device_member().

        template <DeviceBackend B>
        using t_deviceView = FiniteVolumeDeviceView<B>;

        template <DeviceBackend B, class MeshView>
        friend class FiniteVolumeDeviceView;

        /// Host scope retaining the mesh allocation/metadata and read lease.
        /// The FiniteVolume and its metric arrays must outlive this scope and
        /// remain unchanged until all launched work completes. Raw View() copies
        /// have no ownership. Join other using streams before Complete()/Wait().
        template <DeviceBackend B>
        class DeviceReadScope
        {
            friend class FiniteVolume;
            FiniteVolume &_fv;
            ssp<Geom::UnstructuredMesh> _source;
            uint64_t _generation;
            Geom::MeshReadAccess _meshRead;
            DeviceReadScope(FiniteVolume &fv, Runtime::ExecutionContext context)
                : _fv(fv), _source(fv.mesh), _generation(fv.MetricsGeneration()),
                  _meshRead(_source->AcquireRead(Geom::MeshReadRequirements::CellFlux(), context)) {}

        public:
            DeviceReadScope(const DeviceReadScope &) = delete;
            DeviceReadScope(DeviceReadScope &&) noexcept = default;
            DeviceReadScope &operator=(const DeviceReadScope &) = delete;
            t_deviceView<B> View() const
            {
                DNDS_check_throw_info(_source == _fv.mesh && _generation == _fv.MetricsGeneration(),
                                      "finite-volume scope source or metric publication changed");
                _fv.ValidateMeshDependencies();
                _meshRead.Stamp().Validate();
                if constexpr (B == DeviceBackend::Host)
                    return {_fv, UnInitIndex, _meshRead.LegacyHostView()};
                else
                    return {_fv, UnInitIndex, _meshRead.LegacyDeviceView()};
            }
            void Complete(Runtime::Completion completion) { _meshRead.Complete(std::move(completion)); }
            void Wait() { _meshRead.Wait(); }
        };

        /// Acquire once per loop/launch batch. CUDA uses the current device and
        /// the supplied native stream. Native mesh input must first be placed
        /// with PlaceNative(device, context); acquisition never changes storage.
        /// Legacy mesh input is captured through the same read API.
        template <DeviceBackend B>
        DeviceReadScope<B> AcquireDeviceRead(void *stream = nullptr)
        {
            ValidateMeshDependencies();
            const auto placement = device();
            DNDS_check_throw_info(placement == B || (B == DeviceBackend::Host && placement == DeviceBackend::Unknown),
                                  "finite-volume metric placement differs from requested view");
            Runtime::ExecutionContext context = Runtime::ExecutionContext::Host();
            if constexpr (B == DeviceBackend::Host)
                DNDS_check_throw_info(!stream, "host finite-volume acquisition does not accept a CUDA stream");
            else
            {
#ifdef DNDS_USE_CUDA
                int ordinal = -1;
                DNDS_check_throw_info(cudaGetDevice(&ordinal) == cudaSuccess, "cannot query CUDA device for finite-volume view");
                context = Runtime::ExecutionContext::CUDA(ordinal, Runtime::BorrowExternalCUDAStream(ordinal, static_cast<cudaStream_t>(stream)));
#else
                DNDS_check_throw_info(false, "CUDA finite-volume view requires a CUDA build");
#endif
            }
            return DeviceReadScope<B>(*this, context);
        }
    };

}
