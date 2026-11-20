#pragma once
#include "Geom/Elements/Elements.hpp"
#include "Geom/Interface/BoundaryCondition.hpp"
#include "Geom/Periodic/PeriodicInfo.hpp"
#include "Geom/Math/RadialBasisFunction.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "Geom/Transition/ReorderPlan.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include "Geom/Compat/LegacyMeshRead.hpp"
#include "Geom/Compat/LegacyFieldBridge.hpp"
#include "Geom/Access/EntitySpace.hpp"
#include "Geom/Access/RelationAccess.hpp"
#include "Geom/Access/MeshMutation.hpp"
#include "Geom/Unstructured/UnstructuredMeshState.hpp"
#include "Geom/IO/NativeMeshSerialOutput.hpp"
// Legacy field/output schedule types; no legacy mesh allocations are owned.
#include "Geom/Compat/LegacyMeshDeviceView.hpp"

namespace DNDS::Direct
{
    struct SerialSymLUStructure;
    struct DirectPrecControl;
}
namespace DNDS::Geom
{
    struct LocalCellOrderingSettings;
    class NativeSmoothingWorkspace;
    struct PartitionOptions;
    enum class MeshLoc : uint8_t
    {
        Unknown = 0,
        Node = 1,
        Face = 3,
        Cell = 4
    };

    /// Public mesh facade over one immutable native publication. Geometry/topology
    /// ownership belongs to UnstructuredMeshState and shared Runtime value/shape buffers.
    /// Borrowed semantic rows require an outer AcquireRead or serialized setup.
    struct UnstructuredMesh
    {
    private:
        std::shared_ptr<const uint8_t> _entityIdentity = std::make_shared<const uint8_t>(0);
        std::shared_ptr<const UnstructuredMeshState> _nativeState;
        // Keep the facade's raw MPIInfo handle alive independently of state
        // replacement, including replacement by a caller-managed publication.
        std::shared_ptr<Runtime::CommunicationContext> _mpiOwner;
        std::shared_ptr<const NativeSmoothingWorkspace> _nativeSmoothing;
        friend class UnstructuredMeshState;
        friend class MeshEdit;
        friend class PreparedMeshTransition;
        const UnstructuredMeshState &Native() const
        {
            DNDS_check_throw_info(_nativeState, "native mesh state is not published");
            return *_nativeState;
        }
        const UnstructuredMeshView &NativeBorrowedView(MeshRelation relation) const
        {
            const auto &view = Native().BorrowHostView();
            DNDS_check_throw_info(view.HasRelation(relation), "requested mesh relation is not published");
            return view;
        }
        index NativeIndexGlobal2Local(EntityKind kind, index global, bool ownedOnly = false) const
        {
            if (global == UnInitIndex)
                return global;
            const auto space = Entities(kind);
            const auto local = space.GlobalToLocal(global);
            if (local && (!ownedOnly || *local < space.NumOwned()))
                return *local;
            return -1 - global;
        }
        index NativeIndexLocal2Global(EntityKind kind, index local, bool ownedOnly = false) const
        {
            if (local == UnInitIndex)
                return local;
            if (local < 0)
                return -1 - local;
            const auto space = Entities(kind);
            DNDS_check_throw_info(!ownedOnly || local < space.NumOwned(), "local entity index outside owned rows");
            return space.LocalToGlobal(local);
        }
        void PublishNativeGeometry(std::shared_ptr<const UnstructuredMeshState> prepared,
                                   Runtime::VersionStamp sourceStamp,
                                   Runtime::Change changes = Runtime::Change::Geometry | Runtime::Change::Values | Runtime::Change::Interface | Runtime::Change::Allocation);
        void TransformNativeCoords(const std::function<tPoint(const tPoint &)> &transform);

    public:
        MPI_int mRank = 0;
        MPIInfo mpi;
        int dim;
        // Initial reader configuration only. Published access uses native metadata.
        bool isPeriodic = false;
        Periodicity periodicInfo;
        Runtime::VersionTracker versions;
        std::vector<index> node2parentNode, node2bndNode, cell2parentCell;
        struct HDF5OutSetting
        {
            size_t chunkSize = 128;
            int deflateLevel = 1;
            bool coll_on_data = false;
            bool coll_on_meta = true;
        } hdf5OutSetting;
        struct ElevationInfo
        {
            real RBFRadius = 1;
            real MaxIncludedAngle = 15;
            int nIter = 60;
            int nSearch = 30;
            RBF::RBFKernelType kernel = RBF::InversedDistanceA1;
            real refDWall = 1e-3;
            real RBFPower = 1;
        } elevationInfo;
        UnstructuredMesh(const MPIInfo &mpi_, int dim_) : mpi(mpi_), dim(dim_) {}
        UnstructuredMesh(const UnstructuredMesh &) = delete;
        UnstructuredMesh &operator=(const UnstructuredMesh &) = delete;
        bool HasNativeState() const { return bool(_nativeState); }
        /// Immutable setup metadata; no geometry validation, read lease or MPI.
        std::shared_ptr<Runtime::CommunicationContext> PreparationCommunication() const
        {
            return _nativeState ? _nativeState->PreparationCommunication() : nullptr;
        }
        std::shared_ptr<const UnstructuredMeshState> CurrentNativeState() const
        {
            Native().Validate();
            return _nativeState;
        }
        MeshReadRequirements CurrentNativeRequirements() const { return Native().Requirements(); }
        std::vector<AdjacencyDescriptor> AdjacencyKeys() const { return Native().RelationKeys(); }
        MeshAdjacencyRead ReadAdjacency(const AdjacencyDescriptor &key) const { return Native().Relation(key); }
        std::vector<AdjacencyDescriptor> NativeRelationKeys() const { return Native().RelationKeys(); }
        MeshAdjacencyRead ReadNativeRelation(const AdjacencyDescriptor &key) const { return Native().Relation(key); }
        void ReplaceNativeForSetup(std::shared_ptr<const UnstructuredMeshState> source);
        static std::shared_ptr<UnstructuredMesh> FromNative(const MPIInfo &, std::shared_ptr<const UnstructuredMeshState>);
        void PrepareNativeTopology(bool faces = true, bool edges = false, int ghostLayers = 1);
        void PlaceNative(Runtime::Device device, Runtime::ExecutionContext context);
        EntitySpace Entities(EntityKind kind) const;
        MeshReadAccess AcquireRead(MeshReadRequirements requirements = MeshReadRequirements::Primary(),
                                   Runtime::ExecutionContext context = Runtime::ExecutionContext::Host());
        /// Immediate managed borrow. Release this scope before closing its
        /// engine; plain views still require the enclosing read owner.
        MeshReadAccess AcquireRead(MeshReadRequirements requirements, const Runtime::ExecutionStream &stream);
        UnstructuredMeshLegacyRead AcquireLegacyRead(MeshReadRequirements requirements = MeshReadRequirements::Primary(),
                                                     Runtime::ExecutionContext context = Runtime::ExecutionContext::Host())
        {
            return UnstructuredMeshLegacyRead(AcquireRead(requirements, context));
        }
        index OriginalEntityId(EntityKind kind, index row) const { return Native().OriginalEntityId(kind, row); }
        /// Caller serializes submission/publication and coordinates the safety
        /// mode on all ranks. Off mode retains local checks and aborts on failure.
        void CheckCollectiveMutationAvailable(bool collectivePreflight = true) const
        {
            Runtime::CommunicationDetail::Preflight(versions.CanMutate(), mpi.comm, collectivePreflight,
                                                    "mesh mutation blocked by an active reader, writer or failed transition");
        }
        int getDim() const { return _nativeState ? Native().BorrowHostView().Dimension() : dim; }
        bool IsPeriodic() const { return _nativeState ? Native().BorrowHostView().IsPeriodic() : isPeriodic; }
        const Periodicity &PeriodicMetadata() const { return _nativeState ? Native().BorrowHostView().Periodic() : periodicInfo; }
        const Periodicity &Periodic() const { return PeriodicMetadata(); }
        MPIInfo &getMPI() { return mpi; }
        const MPIInfo &getMPI() const { return mpi; }
        MeshRowSpan<index> CellNodes(index row) const { return NativeBorrowedView(MeshRelation::CellNodes).CellNodes(row); }
        MeshRowSpan<index> CellFaces(index row) const { return NativeBorrowedView(MeshRelation::CellFaces).CellFaces(row); }
        MeshRowSpan<index> FaceNodes(index row) const { return NativeBorrowedView(MeshRelation::FaceNodes).FaceNodes(row); }
        MeshRowSpan<index> FaceCells(index row) const { return NativeBorrowedView(MeshRelation::FaceCells).FaceCells(row); }
        MeshRowSpan<index> BoundaryCells(index row) const { return NativeBorrowedView(MeshRelation::BoundaryCells).BoundaryCells(row); }
        MeshRowSpan<index> BoundaryNodes(index row) const { return NativeBorrowedView(MeshRelation::BoundaryNodes).BoundaryNodes(row); }
        index BoundaryFace(index row) const { return NativeBorrowedView(MeshRelation::BoundaryFaces).BoundaryFaces(row)[0]; }
        tPoint Coord(index row) const { return Native().BorrowHostView().Coord(row); }
        bool HasNodeWallDist() const { return Native().BorrowHostView().HasNodeWallDist(); }
        tPoint NodeWallDist(index row) const
        {
            DNDS_check_throw_info(HasNodeWallDist(), "wall distance is not published");
            return Native().BorrowHostView().NodeWallDist(row);
        }
        const ElemInfo &GetCellInfo(index row) const { return Native().BorrowHostView().GetCellInfo(row); }
        Elem::Element GetCellElement(index row) const { return Native().BorrowHostView().GetCellElement(row); }
        t_index GetCellZone(index row) const { return Native().BorrowHostView().GetCellZone(row); }
        const ElemInfo &GetFaceInfo(index row) const { return NativeBorrowedView(MeshRelation::FaceNodes).GetFaceInfo(row); }
        Elem::Element GetFaceElement(index row) const { return NativeBorrowedView(MeshRelation::FaceNodes).GetFaceElement(row); }
        t_index GetFaceZone(index row) const { return NativeBorrowedView(MeshRelation::FaceNodes).GetFaceZone(row); }
        const ElemInfo &GetBndInfo(index row) const { return Native().BorrowHostView().GetBndInfo(row); }
        Elem::Element GetBndElement(index row) const { return Native().BorrowHostView().GetBndElement(row); }
        t_index GetBndZone(index row) const { return Native().BorrowHostView().GetBndZone(row); }
        index NumNode() const { return Native().NumOwned(EntityKind::Node); }
        index NumNodeGhost() const { return Native().NumGhost(EntityKind::Node); }
        index NumNodeProc() const { return NumNode() + NumNodeGhost(); }
        index NumNodeGlobal() const { return Native().NumGlobal(EntityKind::Node); }
        index NodeIndexGlobal2Local(index value) const { return NativeIndexGlobal2Local(EntityKind::Node, value); }
        index NodeIndexGlobal2Local_NoSon(index value) const { return NativeIndexGlobal2Local(EntityKind::Node, value, true); }
        index NodeIndexLocal2Global(index value) const { return NativeIndexLocal2Global(EntityKind::Node, value); }
        index NodeIndexLocal2Global_NoSon(index value) const { return NativeIndexLocal2Global(EntityKind::Node, value, true); }
        index NumCell() const { return Native().NumOwned(EntityKind::Cell); }
        index NumCellGhost() const { return Native().NumGhost(EntityKind::Cell); }
        index NumCellProc() const { return NumCell() + NumCellGhost(); }
        index NumCellGlobal() const { return Native().NumGlobal(EntityKind::Cell); }
        index CellIndexGlobal2Local(index value) const { return NativeIndexGlobal2Local(EntityKind::Cell, value); }
        index CellIndexGlobal2Local_NoSon(index value) const { return NativeIndexGlobal2Local(EntityKind::Cell, value, true); }
        index CellIndexLocal2Global(index value) const { return NativeIndexLocal2Global(EntityKind::Cell, value); }
        index CellIndexLocal2Global_NoSon(index value) const { return NativeIndexLocal2Global(EntityKind::Cell, value, true); }
        index NumFace() const { return Native().NumOwned(EntityKind::Face); }
        index NumFaceGhost() const { return Native().NumGhost(EntityKind::Face); }
        index NumFaceProc() const { return NumFace() + NumFaceGhost(); }
        index NumFaceGlobal() const { return Native().NumGlobal(EntityKind::Face); }
        index FaceIndexGlobal2Local(index value) const { return NativeIndexGlobal2Local(EntityKind::Face, value); }
        index FaceIndexGlobal2Local_NoSon(index value) const { return NativeIndexGlobal2Local(EntityKind::Face, value, true); }
        index FaceIndexLocal2Global(index value) const { return NativeIndexLocal2Global(EntityKind::Face, value); }
        index FaceIndexLocal2Global_NoSon(index value) const { return NativeIndexLocal2Global(EntityKind::Face, value, true); }
        index NumBnd() const { return Native().NumOwned(EntityKind::Bnd); }
        index NumBndGhost() const { return Native().NumGhost(EntityKind::Bnd); }
        index NumBndProc() const { return NumBnd() + NumBndGhost(); }
        index NumBndGlobal() const { return Native().NumGlobal(EntityKind::Bnd); }
        index BndIndexGlobal2Local(index value) const { return NativeIndexGlobal2Local(EntityKind::Bnd, value); }
        index BndIndexGlobal2Local_NoSon(index value) const { return NativeIndexGlobal2Local(EntityKind::Bnd, value, true); }
        index BndIndexLocal2Global(index value) const { return NativeIndexLocal2Global(EntityKind::Bnd, value); }
        index BndIndexLocal2Global_NoSon(index value) const { return NativeIndexLocal2Global(EntityKind::Bnd, value, true); }
        index NumEdge() const { return Native().NumOwned(EntityKind::Edge); }
        index NumEdgeGhost() const { return Native().NumGhost(EntityKind::Edge); }
        index NumEdgeProc() const { return NumEdge() + NumEdgeGhost(); }
        index NumEdgeGlobal() const { return Native().NumGlobal(EntityKind::Edge); }
        index EdgeIndexGlobal2Local(index value) const { return NativeIndexGlobal2Local(EntityKind::Edge, value); }
        index EdgeIndexGlobal2Local_NoSon(index value) const { return NativeIndexGlobal2Local(EntityKind::Edge, value, true); }
        index EdgeIndexLocal2Global(index value) const { return NativeIndexLocal2Global(EntityKind::Edge, value); }
        index EdgeIndexLocal2Global_NoSon(index value) const { return NativeIndexLocal2Global(EntityKind::Edge, value, true); }
        std::vector<NodeIndexPBI> getCell2NodeIndexPbiRow(index row) const
        {
            const auto &view = NativeBorrowedView(MeshRelation::CellNodes);
            const auto nodes = view.CellNodes(row);
            std::vector<NodeIndexPBI> result;
            result.reserve(nodes.size());
            for (size_t slot = 0; slot < nodes.size(); ++slot)
                result.push_back({nodes[slot], view.IsPeriodic() ? view.CellNodePeriodicBits(row)[slot] : NodePeriodicBits{}});
            return result;
        }
        std::vector<NodeIndexPBI> getBnd2NodeIndexPbiRow(index row) const
        {
            const auto &view = NativeBorrowedView(MeshRelation::BoundaryNodes);
            const auto nodes = view.BoundaryNodes(row);
            std::vector<NodeIndexPBI> result;
            result.reserve(nodes.size());
            for (size_t slot = 0; slot < nodes.size(); ++slot)
                result.push_back({nodes[slot], view.IsPeriodic() ? view.BoundaryNodePeriodicBits(row)[slot] : NodePeriodicBits{}});
            return result;
        }
        void GetCoordsOnCell(index row, tSmallCoords &coords) const { Native().BorrowHostView().GetCoordsOnCell(row, coords); }
        void GetCoordsOnCell(index row, tSmallCoords &coords, tCoordPair &external) const { Native().BorrowHostView().GetCoordsOnCell(row, coords, external); }
        void GetCoordsOnFace(index row, tSmallCoords &coords) const { NativeBorrowedView(MeshRelation::FaceNodes).GetCoordsOnFace(row, coords); }
        tPoint GetCoordNodeOnCell(index row, rowsize slot) const { return Native().BorrowHostView().GetCoordNodeOnCell(row, slot); }
        tPoint GetCoordNodeOnFace(index row, rowsize slot) const { return NativeBorrowedView(MeshRelation::FaceNodes).GetCoordNodeOnFace(row, slot); }
        tPoint GetCoordWallDistOnCell(index row, rowsize slot) const { return Native().BorrowHostView().GetCoordWallDistOnCell(row, slot); }
        tPoint GetCoordWallDistOnFace(index row, rowsize slot) const { return NativeBorrowedView(MeshRelation::FaceNodes).GetCoordWallDistOnFace(row, slot); }
        bool CellIsFaceBack(index cell, index face, rowsize slot) const { return NativeBorrowedView(MeshRelation::FaceNodes).CellIsFaceBack(cell, face, slot); }
        index CellFaceOther(index cell, index face, rowsize slot) const { return NativeBorrowedView(MeshRelation::FaceNodes).CellFaceOther(cell, face, slot); }
        template <class FA, class FB, class F0 = std::function<void(void)>>
        auto CellOtherCellPeriodicHandle(
            index iFace, rowsize if2c, FA &&fA, FB &&fB, F0 &&f0 = []() {})
        {
            if (!this->IsPeriodic())
                return f0();
            auto faceID = this->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return f0();
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return fA();
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return fB();
        }
        bool IsO1();
        bool IsO2();
        void AssertOnFaces();
        void AssertOnN2CB();
        std::size_t CoordinateHash();
        tLocalMatStruct GetCell2CellFaceVLocal(bool onLocalPartition = false);
        void ObtainLocalFactFillOrdering(Direct::SerialSymLUStructure &, Direct::DirectPrecControl);
        void ObtainSymmetricSymbolicFactorization(Direct::SerialSymLUStructure &, Direct::DirectPrecControl) const;
        void ReorderLocalCells(int nParts = 1, int nPartsInner = 1, int ghostLayers = 1);
        void ReorderLocalCells(const struct LocalCellOrderingSettings &settings, int ghostLayers = 1);
        int NLocalParts() const { return Native().NLocalParts(); }
        /// Each worker owns at most two ordered intervals in the current owned cell numbering.
        WorkerRowRanges LocalPartRanges(int part) const { return Native().LocalPartRanges(part); }
        index LocalPartStart(int part) const { return Native().LocalPartStart(part); }
        index LocalPartEnd(int part) const { return Native().LocalPartEnd(part); }
        const tLocalMatStruct &LocalCellNeighbors() const { return Native().LocalCellNeighbors(); }
        void BuildO2FromO1Elevation(UnstructuredMesh &source);
        void BuildBisectO1FormO2(UnstructuredMesh &source);
        void ElevatedNodesGetBoundarySmooth(const std::function<bool(t_index)> &);
        void ElevatedNodesSolveInternalSmooth();
        void ElevatedNodesSolveInternalSmoothV1();
        void ElevatedNodesSolveInternalSmoothV2();
        void ElevatedNodesSolveInternalSmoothV1Old();
        std::shared_ptr<const NativeSmoothingWorkspace> BoundarySmoothingWorkspace() const
        {
            DNDS_check_throw_info(_nativeSmoothing, "prepare boundary smoothing workspace before diagnostics");
            return _nativeSmoothing;
        }
        void ApplyGeometryTransform(const Eigen::Matrix3d &rotation, real scale);
        void ApplyAffineGeometryTransform(const Eigen::Matrix3d &, const tPoint &offset = tPoint::Zero());
        /// Snap canonical owned coordinates near selected zero planes (bits x/y/z),
        /// then refresh ghosts. Periodic image labels and frames stay unchanged.
        void RectifyNearPlaneCoordinates(int components, real threshold);
        void RectifyBoundaryCoordinates(std::function<int(t_index)>, bool volumeBoundary = true);
        template <class F>
        void TransformCoords(F &&transform)
        {
            TransformNativeCoords([&](const tPoint &point)
                                  {tPoint value=point;return tPoint(transform(value)); });
        }
        std::shared_ptr<UnstructuredMesh> ExtractBoundary() const;
        void ConstructBndMesh(UnstructuredMesh &destination);
        void WriteNativeArchive(Serializer::SerializerBaseSSP, const std::string &) const;
        static std::shared_ptr<UnstructuredMesh> ReadNativeArchive(const MPIInfo &, Serializer::SerializerBaseSSP, const std::string &,
                                                                   Serializer::ArrayGlobalOffset = Serializer::ArrayGlobalOffset_Unknown, int ghostLayers = 1);
        /// Collective primary wire-compatibility export; use full native archive for restart.
        void WriteSerialize(Serializer::SerializerBaseSSP, const std::string &);
        void ReadSerialize(Serializer::SerializerBaseSSP, const std::string &);
        void ReadSerializeAndDistribute(Serializer::SerializerBaseSSP, const std::string &, const PartitionOptions &);
        void PrintParallelVTKHDFDataArray(
            std::string fname, std::string seriesName,
            int arraySiz, int vecArraySiz, int arraySizPoint, int vecArraySizPoint,
            const tFGetName &names,
            const tFGetData &data,
            const tFGetName &vectorNames,
            const tFGetVecData &vectorData,
            const tFGetName &namesPoint,
            const tFGetData &dataPoint,
            const tFGetName &vectorNamesPoint,
            const tFGetVecData &vectorDataPoint,
            double t, MPI_Comm commDup);
        /// Additional typed arrays. Integer cell rows follow the owned cell
        /// order; field arrays are identical across the output communicator.
        struct VTKHDFExtraFields
        {
            std::vector<std::pair<std::string, SpanView<const int>>> cellIntegers;
            std::vector<std::pair<std::string, std::vector<int>>> fieldIntegers;
            std::vector<std::pair<std::string, std::vector<std::string>>> fieldStrings;
        };
        /// Per-call settings avoid mutating the shared mesh for independent
        /// output consumers. The historical overload retains its ABI.
        void PrintParallelVTKHDFDataArray(
            std::string fname, std::string seriesName,
            int arraySiz, int vecArraySiz, int arraySizPoint, int vecArraySizPoint,
            const tFGetName &names, const tFGetData &data,
            const tFGetName &vectorNames, const tFGetVecData &vectorData,
            const tFGetName &namesPoint, const tFGetData &dataPoint,
            const tFGetName &vectorNamesPoint, const tFGetVecData &vectorDataPoint,
            double t, MPI_Comm commDup, const HDF5OutSetting *settings,
            const VTKHDFExtraFields &extra = {});
        void PrintMeshCGNS(std::string fname, const t_FBCID_2_Name &fbcid2name, const std::vector<std::string> &allNames);
        void SetHDF5OutSetting(size_t chunkSiz, int deflateLevel, bool coll_on_data, bool coll_on_meta)
        {
            hdf5OutSetting.chunkSize = chunkSiz;
            hdf5OutSetting.deflateLevel = deflateLevel;
            hdf5OutSetting.coll_on_data = coll_on_data;
            hdf5OutSetting.coll_on_meta = coll_on_meta;
        }
        struct WallDistOptions
        {
            int subdivide_quad = 1;
            int method = 0;
            int wallDistExecution = 0;
            real minWallDist = 1e-10;
            int verbose = 0;
            // NOLINTNEXTLINE(modernize-use-equals-default): nested class default-init quirk with = default
            WallDistOptions() {}

            DNDS_DECLARE_CONFIG(WallDistOptions)
            {
                // clang-format off
                DNDS_FIELD(subdivide_quad, "Subdivide quads for wall distance computation",
                           DNDS::Config::range(0));
                DNDS_FIELD(method, "Wall distance computation method (0: brute, 1: tree)",
                           DNDS::Config::range(0));
                DNDS_FIELD(wallDistExecution, "MPI concurrency (0: all parallel, 1: serial, >1: batched N ranks)",
                           DNDS::Config::range(0));
                DNDS_FIELD(minWallDist, "Minimum wall distance clamp",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(verbose, "Verbosity level for wall distance computation",
                           DNDS::Config::range(0));
                // clang-format on
            }
        };
        void BuildNodeWallDist(const std::function<bool(Geom::t_index)> &fBndIsWall, WallDistOptions options = WallDistOptions{});
        index getArrayBytes()
        {
            const auto local = Native().ArrayBytes();
            DNDS_check_throw_info(local <= size_t(std::numeric_limits<index>::max()), "mesh allocation bytes overflow");
            index bytes = index(local);
            MPI::AllreduceOneIndex(bytes, MPI_SUM, mpi);
            return bytes;
        }
    };
    struct PartitionOptions
    {
        std::string metisType = "KWAY";
        int metisUfactor = 20;
        int metisSeed = 0;
        int edgeWeightMethod = 0; // 0:no weight 1: faceSize weight
        int metisNcuts = 3;

        DNDS_DECLARE_CONFIG(PartitionOptions)
        {
            // clang-format off
            DNDS_FIELD(metisType, "METIS partitioning method",
                       DNDS::Config::enum_values({"KWAY", "RB"}));
            DNDS_FIELD(metisUfactor, "METIS imbalance factor (ufactor)",
                       DNDS::Config::range(1));
            DNDS_FIELD(metisSeed, "METIS random seed");
            DNDS_FIELD(edgeWeightMethod, "Edge weight method (0: none, 1: face size)",
                       DNDS::Config::range(0, 1));
            DNDS_FIELD(metisNcuts, "Number of cuts for METIS to try",
                       DNDS::Config::range(1));
            // clang-format on
        }
    };
    enum MeshReaderMode
    {
        UnknownMode,
        SerialReadAndDistribute,
        SerialOutput
    };
    /// Native serial output adapter. The maps below route external legacy solver
    /// fields; they neither allocate nor own any legacy geometry payload.
    struct UnstructuredMeshSerialRW
    {
        using PartitionOptions = Geom::PartitionOptions;

    private:
        int ascii_precision = 16;
        std::string vtuFloatEncodeMode = "ascii";

    public:
        std::shared_ptr<NativeMeshSerialOutput> nativeSerialOutput;
        ssp<UnstructuredMesh> mesh;
        MeshReaderMode mode = UnknownMode;
        bool dataIsSerialOut = false, dataIsSerialIn = false;
        MPI_int mRank = 0;
        DNDS::ArrayTransformerType<tCoord::element_type>::Type coordSerialOutTrans;                // used in serial out mode
        DNDS::ArrayTransformerType<tAdj::element_type>::Type cell2nodeSerialOutTrans;              // used in serial out mode
        DNDS::ArrayTransformerType<tPbi::element_type>::Type cell2nodePbiSerialOutTrans;           // used in serial out mode
        DNDS::ArrayTransformerType<tAdj::element_type>::Type bnd2nodeSerialOutTrans;               // used in serial out mode
        DNDS::ArrayTransformerType<tElemInfoArray::element_type>::Type cellElemInfoSerialOutTrans; // used in serial out mode
        DNDS::ArrayTransformerType<tElemInfoArray::element_type>::Type bndElemInfoSerialOutTrans;  // used in serial out mode
        UnstructuredMeshSerialRW(ssp<UnstructuredMesh> source, MPI_int writer) : mesh(std::move(source)), mRank(writer) {}
        void BuildSerialOut();
        void ClearSerial()
        {
            nativeSerialOutput.reset();
            mode = UnknownMode;
            dataIsSerialIn = dataIsSerialOut = false;
        }
        void PrintSerialPartPltBinaryDataArray(
            std::string fname,
            int arraySiz, int arraySizPoint,
            const tFGetName &names,
            const tFGetData &data,
            const tFGetName &namesPoint,
            const tFGetData &dataPoint,
            double t, int flag);
        void PrintSerialPartVTKDataArray(
            std::string fname, std::string seriesName,
            int arraySiz, int vecArraySiz, int arraySizPoint, int vecArraySizPoint,
            const tFGetName &names,
            const tFGetData &data,
            const tFGetName &vectorNames,
            const tFGetVecData &vectorData,
            const tFGetName &namesPoint,
            const tFGetData &dataPoint,
            const tFGetName &vectorNamesPoint,
            const tFGetVecData &vectorDataPoint,
            double t, int flag = 0);
        void SetASCIIPrecision(int n) { ascii_precision = n; }
        void SetVTKFloatEncodeMode(const std::string &v)
        {
            vtuFloatEncodeMode = v;
            DNDS_assert(v == "ascii" || v == "binary");
        }
    };
}
