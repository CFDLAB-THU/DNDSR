#pragma once
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/DeviceTransferable.hpp"
#include "DNDS/Errors.hpp"
#include "FiniteVolumeSettings.hpp"
#include "Geom/Mesh.hpp"
#include "VRDefines.hpp"
#include "Geom/DiffTensors.hpp"
#include "DOFFactory.hpp"

#include "FiniteVolume_DeviceView.hpp"

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

    public:
        [[nodiscard]] const FiniteVolumeSettings &getSettings() const
        {
            return settings;
        }

        void parseSettings(FiniteVolumeSettings::json &j)
        {
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
        int SetAxisSymmetric(int v) { return axisSymmetric = v; }

    public:
        FiniteVolume(MPIInfo nMpi, ssp<Geom::UnstructuredMesh> nMesh)
            : mpi(nMpi), mesh(std::move(nMesh)), settings(mesh->getDim())
        {
            mRank = mesh->mRank;
            periodicity = mesh->periodicInfo; //! copy here
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

        bool CellIsFaceBack(index iCell, index iFace) const
        {
            return mesh->CellIsFaceBack(iCell, iFace);
        }

        index CellFaceOther(index iCell, index iFace) const
        {
            return mesh->CellFaceOther(iCell, iFace);
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
            if (!mesh->isPeriodic)
                return GetFaceNorm(iFace, iG);
            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetFaceNorm(iFace, iG);
            if (if2c < 0)
                if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
                return mesh->periodicInfo.TransVector(GetFaceNorm(iFace, iG), faceID);
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
                return mesh->periodicInfo.TransVectorBack(GetFaceNorm(iFace, iG), faceID);
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
            if (!mesh->isPeriodic)
                return GetFaceQuadraturePPhys(iFace, iG);
            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetFaceQuadraturePPhys(iFace, iG);
            if (if2c < 0)
                if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom ::FaceIDIsPeriodicMain(faceID)) // I am donor
            {
                // std::cout << iFace <<" " << iCell << " " <<if2c << std::endl;
                // std::cout << GetFaceQuadraturePPhys(iFace, iG).transpose() << std::endl;
                // std::cout << mesh->periodicInfo.TransCoord(GetFaceQuadraturePPhys(iFace, iG), faceID).transpose() << std::endl;
                // std::abort();
                return mesh->periodicInfo.TransCoord(GetFaceQuadraturePPhys(iFace, iG), faceID);
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) // I am main
                return mesh->periodicInfo.TransCoordBack(GetFaceQuadraturePPhys(iFace, iG), faceID);
            return GetFaceQuadraturePPhys(iFace, iG);
        }

        Geom::tPoint GetFacePointFromCell(index iFace, index iCell, rowsize if2c, const Geom::tPoint &pnt)
        {
            if (!mesh->isPeriodic)
                return pnt;
            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return pnt;
            if (if2c < 0)
                if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom ::FaceIDIsPeriodicMain(faceID)) // I am donor
            {
                // std::cout << iFace <<" " << iCell << " " <<if2c << std::endl;
                // std::cout << pnt.transpose() << std::endl;
                // std::cout << mesh->periodicInfo.TransCoord(pnt, faceID).transpose() << std::endl;
                // std::abort();
                return mesh->periodicInfo.TransCoord(pnt, faceID);
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) // I am main
                return mesh->periodicInfo.TransCoordBack(pnt, faceID);
            return pnt;
        }

        Geom::tPoint GetOtherCellBaryFromCell(
            index iCell, index iCellOther,
            index iFace)
        {
            if (!mesh->isPeriodic)
                return GetCellBary(iCellOther);

            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetCellBary(iCellOther);
            rowsize if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return mesh->periodicInfo.TransCoord(GetCellBary(iCellOther), faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return mesh->periodicInfo.TransCoordBack(GetCellBary(iCellOther), faceID);
            return GetCellBary(iCellOther);
        }

        Geom::tPoint GetOtherCellPointFromCell(
            index iCell, index iCellOther,
            index iFace, const Geom::tPoint &pnt)
        {
            if (!mesh->isPeriodic)
                return pnt;

            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return pnt;
            rowsize if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return mesh->periodicInfo.TransCoord(pnt, faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return mesh->periodicInfo.TransCoordBack(pnt, faceID);
            return pnt;
        }

        Geom::tGPoint GetOtherCellInertiaFromCell(
            index iCell, index iCellOther,
            index iFace)
        { // structure copy of GetOtherCellBaryFromCell
            if (!mesh->isPeriodic)
                return cellInertia[iCellOther];

            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return cellInertia[iCellOther];
            rowsize if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return mesh->periodicInfo.TransMat<3>(cellInertia[iCellOther], faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return mesh->periodicInfo.TransMatBack<3>(cellInertia[iCellOther], faceID);
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

        template <DeviceBackend B>
        friend class FiniteVolumeDeviceView;

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DeviceBackend B_mesh = mesh->device();
            DNDS_assert(B_mesh == B || (B == DeviceBackend::Host && B_mesh == DeviceBackend::Unknown));
            return {*this, UnInitIndex};
        }
    };

}