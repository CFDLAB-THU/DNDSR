#include "Geom/Quadratures/Quadrature.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/IO/NativeMeshOutput.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/IO/CGNS.hpp"
#include "DNDS/IO/OutputDir.hpp"
#include "DNDS/Serializer/HDF5Access.hpp"
#include <array>
#include <thread>
#include <filesystem>
#include <base64_rfc4648.hpp>
#include <zlib.h>

#include <nanoflann.hpp>
#include "Geom/Spatial/PointCloud.hpp"

#include <hdf5.h>
#include <cgnslib.h>
#include <pcgnslib.h>

namespace DNDS::Geom
{
    namespace
    {
        void CurrentOutputFieldCounts(UnstructuredMesh &source, int cells, int cellVectors, int points, int pointVectors,
                                      MPI_Comm outputComm = MPI_COMM_NULL)
        {
            int counts[]{cells, cellVectors, points, pointVectors, -cells, -cellVectors, -points, -pointVectors};
            MPI_Allreduce(MPI_IN_PLACE, counts, 8, MPI_INT, MPI_MIN, outputComm == MPI_COMM_NULL ? source.mpi.comm : outputComm);
            for (int i = 0; i < 4; ++i)
                DNDS_check_throw_info(counts[i] >= 0 && counts[i] == -counts[i + 4],
                                      "current mesh output field counts differ between ranks");
        }

        std::unique_ptr<MeshReadAccess> CurrentOutputRead(UnstructuredMesh &source, MPI_Comm outputComm = MPI_COMM_NULL)
        {
            std::unique_ptr<MeshReadAccess> result;
            auto mpi = source.mpi;
            if (outputComm != MPI_COMM_NULL)
                mpi.comm = outputComm;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { DNDS_check_throw_info(source.HasNativeState(), "mesh output requires native authority");
                                              result = std::make_unique<MeshReadAccess>(source.AcquireRead()); });
            return result;
        }

        /// Output borrows native current rows or explicitly gathered serial
        /// rows. The retained owner protects payload and shape for the writer.
        class OutputMeshRows
        {
            std::unique_ptr<MeshReadAccess> _read;
            int _dimension;
            bool _periodic;
            Periodicity _frame;
            std::shared_ptr<const NativeMeshSerialOutput> _serial;
            std::optional<Runtime::Array<real>::ReadLease> _serialCoords;
            std::optional<Runtime::Array<index>::ReadLease> _serialNodes;
            std::optional<Runtime::Array<ElemInfo>::ReadLease> _serialInfo;
            std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> _serialBits;

        public:
            index nCell = 0, nNode = 0;
            OutputMeshRows(UnstructuredMeshSerialRW &reader, int flag)
                : _dimension(reader.mesh->getDim()), _periodic(reader.mesh->IsPeriodic()), _frame(reader.mesh->PeriodicMetadata())
            {
                DNDS_check_throw_info(flag == 0 || flag == 1, "output mode must be serial or distributed");
                if (flag == 1)
                {
                    _read = CurrentOutputRead(*reader.mesh);
                    const auto &view = _read->HostView();
                    _dimension = view.Dimension();
                    _periodic = view.IsPeriodic();
                    _frame = view.Periodic();
                    nCell = view.NumCell();
                    nNode = view.NumNodeProc();
                }
                else
                {
                    _serial = reader.nativeSerialOutput;
                    DNDS_check_throw_info(_serial && reader.mRank == _serial->readerRank,
                                          "prepare native serial output before writing");
                    _serial->ValidateSource(*reader.mesh->CurrentNativeState());
                    _dimension = _serial->dimension;
                    _periodic = _serial->periodic;
                    _frame = _serial->periodicInfo;
                    _serialCoords.emplace(_serial->coordinates.Read());
                    _serialNodes.emplace(_serial->cellNodes.Read());
                    _serialInfo.emplace(_serial->cellInfo.Read());
                    if (_periodic)
                        _serialBits.emplace(_serial->cellNodesPbi->Read());
                    nCell = index(_serial->cellNodes.RowCount());
                    nNode = index(_serial->coordinates.RowCount());
                }
            }
            int Dimension() const { return _dimension; }
            bool IsPeriodic() const { return _periodic; }
            const Periodicity &Periodic() const { return _frame; }
            tPoint Coord(index node) const { return _serial ? tPoint(Eigen::Map<const tPoint>(_serialCoords->View().Row(size_t(node)))) : tPoint(_read->HostView().Coord(node)); }
            rowsize CellNodeCount(index cell) const { return _serial ? rowsize(_serialNodes->View().RowSize(size_t(cell))) : rowsize(_read->HostView().CellNodes(cell).size()); }
            index CellNode(index cell, rowsize slot) const { return _serial ? _serialNodes->View()(size_t(cell), size_t(slot)) : _read->HostView().CellNodes(cell)[slot]; }
            NodePeriodicBits CellNodeBits(index cell, rowsize slot) const { return _serial ? _serialBits->View()(size_t(cell), size_t(slot)) : _read->HostView().CellNodePeriodicBits(cell)[slot]; }
            Elem::ElemType CellType(index cell) const { return _serial ? _serialInfo->View()(size_t(cell), 0).getElemType() : _read->HostView().GetCellInfo(cell).getElemType(); }
            std::vector<index> CellNodes(index cell) const
            {
                std::vector<index> result(size_t(CellNodeCount(cell)));
                for (rowsize slot = 0; slot < CellNodeCount(cell); ++slot)
                    result[size_t(slot)] = CellNode(cell, slot);
                return result;
            }
        };
    }

    static void GetViolentNodeDeduplication(
        index nNode, const std::vector<tPoint> &nodesExtra, const OutputMeshRows &rows,
        std::vector<index> &nodeDedu2Old, std::vector<index> &nodeOld2Dedu)
    {
        PointCloudFunctional nodesCloud(
            [&](size_t idx) -> tPoint
            {
                if (idx < nNode)
                    return rows.Coord(index(idx));
                else
                    return nodesExtra.at(idx - nNode);
            },
            nNode + nodesExtra.size());
        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudFunctional>,
            PointCloudFunctional,
            3,
            index>;
        kdtree_t kdTree(3, nodesCloud);
        nodeOld2Dedu.resize(nodesCloud.size(), -1);
        index iNodeDeduTop{0};
        for (index i = 0; i < nodesCloud.size(); i++)
        {
            if (nodeOld2Dedu[i] != -1)
                continue;
            tPoint cn = nodesCloud[i];
#if NANOFLANN_VERSION < 0x150
            std::vector<std::pair<DNDS::index, DNDS::real>> IndicesDists;
#else
            std::vector<nanoflann::ResultItem<DNDS::index, DNDS::real>> IndicesDists;
#endif
            IndicesDists.reserve(5);
#if NANOFLANN_VERSION < 0x150
            nanoflann::SearchParams params{}; // default params
#else
            nanoflann::SearchParameters params{};
#endif
            index nFound = kdTree.radiusSearch(cn.data(), 1e-20, IndicesDists, params);
            int nSelf{0};
            for (auto &v : IndicesDists)
            {
                if (v.first == i)
                    nSelf++;
                nodeOld2Dedu.at(v.first) = iNodeDeduTop;
            }
            iNodeDeduTop++;
            DNDS_assert(nSelf == 1);
        }
        nodeDedu2Old.resize(iNodeDeduTop, -1);
        for (index i = 0; i < nodeOld2Dedu.size(); i++)
            nodeDedu2Old.at(nodeOld2Dedu.at(i)) = i;
    }

    void UnstructuredMeshSerialRW::
        PrintSerialPartPltBinaryDataArray(std::string fname,
                                          int arraySiz, int arraySizPoint,
                                          const tFGetName &names,
                                          const tFGetData &data,
                                          const tFGetName &namesPoint,
                                          const tFGetData &dataPoint,
                                          double t, int flag)
    {
        DNDS_check_throw_info(flag == 0 || flag == 1, "output mode must be serial or distributed");
        if (flag == 1)
            FieldTransferDetail::LocalStage(mesh->getMPI(), [&]
                                            { DNDS_check_throw_info(mesh->HasNativeState(), "mesh output requires native authority"); });
        else
            DNDS_check_throw_info(mesh->HasNativeState() && nativeSerialOutput,
                                  "mesh output requires native authority and prepared serial rows");
        if (flag == 1)
            CurrentOutputFieldCounts(*mesh, arraySiz, 0, arraySizPoint, 0);
        auto mpi = mesh->getMPI();

        if (mpi.rank != mRank && flag == 0) //* now only operating on mRank if serial
            return;
        OutputMeshRows outputRows(*this, flag);

        if (flag == 0)
        {
            fname += ".plt";
            std::filesystem::path outFile{fname};
            createOutputDir(outFile);
            DNDS_assert(mode == SerialOutput && dataIsSerialOut);
        }
        if (flag == 1)
        {

            std::filesystem::path outPath{fname + ".dir"};
            createOutputDirAsDir(outPath, mpi, OutputDirMode::Fast);
            std::array<char, 512> BUF{};
            std::sprintf(BUF.data(), "%06d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF.data()) + ".plt"));
        }

        std::ofstream fout(fname, std::ios::binary);
        if (!fout)
        {
            DNDS::log() << "Error: WriteMeshDebugTecASCII open \"" << fname << "\" failure" << std::endl;
            DNDS_assert(false);
        }
        const std::array<char, 9> magic_word{"#!TDV112"};
        const int b_magic_word = magic_word.size() - 1;
        int32_t intBuf = 0;
        double_t doubleBuf = 0.0;
        float_t floatBuf = 0.0F;

        auto writeInt = [&](int d) -> void
        {
            intBuf = d;
            fout.write(reinterpret_cast<char *>(&intBuf), sizeof(intBuf));
        };
        auto writeFloat = [&](float_t d) -> void
        {
            floatBuf = d;
            fout.write(reinterpret_cast<char *>(&floatBuf), sizeof(floatBuf));
        };
        auto writeDouble = [&](double_t d) -> void
        {
            doubleBuf = d;
            fout.write(reinterpret_cast<char *>(&doubleBuf), sizeof(doubleBuf));
        };
        auto writeString = [&](const std::string &title) -> void
        {
            for (auto i : title)
            {
                intBuf = static_cast<unsigned char>(i);
                fout.write(reinterpret_cast<char *>(&intBuf), sizeof(intBuf));
            }
            intBuf = 0;
            fout.write(reinterpret_cast<char *>(&intBuf), sizeof(intBuf));
        };
        fout.write(magic_word.data(), b_magic_word);
        writeInt(1);
        writeInt(0); //! full: write both grid and data
        writeString("Title Here");
        writeInt(arraySiz + 3 + 1 + arraySizPoint); // nvars
        writeString("X");
        writeString("Y");
        writeString("Z");
        for (int idata = 0; idata < arraySizPoint; idata++)
            writeString(namesPoint(idata));
        for (int idata = 0; idata < arraySiz; idata++)
            writeString(names(idata));
        writeString("iPart");

        /********************************/
        // cellZone in header
        /********************************/
        writeFloat(299.0F); // 299.0 indicates a v112 zone header, available in v191
        writeString("zone_0");
        writeInt(-1);   // ParentZone: No longer used.
        writeInt(-1);   // StrandID: static strand ID
        writeDouble(t); // solution time
        writeInt(-1);   // default zone color
        if (outputRows.Dimension() == 2)
            writeInt(3); // 2d: quad zone
        else if (outputRows.Dimension() == 3)
            writeInt(5); // 3d: brick zone
        else if (outputRows.Dimension() == 1)
            writeInt(1); // 1d: line zone
        writeInt(1);     // specifyVarLocation

        for (int idim = 0; idim < 3; idim++)
            writeInt(0); // xyz at node
        for (int idata = 0; idata < arraySizPoint; idata++)
            writeInt(0); // data at point
        for (int idata = 0; idata < arraySiz; idata++)
            writeInt(1); // data at center
        writeInt(1);     // iPart

        writeInt(0); // Are raw local 1-to-1 face neighbors supplied?
        writeInt(0); // Number of miscellaneous user-defined face

        /*******************************************************/
        // output preparation
        const index nCell = outputRows.nCell, nNode = outputRows.nNode;

        std::vector<Geom::tPoint>
            nodesExtra;
        std::vector<index> nodesExtraAtOriginal;
        if (outputRows.IsPeriodic())
        {
            for (index iCell = 0; iCell < nCell; iCell++)
                for (rowsize ic2n = 0; ic2n < outputRows.CellNodeCount(iCell); ic2n++)
                    if (outputRows.CellNodeBits(iCell, ic2n))
                        nodesExtra.push_back(outputRows.Periodic().GetCoordByBits(outputRows.Coord(outputRows.CellNode(iCell, ic2n)), outputRows.CellNodeBits(iCell, ic2n))),
                            nodesExtraAtOriginal.push_back(outputRows.CellNode(iCell, ic2n));
        }
        // if (mpi.rank == mRank)
        //     std::cout << "PrintSerialPartPltBinaryDataArray === " << std::endl;
        // auto printSize = [&]()
        // { std::cout << "Rank [" << mpi.rank << "]"
        //             << " Size: " << nNode << " " << nCell << " " << nodesExtra.size() << std::endl; };
        // if (flag == 1)
        //     MPISerialDo(mpi, printSize);
        // else
        //     printSize();
        std::vector<index> nodeDedu2Old;
        std::vector<index> nodeOld2Dedu;
        GetViolentNodeDeduplication(nNode, nodesExtra, outputRows, nodeDedu2Old, nodeOld2Dedu);
        /*******************************************************/

        writeInt(nodeDedu2Old.size()); // node number
        writeInt(nCell);               // cell number

        writeInt(0); // I dim
        writeInt(0); // J dim
        writeInt(0); // K dim
        writeInt(0); // No more Auxiliary name/value pairs.

        writeFloat(357.0F); // end of header, EOH marker

        writeFloat(299.0F); // 299.0 indicates a v112 zone header, available in v191

        /********************************/
        // cellZone data
        /********************************/

        for (int idim = 0; idim < 3; idim++)
            writeInt(2); // double for node
        for (int idata = 0; idata < arraySizPoint; idata++)
            writeInt(2); // double for data pint
        for (int idata = 0; idata < arraySiz; idata++)
            writeInt(2); // double for data
        writeInt(2);     // double for iPart

        writeInt(0);  // no passive
        writeInt(0);  // no sharing
        writeInt(-1); // no sharing

        std::vector<double_t> minVal(3 + arraySiz, DNDS::veryLargeReal);
        std::vector<double_t> maxVal(3 + arraySiz, -DNDS::veryLargeReal); // for all non-shared non-passive
        std::vector<double_t> minValPoint(arraySizPoint, DNDS::veryLargeReal);
        std::vector<double_t> maxValPoint(arraySizPoint, -DNDS::veryLargeReal); //! Tecplot is sensitive to the correctness of min/max val
        for (int idim = 0; idim < 3; idim++)
            for (DNDS::index iN : nodeDedu2Old)
            {
                if (iN < nNode)
                {
                    minVal[idim] = std::min(outputRows.Coord(iN)(idim), minVal[idim]);
                    maxVal[idim] = std::max(outputRows.Coord(iN)(idim), maxVal[idim]);
                }
                else
                {
                    minVal[idim] = std::min(nodesExtra.at(iN - nNode)(idim), minVal[idim]);
                    maxVal[idim] = std::max(nodesExtra.at(iN - nNode)(idim), maxVal[idim]);
                }
            }

        for (int idata = 0; idata < arraySiz; idata++)
            for (DNDS::index iv = 0; iv < nCell; iv++)
            {
                minVal[3 + idata] = std::min(data(idata, iv), minVal[3 + idata]);
                maxVal[3 + idata] = std::max(data(idata, iv), maxVal[3 + idata]);
            }

        for (int idata = 0; idata < arraySizPoint; idata++)
            for (DNDS::index iv = 0; iv < nNode; iv++)
            {
                minValPoint[idata] = std::min(dataPoint(idata, iv), minValPoint[idata]);
                maxValPoint[idata] = std::max(dataPoint(idata, iv), maxValPoint[idata]);
            }

        for (int idim = 0; idim < 3; idim++)
        {
            writeDouble(minVal[idim]);
            writeDouble(maxVal[idim]);
        }
        for (int idata = 0; idata < arraySizPoint; idata++)
        {
            writeDouble(minValPoint[idata]);
            writeDouble(maxValPoint[idata]);
        }
        for (int idata = 0; idata < arraySiz; idata++)
        {
            writeDouble(minVal[3 + idata]);
            writeDouble(maxVal[3 + idata]);
        }
        writeDouble(0);
        writeDouble(mpi.size);

        for (int idim = 0; idim < 3; idim++)
        {
            for (auto iN : nodeDedu2Old)
            {
                if (iN < nNode)
                    writeDouble(outputRows.Coord(iN)(idim));
                else
                    writeDouble(nodesExtra.at(iN - nNode)(idim));
            }
        }

        for (int idata = 0; idata < arraySizPoint; idata++)
        {
            for (auto iN : nodeDedu2Old)
            {
                if (iN < nNode)
                    writeDouble(dataPoint(idata, iN));
                else
                    writeDouble(dataPoint(idata, nodesExtraAtOriginal.at(iN - nNode)));
            }
        }

        for (int idata = 0; idata < arraySiz; idata++)
            for (DNDS::index iv = 0; iv < nCell; iv++)
            {
                writeDouble(data(idata, iv));
            }

        for (DNDS::index iv = 0; iv < nCell; iv++)
        {
            DNDS::MPI_int r = -1;
            DNDS::index v = -1;
            if (flag == 0)
            {
                const auto mapping = nativeSerialOutput->cellGlobal;
                bool ret = mapping->search(iv, r, v);
                DNDS_assert_info(ret, "search failed");
            }
            else if (flag == 1)
            {
                r = mesh->getMPI().rank;
                // if (mesh->localPartitionStarts.size()) // debug: for inspection of sub-partitions
                // {
                //     r *= 10000;
                //     auto &lps = mesh->localPartitionStarts;
                //     r += std::upper_bound(lps.begin(), lps.end(), iv) - lps.begin();
                // }
            }
            writeDouble(r);
        }

        index nExtra{0};

        for (DNDS::index iv = 0; iv < nCell; iv++)
        {
            auto elem = Elem::Element{outputRows.CellType(iv)};
            std::vector<index> c2n = outputRows.CellNodes(iv);
            if (outputRows.IsPeriodic())
                for (rowsize ic2n = 0; ic2n < c2n.size(); ic2n++)
                {
                    if (outputRows.CellNodeBits(iv, ic2n))
                        c2n[ic2n] = (nExtra++) + nNode;
                }

            for (auto &v : c2n)
                v = nodeOld2Dedu.at(v);
            switch (elem.GetParamSpace())
            {
            case Elem::ParamSpace::LineSpace:
                writeInt(c2n[0] + 0LL);
                writeInt(c2n[1] + 0LL);
                break;
            case Elem::ParamSpace::TriSpace:
                writeInt(c2n[0] + 0LL);
                writeInt(c2n[1] + 0LL);
                writeInt(c2n[2] + 0LL);
                writeInt(c2n[2] + 0LL);
                break;
            case Elem::ParamSpace::QuadSpace:
                writeInt(c2n[0] + 0LL);
                writeInt(c2n[1] + 0LL);
                writeInt(c2n[2] + 0LL);
                writeInt(c2n[3] + 0LL); // ! note that tis is zero based
                break;
            case Elem::ParamSpace::TetSpace:
                writeInt(c2n[0] + 0LL);
                writeInt(c2n[1] + 0LL);
                writeInt(c2n[2] + 0LL);
                writeInt(c2n[2] + 0LL);
                writeInt(c2n[3] + 0LL);
                writeInt(c2n[3] + 0LL);
                writeInt(c2n[3] + 0LL);
                writeInt(c2n[3] + 0LL);
                break;
            case Elem::ParamSpace::HexSpace:
                writeInt(c2n[0] + 0LL);
                writeInt(c2n[1] + 0LL);
                writeInt(c2n[2] + 0LL);
                writeInt(c2n[3] + 0LL);
                writeInt(c2n[4] + 0LL);
                writeInt(c2n[5] + 0LL);
                writeInt(c2n[6] + 0LL);
                writeInt(c2n[7] + 0LL);
                break;
            case Elem::ParamSpace::PrismSpace:
                writeInt(c2n[0] + 0LL);
                writeInt(c2n[1] + 0LL);
                writeInt(c2n[2] + 0LL);
                writeInt(c2n[2] + 0LL);
                writeInt(c2n[3] + 0LL);
                writeInt(c2n[4] + 0LL);
                writeInt(c2n[5] + 0LL);
                writeInt(c2n[5] + 0LL);
                break;
            case Elem::ParamSpace::PyramidSpace:
                writeInt(c2n[0] + 0LL);
                writeInt(c2n[1] + 0LL);
                writeInt(c2n[2] + 0LL);
                writeInt(c2n[3] + 0LL);
                writeInt(c2n[4] + 0LL);
                writeInt(c2n[4] + 0LL);
                writeInt(c2n[4] + 0LL);
                writeInt(c2n[4] + 0LL);
                break;
            default:
                DNDS_assert(false); //! 2d or 3d elems
            }
        };
        fout.close();
    }

    static void updateVTKSeries(std::string seriesName, std::string fname, real tSimu)
    {
        using json = nlohmann::json;
        json j;
        if (std::filesystem::exists(seriesName))
        {
            std::ifstream fin(seriesName);
            fin >> j;
            fin.close();
            DNDS_assert(j["files"].is_array());
        }
        else
        {
            j["file-series-version"] = "1.0";
            j["files"] = json::array();
        }
        std::map<std::string, real> series;
        for (auto &f : j["files"])
            series[f["name"].get<std::string>()] = f["time"].get<real>();
        series[fname] = tSimu;
        j["files"].clear();
        for (auto &[fn, tS] : series)
        {
            j["files"].emplace_back(json::object());
            j["files"].back()["name"] = fn;
            j["files"].back()["time"] = tS;
        }
        std::ofstream fout(seriesName);
        fout << j.dump(4);
        fout.close();
    }

    // =================================================================
    // File-local helpers for PrintSerialPartVTKDataArray
    // =================================================================
    namespace
    {
        /**
         * \brief Result of VTK cell topology construction.
         */
        struct VTKCellTopology
        {
            std::vector<int64_t> connectivity; ///< Flat node-index list for all cells
            std::vector<int64_t> offsets;      ///< Per-cell cumulative offset into connectivity
            std::vector<uint8_t> cellTypes;    ///< VTK cell type per cell
        };

        /**
         * \brief Build VTK connectivity, offsets, and cell-type arrays from
         *        the output mesh arrays.
         *
         * For periodic meshes, replaces periodic-boundary node references with
         * newly created "extra" node indices, then applies node deduplication.
         * Converts element node ordering to VTK convention.
         *
         * \param[in] rows             Scoped geometry and ordered incidences.
         * \param[in] nodeOld2Dedu     Old-to-deduplicated node index mapping.
         */
        VTKCellTopology BuildVTKCellTopology(
            const OutputMeshRows &rows,
            const std::vector<DNDS::index> &nodeOld2Dedu)
        {
            const auto nCell = rows.nCell, nNode = rows.nNode;
            VTKCellTopology result;
            result.offsets.resize(nCell);
            result.cellTypes.resize(nCell);
            DNDS::index cellEnd = 0;
            DNDS::index nNodeExtra{0};
            for (DNDS::index iCell = 0; iCell < nCell; iCell++)
            {
                auto elem = Elem::Element{rows.CellType(iCell)};
                std::vector<DNDS::index> c2n = rows.CellNodes(iCell);
                if (rows.IsPeriodic()) // alter the pointing
                    for (DNDS::rowsize ic2n = 0; ic2n < c2n.size(); ic2n++)
                        if (rows.CellNodeBits(iCell, ic2n))
                            c2n[ic2n] = (nNodeExtra++) + nNode;
                for (auto &v : c2n)
                    v = nodeOld2Dedu.at(v);
                auto vtkCell = Elem::ToVTKVertsAndData(elem, c2n);
                result.cellTypes[iCell] = vtkCell.first;
                for (auto in : vtkCell.second)
                    result.connectivity.push_back(in);
                cellEnd += vtkCell.second.size();
                result.offsets[iCell] = cellEnd;
            }
            return result;
        }

        /**
         * \brief Write the .pvtu parallel VTK master file referencing all
         *        per-rank .vtu piece files.
         *
         * Only called by rank 0 (mRank) when flag==1 (parallel output).
         */
        void WritePVTUMasterFile(
            const std::string &fnameIN,
            const std::filesystem::path &outPath,
            const std::string &endianName,
            const std::string &seriesName,
            double t,
            DNDS::MPI_int nMPIRanks,
            int arraySiz, int vecArraySiz,
            int arraySizPoint, int vecArraySizPoint,
            const tFGetName &names,
            const tFGetName &vectorNames,
            const tFGetName &namesPoint,
            const tFGetName &vectorNamesPoint,
            const std::string &indentV,
            const std::string &newlineV)
        {
            // Replicate writeXMLEntity locally (thin wrapper for XML output)
            auto writeXMLEntity = [&](std::ostream &out, int level, const auto &name,
                                      const std::vector<std::pair<std::string, std::string>> &attr,
                                      auto &&writeContent) -> void
            {
                for (int i = 0; i < level; i++)
                    out << indentV;
                out << "<" << name << " ";
                for (const auto &a : attr)
                    out << a.first << "=\"" << a.second << "\" ";
                out << ">" << newlineV;
                writeContent(out, level + 1);
                for (int i = 0; i < level; i++)
                    out << indentV;
                out << "</" << name << ">" << newlineV;
            };

            std::filesystem::path foutPP{fnameIN + ".pvtu"};
            std::ofstream foutP{foutPP};
            DNDS_assert(foutP);
            if (!seriesName.empty())
                updateVTKSeries(seriesName + ".pvtu.series", getStringForcePath(foutPP.filename()), t);

            writeXMLEntity(
                foutP, 0, "VTKFile",
                {{"type", "PUnstructuredGrid"},
                 {"byte_order", endianName},
                 {"header_type", "UInt64"}},
                [&](auto &out, int level)
                {
                    writeXMLEntity(
                        out, level, "PUnstructuredGrid",
                        {{"GhostLevel", "0"}},
                        [&](auto &out, int level)
                        {
                            {
                                std::string namesAll;
                                for (int i = 0; i < arraySiz; i++)
                                    namesAll.append(names(i) + " ");
                                std::string vectorsNameAll;
                                for (int i = 0; i < vecArraySiz; i++)
                                    vectorsNameAll.append(vectorNames(i) + " ");
                                writeXMLEntity(
                                    out, level, "PCellData",
                                    {{"Scalars", namesAll},
                                     {"Vectors", vectorsNameAll}},
                                    [&](auto &out, int level)
                                    {
                                        for (int i = 0; i < arraySiz; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", names(i)},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {});
                                        }
                                        for (int i = 0; i < vecArraySiz; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", vectorNames(i)},
                                                 {"NumberOfComponents", "3"},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {});
                                        }
                                    });
                            }

                            {
                                std::string namesAll;
                                for (int i = 0; i < arraySizPoint; i++)
                                    namesAll.append(namesPoint(i) + " ");
                                std::string vectorsNameAll;
                                for (int i = 0; i < vecArraySizPoint; i++)
                                    vectorsNameAll.append(vectorNamesPoint(i) + " ");
                                writeXMLEntity(
                                    out, level, "PPointData",
                                    {{"Scalars", namesAll},
                                     {"Vectors", vectorsNameAll}},
                                    [&](auto &out, int level)
                                    {
                                        for (int i = 0; i < arraySizPoint; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", namesPoint(i)},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {});
                                        }
                                        for (int i = 0; i < vecArraySizPoint; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", vectorNamesPoint(i)},
                                                 {"NumberOfComponents", "3"},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {});
                                        }
                                    });
                            }
                            writeXMLEntity(
                                out, level, "PPoints",
                                {},
                                [&](auto &out, int level)
                                {
                                    writeXMLEntity(
                                        out, level, "PDataArray",
                                        {{"type", "Float64"},
                                         {"NumberOfComponents", "3"}},
                                        [&](auto &out, int level) {});
                                });
                            for (DNDS::MPI_int iRank = 0; iRank < nMPIRanks; iRank++)
                            {
                                std::array<char, 512> BUF{};
                                std::sprintf(BUF.data(), "%04d", iRank);
                                std::string cFileNameRelPVTU = getStringForcePath(
                                    outPath.lexically_relative(outPath.parent_path()) / (std::string(BUF.data()) + ".vtu"));
                                writeXMLEntity(
                                    out, level, "Piece",
                                    {{"Source", cFileNameRelPVTU}},
                                    [&](auto &out, int level) {});
                            }
                        });
                });
        }
    } // anonymous namespace

    /**
     * @brief referencing https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html
     *
     * @param fname
     * @param arraySiz
     * @param vecArraySiz
     * @param arraySizPoint
     * @param vecArraySizPoint
     * @param names
     * @param data
     * @param vectorNames
     * @param vectorData
     * @param namesPoint
     * @param dataPoint
     * @param vectorNamesPoint
     * @param vectorDataPoint
     * @param t
     * @param flag
     */
    void UnstructuredMeshSerialRW::PrintSerialPartVTKDataArray(
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
        double t, int flag)
    {
        DNDS_check_throw_info(flag == 0 || flag == 1, "output mode must be serial or distributed");
        if (flag == 1)
            FieldTransferDetail::LocalStage(mesh->getMPI(), [&]
                                            { DNDS_check_throw_info(mesh->HasNativeState(), "mesh output requires native authority"); });
        else
            DNDS_check_throw_info(mesh->HasNativeState() && nativeSerialOutput,
                                  "mesh output requires native authority and prepared serial rows");
        if (flag == 1)
            CurrentOutputFieldCounts(*mesh, arraySiz, vecArraySiz, arraySizPoint, vecArraySizPoint);
        auto mpi = mesh->getMPI();
        std::string fnameIN = fname;

        if (mpi.rank != mRank && flag == 0) //* now only operating on mRank if serial
            return;
        OutputMeshRows outputRows(*this, flag);

        if (flag == 0)
        {
            fname += ".vtu";
            std::filesystem::path outFile{fname};
            createOutputDir(outFile);
            DNDS_assert(mode == SerialOutput && dataIsSerialOut);
            if (!seriesName.empty())
                updateVTKSeries(seriesName + ".vtu.series", getStringForcePath(outFile.filename()), t);
        }
        std::filesystem::path outPath; // only valid if parallel out
        if (flag == 1)
        {
            outPath = {fname + ".vtu.dir"};
            createOutputDirAsDir(outPath, mpi, OutputDirMode::Fast);
            std::array<char, 512> BUF{};
            std::sprintf(BUF.data(), "%04d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF.data()) + ".vtu"));
        }

        std::ofstream fout(fname);
        if (!fout)
        {
            DNDS::log() << "Error: PrintSerialPartVTKDataArray open \"" << fname << "\" failure" << std::endl;
            DNDS_assert(false);
        }

        /*******************************************************/
        // output preparation
        const index nCell = outputRows.nCell, nNode = outputRows.nNode;

        std::vector<Geom::tPoint>
            nodesExtra;
        std::vector<index> nodesExtraAtOriginal;
        if (outputRows.IsPeriodic())
        {
            for (index iCell = 0; iCell < nCell; iCell++)
                for (rowsize ic2n = 0; ic2n < outputRows.CellNodeCount(iCell); ic2n++)
                    if (outputRows.CellNodeBits(iCell, ic2n))
                        nodesExtra.push_back(outputRows.Periodic().GetCoordByBits(outputRows.Coord(outputRows.CellNode(iCell, ic2n)), outputRows.CellNodeBits(iCell, ic2n))),
                            nodesExtraAtOriginal.push_back(outputRows.CellNode(iCell, ic2n));
        }
        // if (mpi.rank == mRank)
        //     std::cout << "PrintSerialPartVTKDataArray === " << std::endl;
        // auto printSize = [&]()
        // { std::cout << "Rank [" << mpi.rank << "]"
        //             << " Size: " << nNode << " " << nCell << " " << nodesExtra.size() << std::endl; };
        // if (flag == 1)
        //     MPISerialDo(mpi, printSize);
        // else
        //     printSize();
        std::vector<index> nodeDedu2Old;
        std::vector<index> nodeOld2Dedu;
        GetViolentNodeDeduplication(nNode, nodesExtra, outputRows, nodeDedu2Old, nodeOld2Dedu);
        /*******************************************************/

        std::string indentV = "  ";
        std::string newlineV = "\n";

        auto writeXMLEntity = [&](std::ostream &out, int level, const auto &name, const std::vector<std::pair<std::string, std::string>> &attr, auto &&writeContent) -> void
        {
            for (int i = 0; i < level; i++)
                out << indentV;
            out << "<" << name << " ";
            for (const auto &a : attr)
                out << a.first << "=\"" << a.second << "\" ";
            out << ">" << newlineV;

            writeContent(out, level + 1);

            for (int i = 0; i < level; i++)
                out << indentV;
            out << "</" << name << ">" << newlineV;
        };

        auto writeCoords = [&](auto &out, int level)
        {
            writeXMLEntity(
                out, level, "Points",
                {},
                [&](auto &out, int level)
                {
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "Float64"},
                         {"NumberOfComponents", "3"},
                         {"format", vtuFloatEncodeMode}},
                        [&](auto &out, int level)
                        {
                            if (vtuFloatEncodeMode == "binary")
                            {
                                /************************/
                                // base64 inline
                                uint64_t binSize = (nodeDedu2Old.size() * 3) * sizeof(double);
                                std::vector<uint8_t> dataOutBytes;
                                dataOutBytes.resize(binSize + sizeof(binSize), 0);
                                size_t top{0};
                                *reinterpret_cast<uint64_t *>(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                for (auto iN : nodeDedu2Old)
                                {
                                    if (iN < nNode)
                                    {
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = double(outputRows.Coord(iN)(0)), top += sizeof(double);
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = double(outputRows.Coord(iN)(1)), top += sizeof(double);
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = double(outputRows.Coord(iN)(2)), top += sizeof(double);
                                    }
                                    else
                                    {
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = double(nodesExtra.at(iN - nNode)(0)), top += sizeof(double);
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = double(nodesExtra.at(iN - nNode)(1)), top += sizeof(double);
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = double(nodesExtra.at(iN - nNode)(2)), top += sizeof(double);
                                    }
                                }
                                out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                            }
                            else if (vtuFloatEncodeMode == "ascii")
                            {
                                std::vector<double> coordsOutData(nodeDedu2Old.size() * 3);
                                for (index i = 0; i < nodeDedu2Old.size(); i++)
                                {
                                    index iN = nodeDedu2Old.at(i);
                                    if (iN < nNode)
                                    {
                                        coordsOutData[i * 3 + 0LL] = outputRows.Coord(iN)(0);
                                        coordsOutData[i * 3 + 1LL] = outputRows.Coord(iN)(1);
                                        coordsOutData[i * 3 + 2LL] = outputRows.Coord(iN)(2);
                                    }
                                    else
                                    {
                                        coordsOutData[i * 3 + 0LL] = nodesExtra.at(iN - nNode)(0);
                                        coordsOutData[i * 3 + 1LL] = nodesExtra.at(iN - nNode)(1);
                                        coordsOutData[i * 3 + 2LL] = nodesExtra.at(iN - nNode)(2);
                                    }
                                }
                                // out << cppcodec::base64_rfc4648::encode(
                                //     (uint8_t *)coordsOutData.data(),
                                //     nNode * 3 * sizeof(double));
                                for (auto v : coordsOutData)
                                    out << std::setprecision(ascii_precision) << v << " ";
                            }
                            else
                                DNDS_assert(false);
                            out << newlineV;
                        });
                });
        };

        auto vtkTopo = BuildVTKCellTopology(
            outputRows, nodeOld2Dedu);
        auto &cell2nodeOutData = vtkTopo.connectivity;
        auto &cell2nodeOffsetData = vtkTopo.offsets;
        auto &cellTypeData = vtkTopo.cellTypes;

        auto writeCells = [&](auto &out, int level)
        {
            writeXMLEntity(
                out, level, "Cells",
                {},
                [&](auto &out, int level)
                {
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "Int64"},
                         {"Name", "connectivity"},
                         {"format", "ascii"}},
                        [&](auto &out, int level)
                        {
                            // out << cppcodec::base64_rfc4648::encode(
                            //     (uint8_t *)cell2nodeOutData.data(),
                            //     cell2nodeOutData.size() * sizeof(int64_t));
                            for (auto v : cell2nodeOutData)
                                out << v << " ";
                            out << newlineV;
                        });
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "Int64"},
                         {"Name", "offsets"},
                         {"format", "ascii"}},
                        [&](auto &out, int level)
                        {
                            // out << cppcodec::base64_rfc4648::encode(
                            //     (uint8_t *)cell2nodeOffsetData.data(),
                            //     cell2nodeOffsetData.size() * sizeof(int64_t));
                            for (auto v : cell2nodeOffsetData)
                                out << v << " ";
                            out << newlineV;
                        });
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "UInt8"},
                         {"Name", "types"},
                         {"format", "ascii"}},
                        [&](auto &out, int level)
                        {
                            // out << cppcodec::base64_rfc4648::encode(
                            //     (uint8_t *)cellTypeData.data(),
                            //     cellTypeData.size() * sizeof(uint8_t));
                            for (int v : cellTypeData) //! cout << uint8_t(v) is ill-posed
                                out << v << " ";
                            out << newlineV;
                        });
                });
        };

        auto writeCellData = [&](auto &out, int level)
        {
            std::string namesAll;
            for (int i = 0; i < arraySiz; i++)
                namesAll.append(names(i) + " ");
            std::string vectorsNameAll;
            for (int i = 0; i < vecArraySiz; i++)
                vectorsNameAll.append(vectorNames(i) + " ");

            writeXMLEntity(
                out, level, "CellData",
                {{"Scalars", namesAll},
                 {"Vectors", vectorsNameAll}},
                [&](auto &out, int level)
                {
                    for (int i = 0; i < arraySiz; i++)
                    {
                        writeXMLEntity(
                            out, level, "DataArray",
                            {{"type", "Float64"},
                             {"Name", names(i)},
                             {"format", vtuFloatEncodeMode}},
                            [&](auto &out, int level)
                            {
                                if (vtuFloatEncodeMode == "binary")
                                {
                                    /************************/
                                    // base64 inline
                                    uint64_t binSize = nCell * sizeof(double);
                                    std::vector<uint8_t> dataOutBytes;
                                    dataOutBytes.resize(binSize + sizeof(binSize), 0);
                                    size_t top{0};
                                    *reinterpret_cast<uint64_t *>(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (index iCell = 0; iCell < nCell; iCell++)
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = double(data(i, iCell)), top += sizeof(double);
                                    out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                                }
                                else if (vtuFloatEncodeMode == "ascii")
                                {
                                    /************************/
                                    // ascii
                                    std::vector<double> dataOutC(nCell);
                                    for (index iCell = 0; iCell < nCell; iCell++)
                                        dataOutC[iCell] = data(i, iCell);
                                    // auto dataOutCompressed = zlibCompressData((uint8_t *)dataOutC.data(), dataOutC.size() * sizeof(double));
                                    // out << cppcodec::base64_rfc4648::encode((char *)dataOutC.data(), binSize);
                                    out << std::setprecision(ascii_precision);
                                    for (auto v : dataOutC)
                                        out << v << " ";
                                }
                                else
                                    DNDS_assert(false);
                                out << newlineV;
                            });
                    }
                    writeXMLEntity(
                        out, level, "DataArray",
                        {{"type", "Int32"},
                         {"Name", "iPart"},
                         {"format", "ascii"}},
                        [&](auto &out, int level)
                        {
                            std::vector<double> dataOutC(nCell);
                            for (index iCell = 0; iCell < nCell; iCell++)
                            {
                                MPI_int r{0};
                                index v{0};
                                if (flag == 0)
                                {
                                    const auto mapping = nativeSerialOutput->cellGlobal;
                                    bool ret = mapping->search(iCell, r, v);
                                    DNDS_assert_info(ret, "search failed");
                                }
                                else if (flag == 1)
                                    r = mesh->getMPI().rank;
                                dataOutC[iCell] = r;
                            }
                            out << std::setprecision(ascii_precision);
                            for (auto v : dataOutC)
                                out << v << " ";
                            out << newlineV;
                        });
                    for (int i = 0; i < vecArraySiz; i++)
                    {
                        writeXMLEntity(
                            out, level, "DataArray",
                            {{"type", "Float64"},
                             {"Name", vectorNames(i)},
                             {"NumberOfComponents", "3"},
                             {"format", vtuFloatEncodeMode}},
                            [&](auto &out, int level)
                            {
                                if (vtuFloatEncodeMode == "binary")
                                {
                                    /************************/
                                    // base64 inline
                                    uint64_t binSize = (nCell * 3) * sizeof(double);
                                    std::vector<uint8_t> dataOutBytes;
                                    dataOutBytes.resize(binSize + sizeof(binSize), 0);
                                    size_t top{0};
                                    *reinterpret_cast<uint64_t *>(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (index iCell = 0; iCell < nCell; iCell++)
                                    {
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorData(i, iCell, 0), top += sizeof(double);
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorData(i, iCell, 1), top += sizeof(double);
                                        *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorData(i, iCell, 2), top += sizeof(double);
                                    }
                                    out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                                }
                                else if (vtuFloatEncodeMode == "ascii")
                                {
                                    /************************/
                                    // ascii
                                    std::vector<double> dataOutC(nCell * 3);
                                    for (index iCell = 0; iCell < nCell; iCell++)
                                    {
                                        dataOutC[iCell * 3 + 0LL] = vectorData(i, iCell, 0);
                                        dataOutC[iCell * 3 + 1] = vectorData(i, iCell, 1);
                                        dataOutC[iCell * 3 + 2] = vectorData(i, iCell, 2);
                                    }
                                    // out << cppcodec::base64_rfc4648::encode(
                                    //     (uint8_t *)dataOutC.data(),
                                    //     dataOutC.size() * sizeof(double));
                                    for (auto v : dataOutC)
                                        out << std::setprecision(ascii_precision) << v << " ";
                                }
                                else
                                    DNDS_assert(false);
                                out << newlineV;
                            });
                    }
                });
        };

        auto writePointData = [&](auto &out, int level)
        {
            std::string namesAll;
            for (int i = 0; i < arraySizPoint; i++)
                namesAll.append(namesPoint(i) + " ");
            std::string vectorsNameAll;
            for (int i = 0; i < vecArraySizPoint; i++)
                vectorsNameAll.append(vectorNamesPoint(i) + " ");

            writeXMLEntity(
                out, level, "PointData",
                {{"Scalars", namesAll},
                 {"Vectors", vectorsNameAll}},
                [&](auto &out, int level)
                {
                    for (int i = 0; i < arraySizPoint; i++)
                    {
                        writeXMLEntity(
                            out, level, "DataArray",
                            {{"type", "Float64"},
                             {"Name", namesPoint(i)},
                             {"format", vtuFloatEncodeMode}},
                            [&](auto &out, int level)
                            {
                                if (vtuFloatEncodeMode == "binary")
                                {
                                    /************************/
                                    // base64 inline
                                    uint64_t binSize = (nodeDedu2Old.size()) * sizeof(double);
                                    std::vector<uint8_t> dataOutBytes;
                                    dataOutBytes.resize(binSize + sizeof(binSize), 0);
                                    size_t top{0};
                                    *reinterpret_cast<uint64_t *>(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (auto iN : nodeDedu2Old)
                                    {
                                        if (iN < nNode)
                                            *reinterpret_cast<double *>(dataOutBytes.data() + top) = dataPoint(i, iN), top += sizeof(double);
                                        else
                                            *reinterpret_cast<double *>(dataOutBytes.data() + top) = dataPoint(i, nodesExtraAtOriginal.at(iN - nNode)), top += sizeof(double);
                                    }
                                    out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                                }
                                else if (vtuFloatEncodeMode == "ascii")
                                {
                                    /************************/
                                    // ascii
                                    out << std::setprecision(ascii_precision);
                                    for (auto iN : nodeDedu2Old)
                                    {
                                        if (iN < nNode)
                                            out << dataPoint(i, iN) << " ";
                                        else
                                            out << dataPoint(i, nodesExtraAtOriginal.at(iN - nNode)) << " ";
                                    }
                                }
                                else
                                    DNDS_assert(false);
                                out << newlineV;
                            });
                    }
                    for (int i = 0; i < vecArraySizPoint; i++)
                    {
                        writeXMLEntity(
                            out, level, "DataArray",
                            {{"type", "Float64"},
                             {"Name", vectorNamesPoint(i)},
                             {"NumberOfComponents", "3"},
                             {"format", vtuFloatEncodeMode}},
                            [&](auto &out, int level)
                            {
                                if (vtuFloatEncodeMode == "binary")
                                {
                                    /************************/
                                    // base64 inline
                                    uint64_t binSize = (nodeDedu2Old.size()) * 3 * sizeof(double);
                                    std::vector<uint8_t> dataOutBytes;
                                    dataOutBytes.resize(binSize + sizeof(binSize), 0);
                                    size_t top{0};
                                    *reinterpret_cast<uint64_t *>(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (auto iN : nodeDedu2Old)
                                    {
                                        if (iN < nNode)
                                        {
                                            *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorDataPoint(i, iN, 0), top += sizeof(double);
                                            *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorDataPoint(i, iN, 1), top += sizeof(double);
                                            *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorDataPoint(i, iN, 2), top += sizeof(double);
                                        }
                                        else
                                        {
                                            *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 0), top += sizeof(double);
                                            *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 1), top += sizeof(double);
                                            *reinterpret_cast<double *>(dataOutBytes.data() + top) = vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 2), top += sizeof(double);
                                        }
                                    }
                                    out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                                }
                                else if (vtuFloatEncodeMode == "ascii")
                                {
                                    /****************************/
                                    // ascii
                                    out << std::setprecision(ascii_precision);
                                    for (auto iN : nodeDedu2Old)
                                    {
                                        if (iN < nNode)
                                        {
                                            out << vectorDataPoint(i, iN, 0) << " ";
                                            out << vectorDataPoint(i, iN, 1) << " ";
                                            out << vectorDataPoint(i, iN, 2) << " ";
                                        }
                                        else
                                        {
                                            out << vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 0) << " ";
                                            out << vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 1) << " ";
                                            out << vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 2) << " ";
                                        }
                                    }
                                }
                                else
                                    DNDS_assert(false);
                                out << newlineV;
                            });
                    }
                });
        };

        std::string endianName{"BigEndian"};
        uint32_t beTest = 1; // need c++20 for STL support
        uint8_t beTestS = *reinterpret_cast<uint8_t *>(&beTest);
        if (beTestS == 1)
            endianName = "LittleEndian";

        writeXMLEntity(
            fout, 0, "VTKFile",
            {
                {"type", "UnstructuredGrid"},
                {"byte_order", endianName},
                {"header_type", "UInt64"},
                //  {"compressor", "vtkZLibDataCompressor"}
            },
            [&](auto &out, int level)
            {
                writeXMLEntity(
                    out, level, "UnstructuredGrid",
                    {},
                    [&](auto &out, int level)
                    {
                        writeXMLEntity( // https://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
                            out, level, "FieldData",
                            {},
                            [&](auto &out, int level)
                            {
                                writeXMLEntity(
                                    out, level, "DataArray",
                                    {{"type", "Float64"},
                                     {"name", "TIME"},
                                     {"NumberOfTuples", "1"},
                                     {"format", "ascii"}},
                                    [&](auto &out, int level)
                                    {
                                        out << std::setprecision(16);
                                        out << t << '\n';
                                    });
                            });
                        writeXMLEntity(
                            out, level, "Piece",
                            {{"NumberOfPoints", std::to_string(nodeDedu2Old.size())},
                             {"NumberOfCells", std::to_string(nCell)}},
                            [&](auto &out, int level)
                            {
                                /************************/
                                // coord data
                                writeCoords(out, level);
                                // std::cout << "CoordDone" << std::endl;
                                writeCells(out, level);
                                // std::cout << "CellDone" << std::endl;
                                writeCellData(out, level);
                                // std::cout << "CellDataDone" << std::endl;
                                writePointData(out, level);
                                // std::cout << "PointDataDone" << std::endl;
                            });
                    });
            });

        if (mpi.rank == mRank && flag == 1)
        {
            WritePVTUMasterFile(
                fnameIN, outPath, endianName, seriesName, t, mpi.size,
                arraySiz, vecArraySiz, arraySizPoint, vecArraySizPoint,
                names, vectorNames, namesPoint, vectorNamesPoint,
                indentV, newlineV);
        }
    }

    static void H5_WriteDataset(hid_t loc, const char *name, index nGlobal, index nOffset, index nLocal,
                                hid_t file_dataType, hid_t mem_dataType, hid_t plist_id, hid_t dcpl_id, const void *buf, int dim2 = -1)
    {
        int herr{0};
        DNDS_assert_info(nGlobal >= 0 && nLocal >= 0 && nOffset >= 0,
                         fmt::format("{},{},{}", nGlobal, nLocal, nOffset));
        int rank = dim2 >= 0 ? 2 : 1;
        std::array<hsize_t, 2> ranksFull{hsize_t(nGlobal), hsize_t(dim2)};
        std::array<hsize_t, 2> ranksFullUnlim{H5S_UNLIMITED, hsize_t(dim2)};
        std::array<hsize_t, 2> offset{hsize_t(nOffset), 0};
        std::array<hsize_t, 2> siz{hsize_t(nLocal), hsize_t(dim2)};
        hid_t memSpace = H5Screate_simple(rank, siz.data(), nullptr);
        hid_t fileSpace = H5Screate_simple(rank, ranksFull.data(), ranksFullUnlim.data());
        hid_t dset_id = H5Dcreate(loc, name, file_dataType, fileSpace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
        DNDS_assert_info(H5I_INVALID_HID != dset_id, "dataset create failed");
        herr = H5Sclose(fileSpace);
        fileSpace = H5Dget_space(dset_id);
        herr |= H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, offset.data(), nullptr, siz.data(), nullptr);
        herr |= H5Dwrite(dset_id, mem_dataType, memSpace, fileSpace, plist_id, buf);
        herr |= H5Dclose(dset_id);
        herr |= H5Sclose(fileSpace);
        herr |= H5Sclose(memSpace);
        DNDS_assert_info(herr >= 0,
                         "h5 error " + fmt::format(
                                           "nGlobal {}, nOffset {}, nLocal {}",
                                           nGlobal, nOffset, nLocal));
    }

    void UnstructuredMesh::PrintParallelVTKHDFDataArray(
        std::string fname, std::string seriesName,
        int arraySiz, int vecArraySiz, int arraySizPoint, int vecArraySizPoint,
        const tFGetName &names, const tFGetData &data,
        const tFGetName &vectorNames, const tFGetVecData &vectorData,
        const tFGetName &namesPoint, const tFGetData &dataPoint,
        const tFGetName &vectorNamesPoint, const tFGetVecData &vectorDataPoint,
        double t, MPI_Comm commDup)
    {
        PrintParallelVTKHDFDataArray(std::move(fname), std::move(seriesName), arraySiz, vecArraySiz, arraySizPoint, vecArraySizPoint,
                                     names, data, vectorNames, vectorData, namesPoint, dataPoint, vectorNamesPoint, vectorDataPoint,
                                     t, commDup, nullptr);
    }
    void UnstructuredMesh::PrintParallelVTKHDFDataArray(
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
        double t, MPI_Comm commDup, const HDF5OutSetting *settings, const VTKHDFExtraFields &extra)
    {
        const Serializer::HDF5Access hdf5Access(Serializer::HDF5Mutex());
        const auto hdf5OutSetting = settings ? *settings : this->hdf5OutSetting;
        DNDS_check_throw_info(commDup != MPI_COMM_NULL, "VTKHDF output requires its output communicator");
        EntitySpace::CheckCommunicator(mpi.comm, commDup);
        auto outputMPI = mpi;
        outputMPI.comm = commDup;
        FieldTransferDetail::LocalStage(outputMPI, [&]
                                        {
            DNDS_check_throw_info(HasNativeState(), "mesh output requires native authority");
            DNDS_check_throw_info(hdf5OutSetting.chunkSize > 0 && hdf5OutSetting.chunkSize <= size_t(std::numeric_limits<index>::max()) &&
                                      hdf5OutSetting.deflateLevel >= 0 && hdf5OutSetting.deflateLevel <= 9,
                                  "invalid VTKHDF chunk/compression settings"); });
        const index chunkSize = index(hdf5OutSetting.chunkSize), deflateLevel = hdf5OutSetting.deflateLevel;
        index settingsRange[]{chunkSize, deflateLevel, hdf5OutSetting.coll_on_data, hdf5OutSetting.coll_on_meta,
                              -chunkSize, -deflateLevel, -index(hdf5OutSetting.coll_on_data), -index(hdf5OutSetting.coll_on_meta)};
        MPI_Allreduce(MPI_IN_PLACE, settingsRange, 8, DNDS_MPI_INDEX, MPI_MIN, commDup);
        for (int i = 0; i < 4; ++i)
            DNDS_check_throw_info(settingsRange[i] == -settingsRange[i + 4], "VTKHDF settings differ between ranks");
        CurrentOutputFieldCounts(*this, arraySiz, vecArraySiz, arraySizPoint, vecArraySizPoint, commDup);
        auto retained = CurrentOutputRead(*this, commDup);
        auto nativeOutput = NativeMeshOutput::Build(*CurrentNativeState(), outputMPI);
        auto nativeCoordinates = nativeOutput.coordinates.Read();
        fname += ".vtkhdf";
        std::filesystem::path outFile{fname};
        createOutputDir(outFile, mpi, OutputDirMode::Fast);
        if (mpi.rank == mRank)
            if (!seriesName.empty())
                updateVTKSeries(seriesName + ".vtkhdf.series", getStringForcePath(outFile.filename()), t);

        const index outputNodeCount = index(nativeOutput.coordinates.RowCount());
        const index outputCellCount = index(nativeOutput.types.size());
        const real *outputCoordinates = nativeCoordinates.View().data;
        const auto &vtkCell2nodeOffsets = nativeOutput.offsets;
        const auto &vtkCellType = nativeOutput.types;
        const auto &vtkCell2node = nativeOutput.connectivity;
        const index vtkNodeOffset = nativeOutput.nodeOffset;
        const index vtkCellOffset = nativeOutput.cellOffset;
        const index vtkNNodeGlobal = nativeOutput.globalNodes;
        const index vtkNCellGlobal = nativeOutput.globalCells;
        const index vtkCell2NodeGlobalSiz = nativeOutput.globalConnectivity;
        if (!extra.cellIntegers.empty())
            FieldTransferDetail::LocalStage(outputMPI, [&]
                                            {
                for (const auto &[name, values] : extra.cellIntegers)
                    DNDS_check_throw_info(values.size() == outputCellCount, "VTKHDF integer field row count differs"); });
        auto sourcePointRow = [&](index row)
        { return nativeOutput.pointSourceRows.at(size_t(row)); };

        // MPI::AllreduceOneIndex(nNodesLocal, MPI_SUM, mpi);
        long long numberOfNodes = vtkNNodeGlobal;

        long long numberOfCells = vtkNCellGlobal;
        long long numberOfConnectivity = vtkCell2NodeGlobalSiz;

        index gSize = 0;
        index cSize = outputNodeCount;
        MPI::Allreduce(&cSize, &gSize, 1, DNDS_MPI_INDEX, MPI_SUM, commDup);
        long long numberOfNodes1 = gSize;

        gSize = 0;
        cSize = outputCellCount;
        MPI::Allreduce(&cSize, &gSize, 1, DNDS_MPI_INDEX, MPI_SUM, commDup);
        long long numberOfCells1 = gSize;

        DNDS_assert(numberOfNodes1 == numberOfNodes);
        DNDS_assert(numberOfCells1 == numberOfCells);
        // std::cout << vtkNCellGlobal << " " << vtkNNodeGlobal << " ";
        // std::cout << numberOfCells << " " << numberOfNodes << " ";
        // std::cout << std::endl;
        // return;
        // std::this_thread::sleep_for(std::chrono::seconds(1));

        herr_t herr{0};
#define H5SS DNDS_assert_info(herr >= 0, "H5 setting err")
#define H5S_Close DNDS_assert_info(herr >= 0, "H5 closing err")
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

        herr = H5Pset_fapl_mpio(plist_id, commDup, MPI_INFO_NULL), H5SS; // Set up file access property list with parallel I/O access
        herr = H5Pset_all_coll_metadata_ops(plist_id, hdf5OutSetting.coll_on_meta), H5SS;
        herr = H5Pset_coll_metadata_write(plist_id, hdf5OutSetting.coll_on_meta), H5SS;
        hid_t file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        DNDS_assert_info(H5I_INVALID_HID != file_id, "file open failed");
        herr = H5Pclose(plist_id), H5S_Close;
        DNDS_assert(herr >= 0 && file_id);

        hid_t VTKHDF_group_id = H5Gcreate(file_id, "/VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        DNDS_assert_info(H5I_INVALID_HID != file_id, "group create failed");

        {
            hid_t scalar_space = H5Screate(H5S_SCALAR);
            hid_t string_type = H5Tcreate(H5T_STRING, sizeof("UnstructuredGrid") - 1); // this is weird
            herr |= H5Tset_strpad(string_type, H5T_STR_NULLPAD);
            hid_t type_attr_id = H5Acreate(VTKHDF_group_id, "Type", string_type, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
            herr |= H5Awrite(type_attr_id, string_type, "UnstructuredGrid");
            H5Aclose(type_attr_id);
            H5Tclose(string_type);
            H5Sclose(scalar_space);
        }
        DNDS_assert_info(herr >= 0, "h5 error");

        // H5LTset_attribute_string(file_id, "VTKHDF", "Type", "UnstructuredGrid");
        std::array<long long, 2> version{1, 0};
        // herr |= H5LTset_attribute_long_long(file_id, "VTKHDF", "Version", version.data(), 2);
        {
            std::array<hsize_t, 2> dims{2, 1};
            hid_t space = H5Screate_simple(1, dims.data(), nullptr);
            hid_t type_attr_id = H5Acreate(VTKHDF_group_id, "Version", H5T_NATIVE_LLONG, space, H5P_DEFAULT, H5P_DEFAULT);
            herr |= H5Awrite(type_attr_id, H5T_NATIVE_LLONG, version.data());
            H5Aclose(type_attr_id);
            H5Sclose(space);
        }

        std::array<hsize_t, 1> numberSiz{1};
        // herr |= H5LTmake_dataset(VTKHDF_group_id, "NumberOfCells", 1, numberSiz.data(), H5T_NATIVE_LLONG, &numberOfCells);
        // herr |= H5LTmake_dataset(VTKHDF_group_id, "NumberOfPoints", 1, numberSiz.data(), H5T_NATIVE_LLONG, &numberOfNodes);
        // herr |= H5LTmake_dataset(VTKHDF_group_id, "NumberOfConnectivityIds", 1, numberSiz.data(), H5T_NATIVE_LLONG, &numberOfConnectivity);

        DNDS_assert(herr >= 0);

        plist_id = H5Pcreate(H5P_DATASET_XFER);
        if (hdf5OutSetting.coll_on_data || hdf5OutSetting.deflateLevel > 0)
            herr |= H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        else
            herr |= H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
        std::array<hsize_t, 2> chunk_dims{hdf5OutSetting.chunkSize, 3};
        hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        herr |= H5Pset_chunk(dcpl_id, 1, chunk_dims.data());
        hid_t dcpl_id_3arr = H5Pcreate(H5P_DATASET_CREATE);
        herr |= H5Pset_chunk(dcpl_id_3arr, 2, chunk_dims.data());
        DNDS_assert_info(herr >= 0, "h5 error");
#ifdef H5_HAVE_FILTER_DEFLATE
        if (hdf5OutSetting.deflateLevel > 0)
            herr = H5Pset_deflate(dcpl_id, hdf5OutSetting.deflateLevel);
        if (hdf5OutSetting.deflateLevel > 0)
            herr = H5Pset_deflate(dcpl_id_3arr, hdf5OutSetting.deflateLevel);
#endif
        int scalarSize = mpi.rank == mRank ? 1 : 0;
        H5_WriteDataset(VTKHDF_group_id, "NumberOfCells", 1, 0, scalarSize,
                        H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, plist_id, dcpl_id,
                        &numberOfCells);
        H5_WriteDataset(VTKHDF_group_id, "NumberOfPoints", 1, 0, scalarSize,
                        H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, plist_id, dcpl_id,
                        &numberOfNodes);
        H5_WriteDataset(VTKHDF_group_id, "NumberOfConnectivityIds", 1, 0, scalarSize,
                        H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, plist_id, dcpl_id,
                        &numberOfConnectivity);

        /************************************************************/
        // log() << fmt::format("coordsSize {} cellSize {} ", outputNodeCount, outputCellCount);
        // log() << fmt::format("numberOfNodes {}, numberOfCells {} numberOfCon {}", numberOfNodes, numberOfCells, numberOfConnectivity) << std::endl;
        H5_WriteDataset(VTKHDF_group_id, "Points", numberOfNodes, vtkNodeOffset, outputNodeCount,
                        H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id_3arr, outputCoordinates, 3);
        /************************************************************/
        DNDS_assert(vtkCell2nodeOffsets.size() == outputCellCount + 1);
        H5_WriteDataset(VTKHDF_group_id, "Offsets", numberOfCells + 1, vtkCellOffset, outputCellCount + (mpi.rank == mpi.size - 1 ? 1 : 0),
                        H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, plist_id, dcpl_id, vtkCell2nodeOffsets.data(), -1);
        H5_WriteDataset(VTKHDF_group_id, "Types", numberOfCells, vtkCellOffset, outputCellCount,
                        H5T_NATIVE_UINT8, H5T_NATIVE_UINT8, plist_id, dcpl_id, vtkCellType.data(), -1);
        H5_WriteDataset(VTKHDF_group_id, "Connectivity", vtkCell2NodeGlobalSiz, vtkCell2nodeOffsets.front(), vtkCell2node.size(),
                        H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, plist_id, dcpl_id, vtkCell2node.data(), -1);

        /************************************************************/

        {
            std::vector<double> cellDataBuf(outputCellCount * 3);
            hid_t CellData_group_id = H5Gcreate(VTKHDF_group_id, "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            DNDS_assert_info(H5I_INVALID_HID != CellData_group_id, "group create failed");

            for (int iArr = 0; iArr < arraySiz; iArr++)
            {
                for (index iC = 0; iC < outputCellCount; iC++)
                    cellDataBuf[iC] = data(iArr, iC);

                auto arrName = names(iArr);
                H5_WriteDataset(CellData_group_id, arrName.c_str(), numberOfCells, vtkCellOffset, outputCellCount,
                                H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id, cellDataBuf.data(), -1);
            }
            for (int iArr = 0; iArr < vecArraySiz; iArr++)
            {
                for (index iC = 0; iC < outputCellCount; iC++)
                    for (int iRow = 0; iRow < 3; iRow++)
                        cellDataBuf[iC * 3 + iRow] = vectorData(iArr, iC, iRow);

                auto arrName = vectorNames(iArr);
                H5_WriteDataset(CellData_group_id, arrName.c_str(), numberOfCells, vtkCellOffset, outputCellCount,
                                H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id_3arr, cellDataBuf.data(), 3);
            }

            for (const auto &[name, values] : extra.cellIntegers)
                H5_WriteDataset(CellData_group_id, name.c_str(), numberOfCells, vtkCellOffset, outputCellCount,
                                H5T_STD_I32LE, H5T_NATIVE_INT, plist_id, dcpl_id, values.data(), -1);

            herr = H5Gclose(CellData_group_id), H5S_Close;
        }
        {
            index iNMax = outputNodeCount;
            std::vector<double> nodeDataBuf(iNMax * 3);
            hid_t PointData_group_id = H5Gcreate(VTKHDF_group_id, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            DNDS_assert_info(H5I_INVALID_HID != PointData_group_id, "group create failed");

            for (int iArr = 0; iArr < arraySizPoint; iArr++)
            {
                for (index iN = 0; iN < iNMax; iN++)
                    nodeDataBuf[iN] = dataPoint(iArr, sourcePointRow(iN));

                auto arrName = namesPoint(iArr);
                H5_WriteDataset(PointData_group_id, arrName.c_str(), numberOfNodes, vtkNodeOffset, outputNodeCount,
                                H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id, nodeDataBuf.data(), -1);
            }

            for (int iArr = 0; iArr < vecArraySizPoint; iArr++)
            {
                for (index iN = 0; iN < iNMax; iN++)
                    for (int iRow = 0; iRow < 3; iRow++)
                        nodeDataBuf[iN * 3 + iRow] = vectorDataPoint(iArr, sourcePointRow(iN), iRow);

                auto arrName = vectorNamesPoint(iArr);
                H5_WriteDataset(PointData_group_id, arrName.c_str(), numberOfNodes, vtkNodeOffset, outputNodeCount,
                                H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id_3arr, nodeDataBuf.data(), 3);
            }

            herr = H5Gclose(PointData_group_id), H5S_Close;
        }

        herr = H5Pclose(plist_id), H5S_Close;
        herr = H5Pclose(dcpl_id), H5S_Close;
        herr = H5Pclose(dcpl_id_3arr), H5S_Close;

        herr = H5Gclose(VTKHDF_group_id), H5S_Close;
        herr = H5Fclose(file_id), H5S_Close;

        if (!extra.fieldIntegers.empty() || !extra.fieldStrings.empty())
        {
            // The VTK reader uses variable-length strings. Parallel HDF5
            // cannot write them, so append the small common dictionary once
            // all ranks have closed their collective file handles.
            MPI_Barrier(commDup);
            FieldTransferDetail::LocalStage(outputMPI, [&]
                                            {
                if (mpi.rank != mRank)
                    return;
                const hid_t serial = H5Fopen(fname.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                DNDS_assert_info(serial >= 0, "VTKHDF cannot reopen field metadata");
                const hid_t group = H5Gcreate(serial, "/VTKHDF/FieldData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                DNDS_assert_info(group >= 0, "VTKHDF cannot create FieldData");
                const auto writeField = [&](const std::string &name, hsize_t count, hid_t fileType, hid_t memoryType, const void *values)
                {
                    const hid_t space = H5Screate_simple(1, &count, nullptr);
                    DNDS_assert(space >= 0);
                    const hid_t dataset = H5Dcreate(group, name.c_str(), fileType, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    DNDS_assert(dataset >= 0);
                    if (count)
                        herr = H5Dwrite(dataset, memoryType, H5S_ALL, H5S_ALL, H5P_DEFAULT, values), H5SS;
                    herr = H5Dclose(dataset), H5S_Close;
                    herr = H5Sclose(space), H5S_Close;
                };
                for (const auto &[name, values] : extra.fieldIntegers)
                    writeField(name, values.size(), H5T_STD_I32LE, H5T_NATIVE_INT, values.data());
                if (!extra.fieldStrings.empty())
                {
                    const hid_t type = H5Tcopy(H5T_C_S1);
                    DNDS_assert(type >= 0);
                    herr = H5Tset_size(type, H5T_VARIABLE), H5SS;
                    for (const auto &[name, values] : extra.fieldStrings)
                    {
                        std::vector<const char *> strings;
                        strings.reserve(values.size());
                        for (const auto &value : values)
                            strings.push_back(value.c_str());
                        writeField(name, strings.size(), type, type, strings.data());
                    }
                    herr = H5Tclose(type), H5S_Close;
                }
                herr = H5Gclose(group), H5S_Close;
                herr = H5Fclose(serial), H5S_Close; });
        }
    }

    void UnstructuredMesh::PrintMeshCGNS(std::string fname, const t_FBCID_2_Name &fbcid2name, const std::vector<std::string> &allNames)
    {
        // Serialize current native read views without changing adjacency numbering.
        auto retained = CurrentOutputRead(*this);
        const auto view = retained->HostView();
        std::optional<EntitySpace> nodes;
        std::string options;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(!view.IsPeriodic(), "PrintMeshCGNS does not handle periodic mesh");
            DNDS_check_throw_info(view.Dimension() == 2 || view.Dimension() == 3, "CGNS mesh output requires dimension two or three");
            nodes.emplace(Entities(EntityKind::Node));
            options = nlohmann::json{{"file", fname}, {"dimension", view.Dimension()}, {"boundaries", allNames}}.dump();
            DNDS_check_throw_info(options.size() <= size_t(INT_MAX), "CGNS output options exceed MPI range");
            std::set<std::string> unique;
            for (const auto &name : allNames)
                DNDS_check_throw_info(!name.empty() && name.size() <= 32 && unique.insert(name).second,
                                      "CGNS boundary names must be distinct, nonempty and at most 32 bytes"); });
        int length = int(options.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string expected;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { expected = mpi.rank == 0 ? options : std::string(size_t(length), '\0'); });
        MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(options == expected, "CGNS output path, dimension or boundary names differ across ranks"); });

        const index nBnd = this->NumBnd();
        const index nBndGlobal = this->NumBndGlobal();
        const index nCell = this->NumCell();
        const index nCellGlobal = this->NumCellGlobal();
        const index nBndCell = nBnd + nCell;
        const index nNode = this->NumNode();
        const index nNodeGlobal = this->NumNodeGlobal();
        DNDS_check_throw_info(nNodeGlobal > 0 && nCellGlobal > 0, "CGNS output requires nonempty global nodes and cells");
        index nNodeProcOffset{0};

        MPI::Scan(&nNode, &nNodeProcOffset, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        DNDS_assert(mpi.rank != mpi.size - 1 || nNodeProcOffset == nNodeGlobal);
        nNodeProcOffset -= nNode;

        index nBndCellProcOffset{0};
        MPI::Scan(&nBndCell, &nBndCellProcOffset, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        nBndCellProcOffset -= nBndCell;

        std::vector<cgsize_t> bndCellOffsets, elementPolyData;
        std::vector<double> coordsX, coordsY, coordsZ;
        std::map<std::string, std::vector<cgsize_t>> bocoLists;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            bndCellOffsets.reserve(size_t(nBndCell) + 1);
            bndCellOffsets.push_back(0);
            auto append = [&](const ElemInfo &info, const auto &cone)
            {
                elementPolyData.push_back(_getCGNSTypeFromElemType(info.getElemType()));
                for (auto node : cone)
                    elementPolyData.push_back(nodes->LocalToGlobal(node) + 1);
                bndCellOffsets.push_back(cgsize_t(elementPolyData.size()));
            };
            for (index boundary = 0; boundary < nBnd; ++boundary)
                append(view.GetBndInfo(boundary), view.BoundaryNodes(boundary));
            for (index cell = 0; cell < nCell; ++cell)
                append(view.GetCellInfo(cell), view.CellNodes(cell));
            coordsX.resize(size_t(nNode));
            coordsY.resize(size_t(nNode));
            coordsZ.resize(size_t(nNode));
            for (index node = 0; node < nNode; ++node)
            {
                coordsX[size_t(node)] = view.Coord(node)[0];
                coordsY[size_t(node)] = view.Coord(node)[1];
                coordsZ[size_t(node)] = view.Coord(node)[2];
            }
            for (const auto &name : allNames) bocoLists.emplace(name, std::vector<cgsize_t>{});
            for (index boundary = 0; boundary < nBnd; ++boundary)
            {
                auto name = fbcid2name(view.GetBndZone(boundary));
                DNDS_check_throw_info(bocoLists.count(name), "CGNS boundary callback returned a name absent from allNames");
                bocoLists.at(name).push_back(boundary + nBndCellProcOffset + 1);
            } });
        index bndCellOffsetsProcOffset{0};
        index connectivityCount = index(elementPolyData.size());
        MPI::Scan(&connectivityCount, &bndCellOffsetsProcOffset, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        index bndCellOffsetsGlobalMax{bndCellOffsetsProcOffset};
        MPI::Bcast(&bndCellOffsetsGlobalMax, 1, DNDS_MPI_INDEX, mpi.size - 1, mpi.comm);
        bndCellOffsetsProcOffset -= bndCellOffsets.back();
        for (auto &v : bndCellOffsets)
            v += bndCellOffsetsProcOffset; // made into global offset (0-based)

        std::vector<index> bocoListLengths(allNames.size());
        for (size_t i = 0; i < allNames.size(); i++)
            bocoListLengths[i] = bocoLists[allNames[i]].size();
        std::vector<index> bocoListLengthsGlobal{bocoListLengths};
        MPI::Allreduce(bocoListLengths.data(), bocoListLengthsGlobal.data(), bocoListLengths.size(), DNDS_MPI_INDEX, MPI_SUM, mpi.comm);

        /*****************************/
        /*     CGNS operations:      */
        /*****************************/

        std::filesystem::path outFile{fname};
        createOutputDir(outFile, mpi, OutputDirMode::Fast);

        int cgns_file{0};
        DNDS_CGNS_CALL_EXIT(cgp_mpi_comm(mpi.comm));
        DNDS_CGNS_CALL_EXIT(cgp_open(fname.c_str(), CG_MODE_WRITE, &cgns_file));

        int iBase{0};
        DNDS_CGNS_CALL_EXIT(cg_base_write(cgns_file, "Base_0", view.Dimension(), 3, &iBase));
        int iZone{0};
        std::array<cgsize_t, 3> zone_sizes{nNodeGlobal, nCellGlobal, 0};
        DNDS_CGNS_CALL_EXIT(cg_zone_write(cgns_file, iBase, "Zone_0", zone_sizes.data(), Unstructured, &iZone));
        std::array<int, 3> iCoords{};
        DNDS_CGNS_CALL_EXIT(cgp_coord_write(cgns_file, iBase, iZone, RealDouble, "CoordinateX", iCoords.data()));
        DNDS_CGNS_CALL_EXIT(cgp_coord_write(cgns_file, iBase, iZone, RealDouble, "CoordinateY", &iCoords[1]));
        DNDS_CGNS_CALL_EXIT(cgp_coord_write(cgns_file, iBase, iZone, RealDouble, "CoordinateZ", &iCoords[2]));
        std::array<cgsize_t, 1> rmin{nNodeProcOffset + 1};     // note 1 based
        std::array<cgsize_t, 1> rmax{nNodeProcOffset + nNode}; // note inclusive end

        DNDS_CGNS_CALL_EXIT(cgp_coord_write_data(cgns_file, iBase, iZone, iCoords[0], rmin.data(), rmax.data(), nNode ? coordsX.data() : nullptr));
        DNDS_CGNS_CALL_EXIT(cgp_coord_write_data(cgns_file, iBase, iZone, iCoords[1], rmin.data(), rmax.data(), nNode ? coordsY.data() : nullptr));
        DNDS_CGNS_CALL_EXIT(cgp_coord_write_data(cgns_file, iBase, iZone, iCoords[2], rmin.data(), rmax.data(), nNode ? coordsZ.data() : nullptr));

        int iSec{0};
        DNDS_CGNS_CALL_EXIT(cgp_poly_section_write(cgns_file, iBase, iZone, "all_elements", ElementType_t::MIXED, 1, nCellGlobal + nBndGlobal,
                                                   bndCellOffsetsGlobalMax, 0, &iSec));

        DNDS_CGNS_CALL_EXIT(cgp_poly_elements_write_data(cgns_file, iBase, iZone, iSec,
                                                         nBndCellProcOffset + 1, nBndCellProcOffset + nBndCell,
                                                         nBndCell ? elementPolyData.data() : nullptr, nBndCell ? bndCellOffsets.data() : nullptr));

        static_assert(CGNS_VERSION >= 4500, "Need at least CGNS 4.5.0");
        for (size_t i = 0; i < allNames.size(); i++)
            if (bocoListLengthsGlobal[i] > 0)
            {
                int iBC = 0;
                DNDS_CGNS_CALL_EXIT(cg_boco_write(cgns_file, iBase, iZone, allNames[i].data(), BCType_t::BCTypeNull, PointList,
                                                  bocoListLengthsGlobal[i], nullptr, &iBC)); // use nullptr to postpone the data write
                DNDS_CGNS_CALL_EXIT(cg_boco_gridlocation_write(cgns_file, iBase, iZone, iBC, view.Dimension() == 3 ? GridLocation_t::FaceCenter : GridLocation_t::EdgeCenter));
                DNDS_CGNS_CALL_EXIT(cg_goto(cgns_file, iBase, "Zone_t", iZone, "ZoneBC_t", 1, "BC_t", iBC, "PointList", 0, ""));
                index nBCLocal = bocoListLengths[i];
                index nBCLocalProcOffSet{0};
                MPI::Scan(&nBCLocal, &nBCLocalProcOffSet, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
                nBCLocalProcOffSet -= nBCLocal;
                DNDS_CGNS_CALL_EXIT(cgp_ptlist_write_data(cgns_file, nBCLocalProcOffSet + 1, nBCLocalProcOffSet + nBCLocal,
                                                          nBCLocal ? bocoLists.at(allNames[i]).data() : nullptr));
            }

        if (cgp_close(cgns_file))
            cg_error_exit();
        // cgp_close resets HDF access mode, but leaves the file-probe MPI
        // communicator active. Keep later rank-local CGNS operations local.
        DNDS_CGNS_CALL_EXIT(cgp_mpi_comm(MPI_COMM_SELF));
    }
}
