#include "Mesh.hpp"

#include <filesystem>
#include "base64_rfc4648.hpp"
#include <zlib.h>

#include <nanoflann.hpp>
#include "PointCloud.hpp"

#include <hdf5.h>
#include <hdf5_hl.h>

namespace DNDS::Geom
{

    void UnstructuredMeshSerialRW::
        GetCurrentOutputArrays(int flag,
                               tCoordPair &coordOut,
                               tAdjPair &cell2nodeOut,
                               tPbiPair &cell2nodePbiOut,
                               tElemInfoArrayPair &cellElemInfoOut,
                               index &nCell, index &nNode)
    {
        /*************************************************/
        // get the output arrays: serial or parallel
        tCoord coordSerialDummy;
        tAdj cell2nodeSerialDummy;
        tPbi cell2nodePbiSerialDummy;
        tElemInfoArray cellElemInfoSerialDummy;
        DNDS_MAKE_SSP(coordSerialDummy, mesh->getMPI());
        DNDS_MAKE_SSP(cell2nodeSerialDummy, mesh->getMPI());
        DNDS_MAKE_SSP(cell2nodePbiSerialDummy, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(cellElemInfoSerialDummy, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());

        if (flag == 0)
        {
            coordOut.father = coordSerial;
            coordOut.son = coordSerialDummy;
            cell2nodeOut.father = cell2nodeSerial;
            cell2nodeOut.son = cell2nodeSerialDummy;
            if (mesh->isPeriodic)
            {
                cell2nodePbiOut.father = cell2nodePbiSerial;
                cell2nodePbiOut.son = cell2nodePbiSerialDummy;
            }
            cellElemInfoOut.father = cellElemInfoSerial;
            cellElemInfoOut.son = cellElemInfoSerialDummy;
            nCell = cell2nodeOut.Size();
            nNode = coordOut.Size();
        }
        else if (flag == 1)
        {
            coordOut.father = mesh->coords.father;
            coordOut.son = mesh->coords.son;
            cell2nodeOut.father = mesh->cell2node.father;
            cell2nodeOut.son = mesh->cell2node.son;
            if (mesh->isPeriodic)
            {
                cell2nodePbiOut.father = mesh->cell2nodePbi.father;
                cell2nodePbiOut.son = mesh->cell2nodePbi.son;
            }
            cellElemInfoOut.father = mesh->cellElemInfo.father;
            cellElemInfoOut.son = mesh->cellElemInfo.son;
            nCell = cell2nodeOut.father->Size(); //! only non-ghost cells are output
            nNode = coordOut.Size();             //! need all the nodes with ghost
        }
        /*************************************************/
    }

    static void GetViolentNodeDeduplication(
        index nNode, const std::vector<tPoint> &nodesExtra, tCoordPair &coordOut,
        std::vector<index> &nodeDedu2Old, std::vector<index> &nodeOld2Dedu)
    {
        PointCloudFunctional nodesCloud(
            [&](size_t idx) -> tPoint
            {
                if (idx < nNode)
                    return coordOut[idx];
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

            std::vector<std::pair<DNDS::index, DNDS::real>> IndicesDists;
            IndicesDists.reserve(5);
            nanoflann::SearchParams params{}; // default params
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
        auto mpi = mesh->getMPI();
        std::string fnameIn = fname;

        if (mpi.rank != mRank && flag == 0) //* now only operating on mRank if serial
            return;

        if (flag == 0)
        {
            fname += ".plt";
            std::filesystem::path outFile{fname};
            std::filesystem::create_directories(outFile.parent_path() / ".");
            DNDS_assert(mode == SerialOutput && dataIsSerialOut);
        }
        if (flag == 1)
        {

            std::filesystem::path outPath{fname + ".dir"};
            std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%06d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF) + ".plt"));
        }

        std::ofstream fout(fname, std::ios::binary);
        if (!fout)
        {
            DNDS::log() << "Error: WriteMeshDebugTecASCII open \"" << fname << "\" failure" << std::endl;
            DNDS_assert(false);
        }
        const char magic_word[] = "#!TDV112";
        const int b_magic_word = sizeof(magic_word) - 1;
        int32_t intBuf;
        double_t doubleBuf;
        float_t floatBuf;

        auto writeInt = [&](int d) -> void
        {
            intBuf = d;
            fout.write((char *)(&intBuf), sizeof(intBuf));
        };
        auto writeFloat = [&](float_t d) -> void
        {
            floatBuf = d;
            fout.write((char *)(&floatBuf), sizeof(floatBuf));
        };
        auto writeDouble = [&](double_t d) -> void
        {
            doubleBuf = d;
            fout.write((char *)(&doubleBuf), sizeof(doubleBuf));
        };
        auto writeString = [&](const std::string &title) -> void
        {
            for (auto i : title)
            {
                intBuf = i;
                fout.write((char *)(&intBuf), sizeof(intBuf));
            }
            intBuf = 0;
            fout.write((char *)(&intBuf), sizeof(intBuf));
        };
        fout.write(magic_word, b_magic_word);
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
        writeFloat(299.0f); // 299.0 indicates a v112 zone header, available in v191
        writeString("zone_0");
        writeInt(-1);   // ParentZone: No longer used.
        writeInt(-1);   // StrandID: static strand ID
        writeDouble(t); // solution time
        writeInt(-1);   // default zone color
        if (mesh->dim == 2)
            writeInt(3); // 2d: quad zone
        else if (mesh->dim == 3)
            writeInt(5); // 3d: brick zone
        else if (mesh->dim == 1)
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
        tCoordPair coordOut;
        tAdjPair cell2nodeOut;
        tPbiPair cell2nodePbiOut;
        tElemInfoArrayPair cellElemInfoOut;
        index nCell{-1}, nNode{-1};
        this->GetCurrentOutputArrays(flag, coordOut, cell2nodeOut, cell2nodePbiOut, cellElemInfoOut,
                                     nCell, nNode);

        std::vector<Geom::tPoint>
            nodesExtra;
        std::vector<index> nodesExtraAtOriginal;
        if (mesh->isPeriodic)
        {
            for (index iCell = 0; iCell < nCell; iCell++)
                for (rowsize ic2n = 0; ic2n < cell2nodePbiOut.RowSize(iCell); ic2n++)
                    if (cell2nodePbiOut[iCell][ic2n])
                        nodesExtra.push_back(mesh->periodicInfo.GetCoordByBits(coordOut[cell2nodeOut[iCell][ic2n]], cell2nodePbiOut[iCell][ic2n])),
                            nodesExtraAtOriginal.push_back(cell2nodeOut(iCell, ic2n));
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
        GetViolentNodeDeduplication(nNode, nodesExtra, coordOut, nodeDedu2Old, nodeOld2Dedu);
        /*******************************************************/

        writeInt(nodeDedu2Old.size()); // node number
        writeInt(nCell);               // cell number

        writeInt(0); // I dim
        writeInt(0); // J dim
        writeInt(0); // K dim
        writeInt(0); // No more Auxiliary name/value pairs.

        writeFloat(357.0f); // end of header, EOH marker

        writeFloat(299.0f); // 299.0 indicates a v112 zone header, available in v191

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
            for (DNDS::index i = 0; i < nodeDedu2Old.size(); i++)
            {
                index iN = nodeDedu2Old.at(i);
                if (iN < nNode)
                {
                    minVal[idim] = std::min(coordOut[iN](idim), minVal[idim]);
                    maxVal[idim] = std::max(coordOut[iN](idim), maxVal[idim]);
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
            for (DNDS::index i = 0; i < nodeDedu2Old.size(); i++)
            {
                DNDS::index iN = nodeDedu2Old.at(i);
                if (iN < nNode)
                    writeDouble(coordOut[iN](idim));
                else
                    writeDouble(nodesExtra.at(iN - nNode)(idim));
            }
        }

        for (int idata = 0; idata < arraySizPoint; idata++)
        {
            for (DNDS::index i = 0; i < nodeDedu2Old.size(); i++)
            {
                DNDS::index iN = nodeDedu2Old.at(i);
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
                cell2nodeSerialOutTrans.pLGlobalMapping->search(iv, r, v);
            else if (flag == 1)
                r = mesh->getMPI().rank;
            writeDouble(r);
        }

        index nExtra{0};

        for (DNDS::index iv = 0; iv < nCell; iv++)
        {
            auto elem = Elem::Element{cellElemInfoOut[iv]->getElemType()};
            std::vector<index> c2n = cell2nodeOut[iv];
            if (mesh->isPeriodic)
                for (rowsize ic2n = 0; ic2n < c2n.size(); ic2n++)
                {
                    if (cell2nodePbiOut[iv][ic2n])
                        c2n[ic2n] = (nExtra++) + nNode;
                }

            for (auto &v : c2n)
                v = nodeOld2Dedu.at(v);
            switch (elem.GetParamSpace())
            {
            case Elem::ParamSpace::LineSpace:
                writeInt(c2n[0] + 0ll);
                writeInt(c2n[1] + 0ll);
                break;
            case Elem::ParamSpace::TriSpace:
                writeInt(c2n[0] + 0ll);
                writeInt(c2n[1] + 0ll);
                writeInt(c2n[2] + 0ll);
                writeInt(c2n[2] + 0ll);
                break;
            case Elem::ParamSpace::QuadSpace:
                writeInt(c2n[0] + 0ll);
                writeInt(c2n[1] + 0ll);
                writeInt(c2n[2] + 0ll);
                writeInt(c2n[3] + 0ll); // ! note that tis is zero based
                break;
            case Elem::ParamSpace::TetSpace:
                writeInt(c2n[0] + 0ll);
                writeInt(c2n[1] + 0ll);
                writeInt(c2n[2] + 0ll);
                writeInt(c2n[2] + 0ll);
                writeInt(c2n[3] + 0ll);
                writeInt(c2n[3] + 0ll);
                writeInt(c2n[3] + 0ll);
                writeInt(c2n[3] + 0ll);
                break;
            case Elem::ParamSpace::HexSpace:
                writeInt(c2n[0] + 0ll);
                writeInt(c2n[1] + 0ll);
                writeInt(c2n[2] + 0ll);
                writeInt(c2n[3] + 0ll);
                writeInt(c2n[4] + 0ll);
                writeInt(c2n[5] + 0ll);
                writeInt(c2n[6] + 0ll);
                writeInt(c2n[7] + 0ll);
                break;
            case Elem::ParamSpace::PrismSpace:
                writeInt(c2n[0] + 0ll);
                writeInt(c2n[1] + 0ll);
                writeInt(c2n[2] + 0ll);
                writeInt(c2n[2] + 0ll);
                writeInt(c2n[3] + 0ll);
                writeInt(c2n[4] + 0ll);
                writeInt(c2n[5] + 0ll);
                writeInt(c2n[5] + 0ll);
                break;
            case Elem::ParamSpace::PyramidSpace:
                writeInt(c2n[0] + 0ll);
                writeInt(c2n[1] + 0ll);
                writeInt(c2n[2] + 0ll);
                writeInt(c2n[3] + 0ll);
                writeInt(c2n[4] + 0ll);
                writeInt(c2n[4] + 0ll);
                writeInt(c2n[4] + 0ll);
                writeInt(c2n[4] + 0ll);
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
        auto mpi = mesh->getMPI();
        std::string fnameIN = fname;

        if (mpi.rank != mRank && flag == 0) //* now only operating on mRank if serial
            return;

        if (flag == 0)
        {
            fname += ".vtu";
            std::filesystem::path outFile{fname};
            std::filesystem::create_directories(outFile.parent_path() / ".");
            DNDS_assert(mode == SerialOutput && dataIsSerialOut);
            if (seriesName.size())
                updateVTKSeries(seriesName + ".vtu.series", getStringForcePath(outFile.filename()), t);
        }
        std::filesystem::path outPath; // only valid if parallel out
        if (flag == 1)
        {
            outPath = {fname + ".vtu.dir"};
            std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%04d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF) + ".vtu"));
        }

        std::ofstream fout(fname);
        if (!fout)
        {
            DNDS::log() << "Error: PrintSerialPartVTKDataArray open \"" << fname << "\" failure" << std::endl;
            DNDS_assert(false);
        }

        /*******************************************************/
        // output preparation
        tCoordPair coordOut;
        tAdjPair cell2nodeOut;
        tPbiPair cell2nodePbiOut;
        tElemInfoArrayPair cellElemInfoOut;
        index nCell{-1}, nNode{-1};
        this->GetCurrentOutputArrays(flag, coordOut, cell2nodeOut, cell2nodePbiOut, cellElemInfoOut,
                                     nCell, nNode);

        std::vector<Geom::tPoint>
            nodesExtra;
        std::vector<index> nodesExtraAtOriginal;
        if (mesh->isPeriodic)
        {
            for (index iCell = 0; iCell < nCell; iCell++)
                for (rowsize ic2n = 0; ic2n < cell2nodePbiOut.RowSize(iCell); ic2n++)
                    if (cell2nodePbiOut[iCell][ic2n])
                        nodesExtra.push_back(mesh->periodicInfo.GetCoordByBits(coordOut[cell2nodeOut[iCell][ic2n]], cell2nodePbiOut[iCell][ic2n])),
                            nodesExtraAtOriginal.push_back(cell2nodeOut(iCell, ic2n));
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
        GetViolentNodeDeduplication(nNode, nodesExtra, coordOut, nodeDedu2Old, nodeOld2Dedu);
        /*******************************************************/

        std::string indentV = "  ";
        std::string newlineV = "\n";

        auto zlibCompressedSize = [&](index size)
        {
            return size + (size + 999) / 1000 + 12; // form vtk
        };
        int compressLevel = 5;
        auto zlibCompressData = [&](uint8_t *buf, index size)
        {
            std::vector<uint8_t> ret(zlibCompressedSize(size));
            uLongf retSize = ret.size();
            auto v = compress2(ret.data(), &retSize, buf, size, compressLevel);
            if (v != Z_OK)
                DNDS_assert_info(false, "compression failed");
            ret.resize(retSize);
            return ret;
        };

        auto writeXMLEntity = [&](std::ostream &out, int level, const auto &name, const std::vector<std::pair<std::string, std::string>> &attr, auto &&writeContent) -> void
        {
            for (int i = 0; i < level; i++)
                out << indentV;
            out << "<" << name << " ";
            for (auto &a : attr)
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
                                *(uint64_t *)(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                for (index i = 0; i < nodeDedu2Old.size(); i++)
                                {
                                    index iN = nodeDedu2Old.at(i);
                                    if (iN < nNode)
                                    {
                                        *(double *)(dataOutBytes.data() + top) = double(coordOut[iN](0)), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = double(coordOut[iN](1)), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = double(coordOut[iN](2)), top += sizeof(double);
                                    }
                                    else
                                    {
                                        *(double *)(dataOutBytes.data() + top) = double(nodesExtra.at(iN - nNode)(0)), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = double(nodesExtra.at(iN - nNode)(1)), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = double(nodesExtra.at(iN - nNode)(2)), top += sizeof(double);
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
                                        coordsOutData[i * 3 + 0ll] = coordOut[iN](0);
                                        coordsOutData[i * 3 + 1ll] = coordOut[iN](1);
                                        coordsOutData[i * 3 + 2ll] = coordOut[iN](2);
                                    }
                                    else
                                    {
                                        coordsOutData[i * 3 + 0ll] = nodesExtra.at(iN - nNode)(0);
                                        coordsOutData[i * 3 + 1ll] = nodesExtra.at(iN - nNode)(1);
                                        coordsOutData[i * 3 + 2ll] = nodesExtra.at(iN - nNode)(2);
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

        std::vector<int64_t> cell2nodeOutData;
        cell2nodeOutData.reserve(cell2nodeOut.father->DataSize() * 2);
        std::vector<int64_t> cell2nodeOffsetData(nCell);
        std::vector<uint8_t> cellTypeData(nCell);
        index cellEnd = 0;
        index nNodeExtra{0};
        for (index iCell = 0; iCell < nCell; iCell++)
        {
            auto elem = Elem::Element{cellElemInfoOut[iCell]->getElemType()};
            std::vector<index> c2n = cell2nodeOut[iCell];
            if (mesh->isPeriodic) // alter the pointing
                for (rowsize ic2n = 0; ic2n < c2n.size(); ic2n++)
                    if (cell2nodePbiOut(iCell, ic2n))
                        c2n[ic2n] = (nNodeExtra++) + nNode;
            for (auto &v : c2n)
                v = nodeOld2Dedu.at(v);
            auto vtkCell = Elem::ToVTKVertsAndData(elem, c2n);
            cellTypeData[iCell] = vtkCell.first;
            for (auto in : vtkCell.second)
                cell2nodeOutData.push_back(in);
            cellEnd += vtkCell.second.size();
            cell2nodeOffsetData[iCell] = cellEnd;
        }

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
                                    *(uint64_t *)(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (index iCell = 0; iCell < nCell; iCell++)
                                        *(double *)(dataOutBytes.data() + top) = double(data(i, iCell)), top += sizeof(double);
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
                                    cell2nodeSerialOutTrans.pLGlobalMapping->search(iCell, r, v);
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
                                    *(uint64_t *)(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (index iCell = 0; iCell < nCell; iCell++)
                                    {
                                        *(double *)(dataOutBytes.data() + top) = vectorData(i, iCell, 0), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = vectorData(i, iCell, 1), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = vectorData(i, iCell, 2), top += sizeof(double);
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
                                        dataOutC[iCell * 3 + 0ll] = vectorData(i, iCell, 0);
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
                                    *(uint64_t *)(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (index ii = 0; ii < nodeDedu2Old.size(); ii++)
                                    {
                                        index iN = nodeDedu2Old.at(ii);
                                        if (iN < nNode)
                                            *(double *)(dataOutBytes.data() + top) = dataPoint(i, iN), top += sizeof(double);
                                        else
                                            *(double *)(dataOutBytes.data() + top) = dataPoint(i, nodesExtraAtOriginal.at(iN - nNode)), top += sizeof(double);
                                    }
                                    out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                                }
                                else if (vtuFloatEncodeMode == "ascii")
                                {
                                    /************************/
                                    // ascii
                                    out << std::setprecision(ascii_precision);
                                    for (index ii = 0; ii < nodeDedu2Old.size(); ii++)
                                    {
                                        index iN = nodeDedu2Old.at(ii);
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
                                    *(uint64_t *)(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (index ii = 0; ii < nodeDedu2Old.size(); ii++)
                                    {
                                        index iN = nodeDedu2Old.at(ii);
                                        if (iN < nNode)
                                        {
                                            *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, iN, 0), top += sizeof(double);
                                            *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, iN, 1), top += sizeof(double);
                                            *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, iN, 2), top += sizeof(double);
                                        }
                                        else
                                        {
                                            *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 0), top += sizeof(double);
                                            *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 1), top += sizeof(double);
                                            *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, nodesExtraAtOriginal.at(iN - nNode), 2), top += sizeof(double);
                                        }
                                    }
                                    out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                                }
                                else if (vtuFloatEncodeMode == "ascii")
                                {
                                    /****************************/
                                    // ascii
                                    out << std::setprecision(ascii_precision);
                                    for (index ii = 0; ii < nodeDedu2Old.size(); ii++)
                                    {
                                        index iN = nodeDedu2Old.at(ii);
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
        uint8_t beTestS = *(uint8_t *)(&beTest);
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
            std::filesystem::path foutPP{fnameIN + ".pvtu"};
            std::ofstream foutP{foutPP};
            DNDS_assert(foutP);
            if (seriesName.size())
                updateVTKSeries(seriesName + ".pvtu.series", getStringForcePath(foutPP.filename()), t);

            writeXMLEntity(
                foutP, 0, "VTKFile",
                {{"type", "PUnstructuredGrid"},
                 {"byte_order", endianName},
                 {"header_type", "UInt64"}}, // ! use uint64_t as base64 Header
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
                                                [&](auto &out, int level) {
                                                });
                                        }
                                        for (int i = 0; i < vecArraySiz; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", vectorNames(i)},
                                                 {"NumberOfComponents", "3"},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {
                                                });
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
                                                [&](auto &out, int level) {
                                                });
                                        }
                                        for (int i = 0; i < vecArraySizPoint; i++)
                                        {
                                            writeXMLEntity(
                                                out, level, "PDataArray",
                                                {{"type", "Float64"},
                                                 {"Name", vectorNamesPoint(i)},
                                                 {"NumberOfComponents", "3"},
                                                 {"format", "ascii"}},
                                                [&](auto &out, int level) {
                                                });
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
                                        [&](auto &out, int level) {
                                        });
                                });
                            for (MPI_int iRank = 0; iRank < mpi.size; iRank++)
                            {
                                char BUF[512];
                                std::sprintf(BUF, "%04d", iRank);
                                std::string cFileName = getStringForcePath(outPath / (std::string(BUF) + ".vtu"));
                                std::string cFileNameRelPVTU = getStringForcePath(outPath.lexically_relative(outPath.parent_path()) / (std::string(BUF) + ".vtu"));
                                writeXMLEntity(
                                    out, level, "Piece",
                                    {{"Source", cFileNameRelPVTU}},
                                    [&](auto &out, int level) {

                                    });
                            }
                        });
                });
        }
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
        double t)
    {
        fname += ".vtkhdf";
        std::filesystem::path outFile{fname};
        if (mpi.rank == mRank)
            std::filesystem::create_directories(outFile.parent_path() / ".");
        if (mpi.rank == mRank)
            if (seriesName.size())
                updateVTKSeries(seriesName + ".vtu.series", getStringForcePath(outFile.filename()), t);

        herr_t herr{0};
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        herr = H5Pset_fapl_mpio(plist_id, mpi.comm, MPI_INFO_NULL); // Set up file access property list with parallel I/O access
        herr = H5Pset_all_coll_metadata_ops(plist_id, true);
        herr = H5Pset_coll_metadata_write(plist_id, true);
        hid_t file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        herr = H5Pclose(plist_id);
        DNDS_assert(herr >= 0 && file_id);

        hid_t VTKHDF_group_id = H5Gcreate(file_id, "/VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        {
            hid_t scalar_space = H5Screate(H5S_SCALAR);
            hid_t string_type = H5Tcreate(H5T_STRING, sizeof("UnstructuredGrid") - 1); // this is wierd
            herr = H5Tset_strpad(string_type, H5T_STR_NULLPAD);
            hid_t type_attr_id = H5Acreate(VTKHDF_group_id, "Type", string_type, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
            herr = H5Awrite(type_attr_id, string_type, "UnstructuredGrid");
            H5Aclose(type_attr_id);
            H5Tclose(string_type);
            H5Sclose(scalar_space);
        }

        // H5LTset_attribute_string(file_id, "VTKHDF", "Type", "UnstructuredGrid");
        std::array<long long, 2>
            version{1, 0};
        H5LTset_attribute_long_long(file_id, "VTKHDF", "Version", version.data(), 2);

        if (isPeriodic)
            DNDS_assert(coordsPeriodicRecreated.father);
        else
            DNDS_assert(coords.father);
        tCoord coordOut = isPeriodic ? coordsPeriodicRecreated.father : coords.father;
        long long numberOfNodes = coordOut->globalSize();
        DNDS_assert(cell2node.father);
        long long numberOfCells = cell2node.father->globalSize();
        long long numberOfConnectivity = vtkCell2NodeGlobalSiz;
        std::array<hsize_t, 1> numberSiz{1};
        H5LTmake_dataset(VTKHDF_group_id, "NumberOfCells", 1, numberSiz.data(), H5T_NATIVE_LLONG, &numberOfCells);
        H5LTmake_dataset(VTKHDF_group_id, "NumberOfPoints", 1, numberSiz.data(), H5T_NATIVE_LLONG, &numberOfNodes);
        H5LTmake_dataset(VTKHDF_group_id, "NumberOfConnectivityIds", 1, numberSiz.data(), H5T_NATIVE_LLONG, &numberOfConnectivity);

        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
        std::array<hsize_t, 2> chunk_dims{hdf5OutSetting.chunkSize, 3};
        hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        herr = H5Pset_chunk(dcpl_id, 1, chunk_dims.data());
        hdf5OutSetting.deflateLevel = 0;
        if (hdf5OutSetting.deflateLevel > 0)
            herr = H5Pset_deflate(dcpl_id, hdf5OutSetting.deflateLevel);
        hid_t dcpl_id_3arr = H5Pcreate(H5P_DATASET_CREATE);
        if (hdf5OutSetting.deflateLevel > 0)
            herr = H5Pset_deflate(dcpl_id_3arr, hdf5OutSetting.deflateLevel);
        herr = H5Pset_chunk(dcpl_id_3arr, 2, chunk_dims.data());

        auto H5_WriteDataset =
            [](hid_t loc, const char *name, index nGlobal, index nOffset, index nLocal,
               hid_t file_dataType, hid_t mem_dataType, hid_t plist_id, hid_t dcpl_id, void *buf, int dim2 = -1)
        {
            int herr{0};
            DNDS_assert(nGlobal >= 0 && nLocal >= 0 && nOffset >= 0);
            int rank = dim2 >= 0 ? 2 : 1;
            std::array<hsize_t, 2> ranksFull{hsize_t(nGlobal), hsize_t(dim2)};
            std::array<hsize_t, 2> offset{hsize_t(nOffset), 0};
            std::array<hsize_t, 2> siz{hsize_t(nLocal), hsize_t(dim2)};
            hid_t memSpace = H5Screate_simple(rank, siz.data(), NULL);
            hid_t fileSpace = H5Screate_simple(rank, ranksFull.data(), NULL);
            hid_t dset_id = H5Dcreate(loc, name, file_dataType, fileSpace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
            herr = H5Sclose(fileSpace);
            fileSpace = H5Dget_space(dset_id);
            H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, offset.data(), NULL, siz.data(), NULL);
            herr = H5Dwrite(dset_id, mem_dataType, memSpace, fileSpace, plist_id, buf);
            herr = H5Dclose(dset_id);
            herr = H5Sclose(fileSpace);
            herr = H5Sclose(memSpace);
        };
        /************************************************************/
        H5_WriteDataset(VTKHDF_group_id, "Points", numberOfNodes, vtkNodeOffset, coordOut->Size(),
                        H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id_3arr, coordOut->data(), 3);
        /************************************************************/
        DNDS_assert(vtkCell2nodeOffsets.size() == cell2node.father->Size() + 1);
        H5_WriteDataset(VTKHDF_group_id, "Offsets", numberOfCells + 1, vtkCellOffset, cell2node.father->Size() + (mpi.rank == mpi.size - 1 ? 1 : 0),
                        H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, plist_id, dcpl_id, vtkCell2nodeOffsets.data(), -1);
        H5_WriteDataset(VTKHDF_group_id, "Types", numberOfCells, vtkCellOffset, cell2node.father->Size(),
                        H5T_NATIVE_UINT8, H5T_NATIVE_UINT8, plist_id, dcpl_id, vtkCellType.data(), -1);
        H5_WriteDataset(VTKHDF_group_id, "Connectivity", vtkCell2NodeGlobalSiz, vtkCell2nodeOffsets.front(), vtkCell2node.size(),
                        H5T_NATIVE_LLONG, H5T_NATIVE_LLONG, plist_id, dcpl_id, vtkCell2node.data(), -1);

        /************************************************************/

        {
            std::vector<double> cellDataBuf(cell2node.father->Size() * 3);
            hid_t CellData_group_id = H5Gcreate(VTKHDF_group_id, "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            for (int iArr = 0; iArr < arraySiz; iArr++)
            {
                for (index iC = 0; iC < cell2node.father->Size(); iC++)
                    cellDataBuf[iC] = data(iArr, iC);

                auto arrName = names(iArr);
                H5_WriteDataset(CellData_group_id, arrName.c_str(), numberOfCells, vtkCellOffset, cell2node.father->Size(),
                                H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id, cellDataBuf.data(), -1);
            }
            for (int iArr = 0; iArr < vecArraySiz; iArr++)
            {
                for (index iC = 0; iC < cell2node.father->Size(); iC++)
                    for (int iRow = 0; iRow < 3; iRow++)
                        cellDataBuf[iC * 3 + iRow] = vectorData(iArr, iC, iRow);

                auto arrName = vectorNames(iArr);
                H5_WriteDataset(CellData_group_id, arrName.c_str(), numberOfCells, vtkCellOffset, cell2node.father->Size(),
                                H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id_3arr, cellDataBuf.data(), 3);
            }

            herr = H5Gclose(CellData_group_id);
        }
        if (isPeriodic)
        {
            DNDS_assert(coordsPeriodicRecreated.father && coordsPeriodicRecreated.father->Size() == nodeRecreated2nodeLocal.size());
        }

        {
            index iNMax = coordOut->Size();
            std::vector<double> nodeDataBuf(iNMax * 3);
            hid_t PointData_group_id = H5Gcreate(VTKHDF_group_id, "PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            for (int iArr = 0; iArr < arraySizPoint; iArr++)
            {
                for (index iN = 0; iN < iNMax; iN++)
                    nodeDataBuf[iN] = dataPoint(iArr, isPeriodic ? nodeRecreated2nodeLocal.at(iN) : iN);

                auto arrName = namesPoint(iArr);
                H5_WriteDataset(PointData_group_id, arrName.c_str(), numberOfNodes, vtkNodeOffset, coordOut->Size(),
                                H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id, nodeDataBuf.data(), -1);
            }

            for (int iArr = 0; iArr < vecArraySizPoint; iArr++)
            {
                for (index iN = 0; iN < iNMax; iN++)
                    for (int iRow = 0; iRow < 3; iRow++)
                        nodeDataBuf[iN * 3 + iRow] = vectorDataPoint(iArr, isPeriodic ? nodeRecreated2nodeLocal.at(iN) : iN, iRow);

                auto arrName = vectorNamesPoint(iArr);
                H5_WriteDataset(PointData_group_id, arrName.c_str(), numberOfNodes, vtkNodeOffset, coordOut->Size(),
                                H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, plist_id, dcpl_id_3arr, nodeDataBuf.data(), 3);
            }

            herr = H5Gclose(PointData_group_id);
        }

        herr = H5Pclose(plist_id);

        herr = H5Gclose(VTKHDF_group_id);
        herr = H5Fclose(file_id);
    }
}