#include "Mesh.hpp"

#include <filesystem>
#include "base64_rfc4648.hpp"
#include <zlib.h>

#include <nanoflann.hpp>
#include "PointCloud.hpp"

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
                                          const std::function<std::string(int)> &names,
                                          const std::function<DNDS::real(int, DNDS::index)> &data,
                                          const std::function<std::string(int)> &namesPoint,
                                          const std::function<DNDS::real(int, DNDS::index)> &dataPoint,
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
        const std::function<std::string(int)> &names,
        const std::function<DNDS::real(int, DNDS::index)> &data,
        const std::function<std::string(int)> &vectorNames,
        const std::function<DNDS::real(int, DNDS::index, DNDS::rowsize)> &vectorData,
        const std::function<std::string(int)> &namesPoint,
        const std::function<DNDS::real(int, DNDS::index)> &dataPoint,
        const std::function<std::string(int)> &vectorNamesPoint,
        const std::function<DNDS::real(int, DNDS::index, DNDS::rowsize)> &vectorDataPoint,
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
                                        coordsOutData[(i + nNode) * 3 + 0ll] = nodesExtra.at(iN - nNode)(0);
                                        coordsOutData[(i + nNode) * 3 + 1ll] = nodesExtra.at(iN - nNode)(1);
                                        coordsOutData[(i + nNode) * 3 + 2ll] = nodesExtra.at(iN - nNode)(2);
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
}