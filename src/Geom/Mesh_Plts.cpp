#include "Mesh.hpp"

#include <filesystem>
#include "base64_rfc4648.hpp"
#include <zlib.h>

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
        /*******************************************************/

        writeInt(nNode + nodesExtra.size()); // node number
        writeInt(nCell);                     // cell number

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
            for (DNDS::index i = 0; i < nNode; i++)
            {
                if (i < nNode)
                {
                    minVal[idim] = std::min(coordOut[i](idim), minVal[idim]);
                    maxVal[idim] = std::max(coordOut[i](idim), maxVal[idim]);
                }
            }

        for (int idim = 0; idim < 3; idim++)
            for (DNDS::index i = 0; i < nodesExtra.size(); i++)
            {
                if (i < nNode)
                {
                    minVal[idim] = std::min(nodesExtra[i](idim), minVal[idim]);
                    maxVal[idim] = std::max(nodesExtra[i](idim), maxVal[idim]);
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
            for (DNDS::index i = 0; i < nNode; i++)
            {
                writeDouble(coordOut[i](idim));
                // std::cout << (*coordSerial)[i](idim) << std::endl;
            }
            for (DNDS::index i = 0; i < nodesExtra.size(); i++)
            {
                writeDouble(nodesExtra[i](idim));
                // std::cout << (*coordSerial)[i](idim) << std::endl;
            }
        }

        for (int idata = 0; idata < arraySizPoint; idata++)
        {
            for (DNDS::index in = 0; in < nNode; in++)
            {
                writeDouble(dataPoint(idata, in));
            }
            index nExtra{0};
            // if (mesh->isPeriodic)
            //     for (DNDS::index iv = 0; iv < nCell; iv++)
            //         for (rowsize ic2n = 0; ic2n < cell2nodeOut.RowSize(iv); ic2n++)
            //             if (cell2nodePbiOut[iv][ic2n])
            //                 writeDouble(dataPoint(idata, cell2nodeOut(iv, ic2n)));
            for (auto i : nodesExtraAtOriginal)
                writeDouble(dataPoint(idata, i));
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

    void UnstructuredMeshSerialRW::PrintSerialPartVTKDataArray(
        std::string fname,
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
                                uint64_t binSize = ((nNode + nodesExtra.size()) * 3) * sizeof(double);
                                std::vector<uint8_t> dataOutBytes;
                                dataOutBytes.resize(binSize + sizeof(binSize), 0);
                                size_t top{0};
                                *(uint64_t *)(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                for (index i = 0; i < nNode; i++)
                                {
                                    *(double *)(dataOutBytes.data() + top) = double(coordOut[i](0)), top += sizeof(double);
                                    *(double *)(dataOutBytes.data() + top) = double(coordOut[i](1)), top += sizeof(double);
                                    *(double *)(dataOutBytes.data() + top) = double(coordOut[i](2)), top += sizeof(double);
                                }
                                for (index i = 0; i < nodesExtra.size(); i++)
                                {
                                    *(double *)(dataOutBytes.data() + top) = double(nodesExtra[i](0)), top += sizeof(double);
                                    *(double *)(dataOutBytes.data() + top) = double(nodesExtra[i](1)), top += sizeof(double);
                                    *(double *)(dataOutBytes.data() + top) = double(nodesExtra[i](2)), top += sizeof(double);
                                }
                                out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                            }
                            else if (vtuFloatEncodeMode == "ascii")
                            {
                                std::vector<double> coordsOutData((nNode + nodesExtra.size()) * 3);
                                for (index i = 0; i < nNode; i++)
                                {
                                    coordsOutData[i * 3 + 0ll] = coordOut[i](0);
                                    coordsOutData[i * 3 + 1ll] = coordOut[i](1);
                                    coordsOutData[i * 3 + 2ll] = coordOut[i](2);
                                }
                                for (index i = 0; i < nodesExtra.size(); i++)
                                {
                                    coordsOutData[(i + nNode) * 3 + 0ll] = nodesExtra[i](0);
                                    coordsOutData[(i + nNode) * 3 + 1ll] = nodesExtra[i](1);
                                    coordsOutData[(i + nNode) * 3 + 2ll] = nodesExtra[i](2);
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
        cell2nodeOutData.reserve(cell2nodeOut.father->DataSize());
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
                                    uint64_t binSize = (nNode + nodesExtraAtOriginal.size()) * sizeof(double);
                                    std::vector<uint8_t> dataOutBytes;
                                    dataOutBytes.resize(binSize + sizeof(binSize), 0);
                                    size_t top{0};
                                    *(uint64_t *)(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (index ii = 0; ii < nNode; ii++)
                                        *(double *)(dataOutBytes.data() + top) = dataPoint(i, ii), top += sizeof(double);
                                    for (auto ii : nodesExtraAtOriginal)
                                        *(double *)(dataOutBytes.data() + top) = dataPoint(i, ii), top += sizeof(double);

                                    out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                                }
                                else if (vtuFloatEncodeMode == "ascii")
                                {
                                    /************************/
                                    // ascii
                                    out << std::setprecision(ascii_precision);
                                    for (index ii = 0; ii < nNode; ii++)
                                        out << dataPoint(i, ii) << " ";
                                    // if (mesh->isPeriodic)
                                    //     for (index iCell = 0; iCell < nCell; iCell++)
                                    //         for (rowsize ic2n = 0; ic2n < cell2nodeOut.RowSize(iCell); ic2n++)
                                    //             if (cell2nodePbiOut(iCell, ic2n))
                                    //                 out << std::setprecision(ascii_precision) << dataPoint(i, cell2nodeOut(iCell, ic2n)) << " ";
                                    for (auto ii : nodesExtraAtOriginal)
                                        out << dataPoint(i, ii) << " ";
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
                                    uint64_t binSize = (nNode + nodesExtraAtOriginal.size()) * 3 * sizeof(double);
                                    std::vector<uint8_t> dataOutBytes;
                                    dataOutBytes.resize(binSize + sizeof(binSize), 0);
                                    size_t top{0};
                                    *(uint64_t *)(dataOutBytes.data() + top) = binSize, top += sizeof(uint64_t);
                                    for (index ii = 0; ii < nNode; ii++)
                                    {
                                        *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, ii, 0), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, ii, 1), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, ii, 2), top += sizeof(double);
                                    }
                                    for (auto ii : nodesExtraAtOriginal)
                                    {
                                        *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, ii, 0), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, ii, 1), top += sizeof(double);
                                        *(double *)(dataOutBytes.data() + top) = vectorDataPoint(i, ii, 2), top += sizeof(double);
                                    }
                                    out << cppcodec::base64_rfc4648::encode(dataOutBytes);
                                }
                                else if (vtuFloatEncodeMode == "ascii")
                                {
                                    /****************************/
                                    // ascii
                                    out << std::setprecision(ascii_precision);
                                    for (index ii = 0; ii < nNode; ii++)
                                    {
                                        out << vectorDataPoint(i, ii, 0) << " ";
                                        out << vectorDataPoint(i, ii, 1) << " ";
                                        out << vectorDataPoint(i, ii, 2) << " ";
                                    }
                                    // std::cout << "print vec" << std::endl;
                                    // if (mesh->isPeriodic)
                                    //     for (index iCell = 0; iCell < nCell; iCell++)
                                    //         for (rowsize ic2n = 0; ic2n < cell2nodeOut.RowSize(iCell); ic2n++)
                                    //             if (cell2nodePbiOut(iCell, ic2n))
                                    //             {
                                    //                 out << std::setprecision(ascii_precision) << vectorDataPoint(i, cell2nodeOut(iCell, ic2n), 0) << " ";
                                    //                 out << std::setprecision(ascii_precision) << vectorDataPoint(i, cell2nodeOut(iCell, ic2n), 1) << " ";
                                    //                 out << std::setprecision(ascii_precision) << vectorDataPoint(i, cell2nodeOut(iCell, ic2n), 2) << " ";
                                    //             }
                                    for (auto ii : nodesExtraAtOriginal)
                                    {
                                        out << vectorDataPoint(i, ii, 0) << " ";
                                        out << vectorDataPoint(i, ii, 1) << " ";
                                        out << vectorDataPoint(i, ii, 2) << " ";
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
                        writeXMLEntity(
                            out, level, "Piece",
                            {{"NumberOfPoints", std::to_string(nNode + nodesExtra.size())},
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
            std::ofstream foutP{fnameIN + ".pvtu"};
            DNDS_assert(foutP);

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