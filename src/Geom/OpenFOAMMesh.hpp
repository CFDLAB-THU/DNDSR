#pragma once

#include "Geometric.hpp"
#include "Elements.hpp"
#include <map>

namespace DNDS::Geom::OpenFOAM
{
    struct OpenFOAMBoundaryCondition
    {
        std::string type;
        index nFaces;
        index startFace;
    };

    inline int passOpenFOAMSpaces(std::istream &in)
    {
        int wsCount = 0;
        while (!in.eof())
        {
            char c = in.peek();
            if (std::isspace(c))
            {
                in.get();
                wsCount++;
                continue;
            }
            if (c == '/')
            {
                try
                {
                    in.get();
                    c = in.peek();
                    if (c == '/') // line comment
                    {
                        std::string line;
                        std::getline(in, line);
                    }
                    else if (c == '*') // block comment
                    {
                        while (!in.eof())
                        {
                            c = in.get();
                            // std::cout << c << " : ";
                            if (c == '*')
                            {
                                c = in.get();
                                if (c == '/')
                                    break;
                            }
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "seems in a comment but failed to parse it: ";
                    std::cerr << e.what() << '\n';
                }
                wsCount++;
                continue;
            }
            break;
        }
        return wsCount;
    }

    inline bool getUntil(std::istream &in, char v, std::string &buf)
    {
        // std::cout << "getUntil " << v << std::endl;
        buf.clear();
        while (!in.eof())
        {
            if (passOpenFOAMSpaces(in))
            {
                buf.push_back(' ');
                continue;
            }
            char c = in.peek();
            if (c == v)
                break;
            in.get();
            buf.push_back(c);
        }
        // std::cout << "getUntil " << v << "is done" << std::endl;
        return in.eof();
    }

    inline void readOpenFOAMList(std::istream &in, const std::function<void(std::istream &)> readOneItem)
    {
        bool gotLeft = false;
        std::string buf;
        if (getUntil(in, '(', buf))
            DNDS_assert(false);
        in.get(); // omit the '('
        while (!in.eof())
        {

            char c = in.peek();
            if (c == ')')
            {
                in.get(); // omit the ')'
                break;
            }
            readOneItem(in);
            passOpenFOAMSpaces(in);
        }
    }

    inline void readOpenFOAMObj(std::istream &in, const std::function<void(const std::string &)> readOneItemLine)
    {
        bool gotLeft = false;
        std::string buf;
        if (getUntil(in, '{', buf))
            DNDS_assert(false);
        in.get(); // omit the '{'
        passOpenFOAMSpaces(in);
        while (!in.eof())
        {
            char c = in.peek();
            if (c == '}')
            {
                in.get(); // omit the '}'
                break;
            }
            std::string lineBuf;
            if (getUntil(in, ';', lineBuf))
                DNDS_assert(false);
            in.get(); // omit the ';'
            readOneItemLine(lineBuf);
            passOpenFOAMSpaces(in);
        }
    }

    inline std::vector<std::string> readOpenFOAMHeader(std::istream &in)
    {
        std::string buf;
        if (getUntil(in, '{', buf))
            DNDS_assert(false);
        std::stringstream bufSS(buf);
        std::string name;
        bufSS >> name;
        DNDS_assert(name == "FoamFile");
        std::vector<std::string> ret;
        readOpenFOAMObj(in, [&](const std::string &lineBuf)
                        { ret.push_back(lineBuf); });
        return ret;
    }

    struct OpenFOAMReader
    {
        std::vector<tPoint> points;
        std::vector<std::vector<int>> faces;
        std::vector<index> owner;
        std::vector<index> neighbour;
        std::map<std::string, OpenFOAMBoundaryCondition> boundaryConditions;

        void ReadPoints(std::istream &IFS)
        {
            DNDS_assert(IFS);
            auto header = readOpenFOAMHeader(IFS);
            // for (auto &i : header)
            //     std::cout << i << std::endl;

            passOpenFOAMSpaces(IFS);
            index nPoints;
            IFS >> nPoints;
            points.resize(nPoints);
            index iPoint = 0;
            readOpenFOAMList(IFS, [&](std::istream &in)
                             {
                    std::vector<real> readReals;
                    readReals.reserve(4);
                    readOpenFOAMList(in, [&](std::istream &iin)
                        {
                            real v{0};
                            iin >> v;
                            readReals.push_back(v);
                    });
                    DNDS_assert(readReals.size() == 3);
                    points.at(iPoint).x() = readReals.at(0);
                    points.at(iPoint).y() = readReals.at(1);
                    points.at(iPoint).z() = readReals.at(2);
                    iPoint++; });
            // for (auto &i : points)
            //     std::cout << i.transpose() << std::endl;
        }

        void ReadFaces(std::istream &IFS)
        {
            DNDS_assert(IFS);
            auto header = readOpenFOAMHeader(IFS);
            // for (auto &i : header)
            //     std::cout << i << std::endl;

            passOpenFOAMSpaces(IFS);
            index nFaces;
            IFS >> nFaces;
            faces.resize(nFaces);
            index iFace = 0;
            readOpenFOAMList(IFS, [&](std::istream &in)
                             {
                    int faceSiz{0};
                    in >> faceSiz;
                    faces.at(iFace).reserve(faceSiz);
                    readOpenFOAMList(in, [&](std::istream &iin)
                        {
                            index inode;
                            iin >> inode;
                            faces.at(iFace).push_back(inode);
                    });
                    iFace++; });
            // for (auto &i : faces)
            // {
            //     for(auto in : i)
            //         std::cout << in << " ";
            //     std::cout << std::endl;
            // }
        }

        void ReadOwner(std::istream &IFS)
        {
            DNDS_assert(IFS);
            auto header = readOpenFOAMHeader(IFS);
            // for (auto &i : header)
            //     std::cout << i << std::endl;

            passOpenFOAMSpaces(IFS);
            index nFaces;
            IFS >> nFaces;
            owner.resize(nFaces);
            index iFace = 0;
            readOpenFOAMList(IFS, [&](std::istream &in)
                             {
                    index iCell;
                    in >> iCell;
                    owner.at(iFace) = iCell;
                    iFace++; });

            // for (auto i : owner)
            //     std::cout << i << std::endl;
        }

        void ReadNeighbour(std::istream &IFS)
        {
            DNDS_assert(IFS);
            auto header = readOpenFOAMHeader(IFS);
            // for (auto &i : header)
            //     std::cout << i << std::endl;

            passOpenFOAMSpaces(IFS);
            index nFaces; // smaller than faces.size()
            IFS >> nFaces;
            neighbour.resize(nFaces); // only internal
            index iFace = 0;
            readOpenFOAMList(IFS, [&](std::istream &in)
                             {
                    index iCell;
                    in >> iCell;
                    neighbour.at(iFace) = iCell;
                    iFace++; });

            // for (auto i : neighbour)
            //     std::cout << i << std::endl;
        }

        void ReadBoundary(std::istream &IFS)
        {
            DNDS_assert(IFS);
            auto header = readOpenFOAMHeader(IFS);
            // for (auto &i : header)
            //     std::cout << i << std::endl;

            passOpenFOAMSpaces(IFS);
            index nBoundary;
            IFS >> nBoundary;

            readOpenFOAMList(
                IFS,
                [&](std::istream &in)
                {
                    std::string buf;
                    if (getUntil(in, '{', buf))
                        DNDS_assert(false);
                    std::stringstream bufSS(buf);
                    std::string name;
                    bufSS >> name;
                    readOpenFOAMObj(
                        in,
                        [&](const std::string &lineBuf)
                        {
                            std::stringstream lineBufSS(lineBuf);
                            std::string key;
                            lineBufSS >> key;
                            if (key == "type")
                            {
                                std::string type;
                                lineBufSS >> type;
                                boundaryConditions[name].type = type;
                            }
                            if (key == "nFaces")
                            {
                                index nFaces;
                                lineBufSS >> nFaces;
                                boundaryConditions[name].nFaces = nFaces;
                            }
                            if (key == "startFace")
                            {
                                index startFace;
                                lineBufSS >> startFace;
                                boundaryConditions[name].startFace = startFace;
                            }
                        });
                });

            for (auto &bc : boundaryConditions)
            {
                std::cout << bc.first << " " << bc.second.type << " " << bc.second.nFaces << " " << bc.second.startFace << std::endl;
            }
        }
    };

    inline std::tuple<Elem::ElemType, std::vector<index>>
    ExtractTopologicalElementFromPolyMeshCell(const std::vector<std::vector<index>> &facesIn,
                                              const std::vector<int> &ownIn)
    {
        std::vector<std::vector<index>> faces = facesIn;
        std::vector<int> own = ownIn;
        // std::sort(faces.begin(), faces.end(), [](auto &&l, auto &&r)
        //           { return l.size() < r.size(); });
        // for (auto &v : faces)
        //     std::sort(v.begin(), v.end());
        std::vector<index> nodes;
        Elem::ElemType etype = Elem::ElemType::UnknownElem;
        if (faces.size() == 6)
        {
            for (auto &v : faces)
            {
                if (v.size() != 4)
                    return std::make_tuple(etype, nodes);
            }
            if (!own[0])
                for (auto v : faces[0])
                    nodes.push_back(v);
            else
                for (size_t i = faces[0].size() - 1; i >= 0; i--)
                    nodes.push_back(faces[0][i]);
            for(int iFoot = 0; iFoot < 4; iFoot++)
            {
                index foot = nodes.at(iFoot);
                index head = -1;
                for(int iFace = 1; iFace < faces.size(); iFace++)
                {
                    int iFootC = -1;
                    index headL = -1;
                    index headR = -1;
                    for(int iF2N = 0; iF2N < faces[iFace].size(); iF2N ++)
                        if(faces[iFace][iF2N] == foot)
                            iFootC = iF2N;
                    if(iFootC >= 0)
                    {
                        headL = faces[iFace][(iFootC - 1) % faces[iFace].size()];
                        headR = faces[iFace][(iFootC + 1) % faces[iFace].size()];
                        auto itFoundL = std::find(nodes.begin(), nodes.begin() + 4, headL);
                        if (itFoundL == nodes.begin() + 4)
                            head = headR;
                        else
                            head = headL;
                    }
                }
                if(!(head >= 0))
                    return std::make_tuple(etype, nodes);
                nodes.push_back(head);
            }
            etype = Elem::ElemType::Hex8;
            return std::make_tuple(etype, nodes);
        }
        else
        {
            DNDS_assert_info(false, fmt::format("faces size {} not supported", faces.size()));
        }
    }

}