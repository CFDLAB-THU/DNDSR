#include "Geom/OpenFOAMMesh.hpp"
#include <fstream>

void test()
{
    using namespace DNDS::Geom::OpenFOAM;
    // OpenFOAMMesh ofMesh("points");
    OpenFOAMReader ofReader;
    std::ifstream pointsIFS("points");
    ofReader.ReadPoints(pointsIFS);
    std::ifstream facesIFS("faces");
    ofReader.ReadFaces(facesIFS);
    std::ifstream ownerIFS("owner");
    ofReader.ReadOwner(ownerIFS);
    std::ifstream neighbourIFS("neighbour");
    ofReader.ReadNeighbour(neighbourIFS);
    std::ifstream boundaryIFS("boundary");
    ofReader.ReadBoundary(boundaryIFS);
    OpenFOAMConverter ofConverter;
    ofConverter.BuildCell2Face(ofReader);
    ofConverter.BuildCell2Node(ofReader);
}

int main()
{
    try
    {
        test();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "error" << std::endl;
    }
    return 0;
}