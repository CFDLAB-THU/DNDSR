#include "Geom/OpenFOAMMesh.hpp"
#include <fstream>

int main()
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
    return 0;
}