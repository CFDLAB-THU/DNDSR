#include "Geom/DiffTensors.hpp"

namespace DNDS::CFV
{
    void testDiffTensors()
    {
        using namespace Geom::Base;
        Eigen::MatrixXd DxDxi, DxiDx, DxDxiR;
        DxDxi.setZero(20, 3);
        DxiDx.setZero(20, 3);
        DxDxiR.setZero(20, 3);
        DxDxi({1, 2, 3}, {0, 1, 2}) << 1, 0, 1,
            0, 1, 0,
            0, 0, 1;
        DxDxi({4, 5, 6, 7, 8, 9}, {0, 1, 2}) << 2, 2, 0,
            2, -2, 2,
            2, 0, 2,
            1, 0, 0,
            0, -1, 0,
            0, 0, 0;
        std::cout << "DxDxi\n"
                  << DxDxi << std::endl;
        DxDxi2DxiDx<3, 3>(DxDxi, DxiDx);
        std::cout << "DxiDx\n"
                  << DxiDx << std::endl;
        DxDxi2DxiDx<3, 3>(DxiDx, DxDxiR);
        std::cout << "R err " << (DxDxiR - DxDxi).norm() << std::endl;

        DxDxi({1, 2, 3}, {0, 1, 2}).setRandom();
        DxDxi({4, 5, 6, 7, 8, 9}, {0, 1, 2}).setRandom();
        DxDxi({10, 12}, {0, 1, 2}).setRandom();
        DxDxi(Eigen::seq(1, Eigen::last), Eigen::all).setRandom();
        DxDxi2DxiDx<3, 3>(DxDxi, DxiDx);
        DxDxi2DxiDx<3, 3>(DxiDx, DxDxiR);
        std::cout << "R err rand " << (DxDxiR - DxDxi).norm() / DxDxi.norm() << std::endl;
    }
}

int main(int argc, char *argv[])
{
    DNDS::CFV::testDiffTensors();
    return 0;
}