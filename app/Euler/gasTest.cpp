#include "Euler/Gas.hpp"

int main()
{
    using namespace DNDS::Euler;
    using namespace DNDS;

    auto V1 = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5};
    auto V2 = Eigen::Vector<real, 5>{1, 1, 1, 1, -2.5};
    real e0 = V1(4) - 0.5 * V1({1, 2, 3}).squaredNorm() / V1(0);
    Eigen::Vector<real, 5> Vnew = V1 + V2;
    real e1 = Vnew(4) - 0.5 * Vnew({1, 2, 3}).squaredNorm() / Vnew(0);
    std::cout << e0 << " " << e1 << std::endl;
    for (int i = 0; i < 100; i++)
    {
        real k = i / 100.;
        Vnew = V1 + V2 * k;
        real e1 = Vnew(4) - 0.5 * Vnew({1, 2, 3}).squaredNorm() / Vnew(0);
        std::cout << k << ": " << e1 << std::endl;
    }

    real compress = Gas::IdealGasGetCompressionRatioPressure<3, 0, 5>(
        V1,
        V2,
        1e-10,
        0, 0);
    std::cout << compress << std::endl;
}