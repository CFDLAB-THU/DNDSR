#ifndef __DNDS_REALLY_COMPILING__
#define __DNDS_REALLY_COMPILING__
#define __DNDS_REALLY_COMPILING__HEADER_ON__
#endif
#include "Euler/EulerBC.hpp"
#ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
#undef __DNDS_REALLY_COMPILING__
#endif

namespace DNDS::Euler
{

    void oneDimProfileTest(MPIInfo &mpi)
    {
        OneDimProfile<1> odP(mpi);
        odP.GenerateUniform(10, 1, 0, 10);
        odP.SetZero();
        auto addUniformA = [&](real v, real L, real R)
        {
            odP.AddSimpleInterval(Eigen::Vector<real, 1>{v}, R - L, L, R);
        };
        addUniformA(1, 0, 1);
        addUniformA(1, 0.5, 1.5);
        addUniformA(1, 1.5, 2);
        addUniformA(1, 100, 101);
        addUniformA(1, -1, 101);
        odP.Reduce();

        if (mpi.rank == 0)
        {
            std::cout << "Results" << std::endl;
            for (index i = 0; i < odP.Size(); i++)
                std::cout << odP.Get(i) << std::endl;
            std::cout << odP.v << std::endl;
            std::cout << odP.GetPlain(0.5) << std::endl;
            std::cout << odP.GetPlain(1.1) << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    DNDS::Euler::oneDimProfileTest(mpi);
    MPI_Finalize();
}
