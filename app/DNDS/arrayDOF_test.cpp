#include "DNDS/ArrayDOF.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/MPI.hpp"

namespace DNDS::arrayDOF_test
{
    void test(const MPIInfo &mpi, int arr_size, int v_size)
    {
        auto arr = ArrayDof<3, DynamicSize>();
        DNDS_MAKE_SSP(arr.father, mpi);
        DNDS_MAKE_SSP(arr.son, mpi);
        arr.father->Resize(arr_size, 3, v_size);
        arr.son->Resize(0, 3, v_size);

        arr.setConstant(1.0);
        std::cout << arr.norm2() << std::endl;

        arr.to_device(DeviceBackend::CUDA);
        arr.setConstant(2.0);

        std::cout << arr.norm2() << std::endl;
    }
}

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    assert(argc > 2 && "need 2 args: arr_size v_size");

    DNDS::arrayDOF_test::test(mpi,
                              std::stoi(argv[1]),
                              std::stoi(argv[2]));

    DNDS::MPI::Finalize();
    return 0;
}