#include "ArrayDOF.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include <vector>

namespace DNDS
{
    class ArrayDofSinglePack
    {
        std::vector<ArrayDof<1, 1>> arrs;

    public:
        // template <class TTrans>
        // void TransBorrowGGIndexing(TTrans &&r_trans)
        // {
        //     for (auto &arr : arrs)
        //     {
        //         arr.trans.BorrowGGIndexing(r_trans);
        //     }
        // }
        template <class TTrans>
        void BuildResizeFatherSon(index s_father, index s_son, const MPIInfo &mpi, TTrans &&r_trans)
        {
            for (auto &arr : arrs)
            {
                if (&arr == arrs.data())
                {
                    DNDS_MAKE_SSP(arr.father, mpi);
                    DNDS_MAKE_SSP(arr.son, mpi);
                    arr.father->Resize(s_father, 1, 1);
                    arr.son->Resize(s_son, 1, 1);
                    arr.TransAttach();
                    arr.trans.BorrowGGIndexing(r_trans);
                    arr.trans.createMPITypes();
                }
            }
        }


    };
}