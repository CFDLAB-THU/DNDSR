#include "../EulerSolver_PrintData.hxx"

#define DNDS_EULERSOLVER_PRINTDATA_INS(model)                         \
    namespace DNDS::Euler                                             \
    {                                                                 \
        template void EulerSolver<model>::PrintData(                  \
            const std::string &fname, const std::string &fnameSeries, \
            const tCellScalarFGet &odeResidualF,                      \
            tAdditionalCellScalarList &additionalCellScalars,         \
            TEval &eval, real tSimu,                                  \
            PrintDataMode mode);                                      \
    }
