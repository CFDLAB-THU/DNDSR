#include <iostream>
#include <cstdlib>

#include "cantera/core.h"
#include "cantera/base/Solution.h"

int main()
{
    std::cout << Cantera::version() << std::endl;

    const char *dataDir = nullptr;
#ifdef DNDS_CANTERA_DATA_DIR
    dataDir = DNDS_CANTERA_DATA_DIR;
#else
    dataDir = std::getenv("CANTERA_DATA");
#endif
    if (!dataDir)
    {
        std::cerr << "CANTERA_DATA not set and DNDS_CANTERA_DATA_DIR not compiled in" << std::endl;
        return 1;
    }

    std::string mechPath = std::string(dataDir) + "/gri30.yaml";
    auto sol = Cantera::newSolution(mechPath, "", "none");
    auto &gas = *sol->thermo();
    gas.setState_TPX(300, 101325, "CH4:1,O2:2,N2:7.52");

    std::cout << "T=" << gas.temperature() << " K"
              << " cp=" << gas.cp_mass() << " J/kg/K"
              << " nSpecies=" << gas.nSpecies()
              << std::endl;

    return 0;
}
