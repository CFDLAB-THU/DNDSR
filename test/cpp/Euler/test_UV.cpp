#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Euler/Chemistry/ChemicalSource.hpp"
#include <cstdio>
#include <vector>
using namespace DNDS::Euler::Chemistry;

static std::string mechFile()
{
    const char *env = std::getenv("DNDS_MECH_PATH");
    return env ? std::string(env) + "/h2o2.yaml" : "h2o2.yaml";
}

TEST_CASE("setState_UV internal energy consistency")
{
    ChemicalSource chem(mechFile(), "", 1, 1);
    int Ns = chem.nSpecies();

    std::vector<double> Y(Ns, 0.0);
    Y[0] = 0.028;
    Y[3] = 0.222;
    Y[9] = 0.75;
    double ysum = 0;
    for (auto y : Y)
        ysum += y;
    for (auto &y : Y)
        y /= ysum;

    ConstSpeciesBufferView Yv{Y.data(), Ns};
    double Rmix = chem.mixtureR(Yv);

    for (double u_target : {1e3, 1e4, 1e5, 861846.0, 2.15e6, 4.3e6})
    {
        double T = chem.temperatureFromUV(u_target, 1.0, Yv);
        fprintf(stderr, "[UV] u=%12.1f T=%.1fK v=%g\n", u_target, T, 1.0);
        if (u_target > 1e5)
            CHECK(T > 300);
    }
    CHECK(chem.temperatureFromUV(4.3e6, 1.0, Yv) > chem.temperatureFromUV(861846, 1.0, Yv));
}

TEST_CASE("setState_UV with evolved composition from ODE step 5")
{
    ChemicalSource chem(mechFile(), "", 1, 1);
    int Ns = chem.nSpecies();

    // Composition from ODE step 5 (slightly drifted, H/H2O clamped)
    std::vector<double> Y(Ns, 0.0);
    Y[0] = 0.028449;
    Y[1] = 1e-30;
    Y[2] = 0;
    Y[3] = 0.22547;
    Y[4] = 0;
    Y[5] = 1e-30;
    Y[6] = 0;
    Y[7] = 0;
    Y[8] = 0;
    Y[9] = 1.0 - (0.028449 + 1e-30 + 0.22547 + 1e-30);
    ConstSpeciesBufferView Yv{Y.data(), Ns};

    double T = chem.temperatureFromUV(861846, 1.0, Yv, 1200);
    fprintf(stderr, "[UV-evolved] T=%.1fK\n", T);
    CHECK(T > 300);
}
