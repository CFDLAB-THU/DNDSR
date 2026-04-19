#pragma once

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif
#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrix.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "Geom/BaseFunction.hpp"
#include "Limiters.hpp"
#include "DNDS/JsonUtil.hpp"
#include "DNDS/ConfigParam.hpp"
#include "DNDS/ConfigEnum.hpp"
#include "FiniteVolumeSettings.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

namespace DNDS::CFV
{
    /**
     * @brief
     * A means to translate nlohmann json into c++ primitive data types and back;
     * and stores then during computation.
     *
     */
    struct VRSettings : public FiniteVolumeSettings
    {
        using json = nlohmann::ordered_json;
        using t_base = FiniteVolumeSettings;

        int intOrderVR{-1};         /// @brief integration degree for VR matrices, <0 means using intOrder
        int intOrderVRBC{-2};       /// @brief integration degree for VR matrices on BC faces, -1 means using int Order, < -1 means using intOrderVRValue() (the same as intOrderVR)
        bool cacheDiffBase = false; /// @brief if cache the base function values on each of the quadrature points
        uint8_t cacheDiffBaseSize = UINT8_MAX;

        real jacobiRelax = 1.0; /// @brief VR SOR/Jacobi iteration relaxation factor
        bool SORInstead = true; /// @brief use SOR instead of relaxed Jacobi iteration

        real smoothThreshold = 0.01;  /// @brief limiter's smooth indicator threshold
        real WBAP_nStd = 10;          /// @brief n used in WBAP limiters
        bool normWBAP = false;        /// @brief if switch to normWBAP
        int limiterBiwayAlter = 0;    /// @brief 0=wbap-L2-biway, 1=minmod-biway
        int subs2ndOrder = 0;         /// @brief 0: vfv; 1: gauss rule; 2: least square; 11: GGMP;
        int subs2ndOrderGGScheme = 1; /// @brief 1: gauss rule using distance for interpolation; 0: no interpolation
        real svdTolerance = 0;        /// @brief tolerance used in svd

        real bcWeight = 1;

        struct BaseSettings
        {
            bool localOrientation = false;
            bool anisotropicLengths = false;

            DNDS_DEVICE_TRIVIAL_COPY_DEFINE(BaseSettings, BaseSettings)

            DNDS_DECLARE_CONFIG(BaseSettings)
            {
                DNDS_FIELD(localOrientation,   "Use local orientation for basis");
                DNDS_FIELD(anisotropicLengths,  "Use anisotropic length scales");
            }
        } baseSettings;

        struct FunctionalSettings
        {
            enum class ScaleType
            {
                UnknownScale = -1,
                MeanAACBB = 0,
                BaryDiff = 1,
                CellMax = 2,
            } scaleType = ScaleType::BaryDiff;

            real scaleMultiplier = 1.0;

            enum class DirWeightScheme
            {
                UnknownDirWeight = -1,
                Factorial = 0,
                HQM_OPT = 1,
                TEST_OPT = 1000,
                ManualDirWeight = 999,
            } dirWeightScheme = DirWeightScheme::Factorial;

            int dirWeightCombPowV = 1;

            std::array<real, 5> manualDirWeights{{1, 1, 0.5, 1. / 6, 1. / 24}};

            enum class GeomWeightScheme
            {
                UnknownGeomWeight = -1,
                GWNone = 0,
                HQM_SD = 1,
                SD_Power = 2,
            } geomWeightScheme = GeomWeightScheme::GWNone;

            real geomWeightBias = 0;
            real geomWeightPower = 0.5;
            real geomWeightPower1 = 0;
            real geomWeightPower2 = 0;

            bool useAnisotropicFunctional = false;
            real tanWeightScale = 1.;

            enum class AnisotropicType
            {
                UnknownAnisotropic = -1,
                InertiaCoord = 0,
                InertiaCoordBB = 1,
                Norm = 2,
                CentDiff = 3,
                WallDist = 4,
                InertiaCoordBBNorm = 5,
                InertiaCoordBBSym = 6,
            } anisotropicType = AnisotropicType::InertiaCoord;

            real inertiaWeightPower = 1.0;

            real greenGauss1Weight = 0.0;
            real greenGauss1Bias = 0.5;
            real greenGauss1Penalty = 0.0;
            int greenGaussSpacial = 0; // 1 for uniform weight

            DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(FunctionalSettings, FunctionalSettings)

            DNDS_DECLARE_CONFIG(FunctionalSettings)
            {
                DNDS_FIELD(scaleType,                "Functional scale type");
                DNDS_FIELD(scaleMultiplier,          "Functional scale multiplier",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(dirWeightScheme,          "Directional weight scheme");
                DNDS_FIELD(dirWeightCombPowV,        "Directional weight combination power");
                DNDS_FIELD(manualDirWeights,          "Manual directional weights vector");
                DNDS_FIELD(geomWeightScheme,          "Geometric weight scheme");
                DNDS_FIELD(geomWeightPower,           "Geometric weight power");
                DNDS_FIELD(geomWeightPower1,          "Geometric weight power 1");
                DNDS_FIELD(geomWeightPower2,          "Geometric weight power 2");
                DNDS_FIELD(useAnisotropicFunctional,  "Use anisotropic functional");
                DNDS_FIELD(tanWeightScale,            "Tangential weight scale");
                DNDS_FIELD(anisotropicType,           "Anisotropic functional type");
                DNDS_FIELD(inertiaWeightPower,        "Inertia weight power");
                DNDS_FIELD(geomWeightBias,            "Geometric weight bias");
                DNDS_FIELD(greenGauss1Weight,         "Green-Gauss type-1 weight");
                DNDS_FIELD(greenGauss1Bias,           "Green-Gauss type-1 bias");
                DNDS_FIELD(greenGauss1Penalty,        "Green-Gauss type-1 penalty");
                DNDS_FIELD(greenGaussSpacial,         "Green-Gauss spatial mode: 0=default, 1=uniform");
            }
            DNDS_DEVICE_CALLABLE FunctionalSettings() = default;
        } functionalSettings;

        // VRSettings()
        // {
        // }

        DNDS_HOST VRSettings() = default;
        DNDS_HOST VRSettings(int dim) : FiniteVolumeSettings(dim)
        {
            cacheDiffBaseSize = uint8_t(dim + 1);
        }

        DNDS_DECLARE_CONFIG(VRSettings)
        {
            // Base class fields (FiniteVolumeSettings) — flattened into the same JSON object.
            // Cast base-class pointer-to-member to derived type for template deduction.
            config.field(static_cast<int T::*>(&T::maxOrder),                     "maxOrder",
                         "Polynomial degree of reconstruction",
                         DNDS::Config::range(0));
            config.field(static_cast<int T::*>(&T::intOrder),                     "intOrder",
                         "Global integration degree",
                         DNDS::Config::range(0));
            config.field(static_cast<bool T::*>(&T::ignoreMeshGeometryDeficiency), "ignoreMeshGeometryDeficiency",
                         "Ignore mesh geometry deficiency warnings");
            config.field(static_cast<int T::*>(&T::nIterCellSmoothScale),         "nIterCellSmoothScale",
                         "Cell smooth scale iterations",
                         DNDS::Config::range(0));

            // VRSettings own fields
            DNDS_FIELD(intOrderVR,            "VR integration degree (<0 = use intOrder)");
            DNDS_FIELD(intOrderVRBC,          "VR BC integration degree (-1=intOrder, <-1=intOrderVR)");
            DNDS_FIELD(cacheDiffBase,         "Cache base function values at quadrature points");
            DNDS_FIELD(cacheDiffBaseSize,     "Cached diff base size (dim+1)");
            DNDS_FIELD(jacobiRelax,           "VR SOR/Jacobi relaxation factor",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(SORInstead,            "Use SOR instead of relaxed Jacobi");
            DNDS_FIELD(smoothThreshold,       "Smooth indicator threshold",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(WBAP_nStd,             "WBAP limiter n parameter",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(normWBAP,              "Use normWBAP limiter variant");
            DNDS_FIELD(limiterBiwayAlter,     "Limiter biway alter: 0=wbap-L2, 1=minmod");
            DNDS_FIELD(subs2ndOrder,          "2nd order substitution: 0=vfv, 1=gauss, 2=LS, 11=GGMP");
            DNDS_FIELD(subs2ndOrderGGScheme,  "2nd order GG scheme: 0=no interp, 1=distance interp");
            DNDS_FIELD(svdTolerance,          "SVD tolerance for reconstruction",
                       DNDS::Config::range(0.0));
            config.field_section(&T::baseSettings, "baseSettings",
                                 "Basis function settings");
            config.field_section(&T::functionalSettings, "functionalSettings",
                                 "Functional/weight settings");
            DNDS_FIELD(bcWeight,              "Boundary condition weight",
                       DNDS::Config::range(0.0));
        }

        /// @brief Backward-compatible write (used by Python bindings).
        void WriteIntoJson(json &jsonSetting) const
        {
            to_json(jsonSetting, *this);
        }

        /// @brief Backward-compatible read (used by Python bindings).
        void ParseFromJson(const json &jsonSetting)
        {
            from_json(jsonSetting, *this);
        }

        [[nodiscard]] bool intOrderVRIsSame() const { return intOrderVR == intOrder || intOrderVR < 0; }
        [[nodiscard]] int intOrderVRValue() const { return intOrderVR < 0 ? intOrder : intOrderVR; }

        [[nodiscard]] bool intOrderVRBCIsSame() const
        {
            if (intOrderVRBC >= 0)
                return intOrderVRBC == intOrder;
            if (intOrderVRBC == -1)
                return true;
            return intOrderVRIsSame();
        }
        [[nodiscard]] int intOrderVRBCValue() const
        {
            if (intOrderVRBC >= 0)
                return intOrderVRBC;
            if (intOrderVRBC == -1)
                return intOrder;
            return intOrderVRValue();
        }
    };

    DNDS_DEFINE_ENUM_JSON(
        VRSettings::FunctionalSettings::ScaleType,
        {{VRSettings::FunctionalSettings::ScaleType::UnknownScale, nullptr},
         {VRSettings::FunctionalSettings::ScaleType::MeanAACBB, "MeanAACBB"},
         {VRSettings::FunctionalSettings::ScaleType::BaryDiff, "BaryDiff"},
         {VRSettings::FunctionalSettings::ScaleType::CellMax, "CellMax"}})

    DNDS_DEFINE_ENUM_JSON(
        VRSettings::FunctionalSettings::DirWeightScheme,
        {{VRSettings::FunctionalSettings::DirWeightScheme::UnknownDirWeight, nullptr},
         {VRSettings::FunctionalSettings::DirWeightScheme::Factorial, "Factorial"},
         {VRSettings::FunctionalSettings::DirWeightScheme::HQM_OPT, "HQM_OPT"},
         {VRSettings::FunctionalSettings::DirWeightScheme::ManualDirWeight, "ManualDirWeight"},
         {VRSettings::FunctionalSettings::DirWeightScheme::TEST_OPT, "TEST_OPT"}})

    DNDS_DEFINE_ENUM_JSON(
        VRSettings::FunctionalSettings::GeomWeightScheme,
        {{VRSettings::FunctionalSettings::GeomWeightScheme::UnknownGeomWeight, nullptr},
         {VRSettings::FunctionalSettings::GeomWeightScheme::GWNone, "GWNone"},
         {VRSettings::FunctionalSettings::GeomWeightScheme::HQM_SD, "HQM_SD"},
         {VRSettings::FunctionalSettings::GeomWeightScheme::SD_Power, "SD_Power"}})

    DNDS_DEFINE_ENUM_JSON(
        VRSettings::FunctionalSettings::AnisotropicType,
        {
            {VRSettings::FunctionalSettings::AnisotropicType::UnknownAnisotropic, nullptr},
            {VRSettings::FunctionalSettings::AnisotropicType::InertiaCoord, "InertiaCoord"},
            {VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBB, "InertiaCoordBB"},
            {VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBBNorm, "InertiaCoordBBNorm"},
            {VRSettings::FunctionalSettings::AnisotropicType::InertiaCoordBBSym, "InertiaCoordBBSym"},
            {VRSettings::FunctionalSettings::AnisotropicType::Norm, "Norm"},
            {VRSettings::FunctionalSettings::AnisotropicType::CentDiff, "CentDiff"},
            {VRSettings::FunctionalSettings::AnisotropicType::WallDist, "WallDist"},
        })
}
