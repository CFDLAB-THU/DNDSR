#pragma once

#include "Geom/Access/MeshView.hpp"

namespace DNDS::CFV
{
    /** @brief Canonical face-parent map: parent zero to parent one.
     * @details Values 1..3 select a forward generator and 4..6 its inverse.
     * Seven is reserved for a future composed-map registry and must never be
     * silently interpreted as one generator. This is an ordinal, not a bitmask.
     * An own-side zero Pbi does not imply identity of the relative face map.
     */
    enum class FacePeriodicCode : uint8_t
    {
        Identity = 0,
        Forward1 = 1,
        Forward2 = 2,
        Forward3 = 3,
        Inverse1 = 4,
        Inverse2 = 5,
        Inverse3 = 6,
        Composed = 7
    };

    /// Small immutable backend metadata shared by all face consumers. Exact
    /// generator identity flags bypass vector rotations but retain translations.
    struct FacePeriodicParameters
    {
        Geom::Periodicity generators;
        uint8_t rotatedGenerators = 0;
    };

    /** @brief Borrowed whole-array periodic publication; no per-face matrices.
     * @details The metric publication owns codes and one generator record.
     * Wrappers retain it through asynchronous execution and check residency.
     * Coordinates include affine translations and rotation centers. Vector and
     * equation transforms use the same direction, with inverse equal to transpose.
     */
    struct FacePeriodicView
    {
        Runtime::ArrayView<const uint8_t, false, Runtime::FixedRows<1>> codes{};
        Runtime::ArrayView<const FacePeriodicParameters, false, Runtime::FixedRows<1>> parameters{};
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, codes, parameters);
        }
        DNDS_DEVICE_CALLABLE FacePeriodicCode Code(index iFace) const
        {
            return codes.RowCount() ? FacePeriodicCode(codes(iFace, 0)) : FacePeriodicCode::Identity;
        }
        DNDS_DEVICE_CALLABLE static FacePeriodicCode Inverse(FacePeriodicCode code)
        {
            const auto value = uint8_t(code);
            DNDS_HD_assert(value < uint8_t(FacePeriodicCode::Composed));
            return FacePeriodicCode(value == 0 ? 0 : value <= 3 ? value + 3
                                                                : value - 3);
        }
        DNDS_DEVICE_CALLABLE static int Generator(FacePeriodicCode code)
        {
            const auto value = uint8_t(code);
            DNDS_HD_assert(value > 0 && value < uint8_t(FacePeriodicCode::Composed));
            return value <= 3 ? value : value - 3;
        }
        DNDS_DEVICE_CALLABLE bool Rotates(index iFace) const
        {
            const auto code = Code(iFace);
            return code != FacePeriodicCode::Identity &&
                   bool(parameters(0, 0).rotatedGenerators & (uint8_t(1) << (Generator(code) - 1)));
        }
        DNDS_DEVICE_CALLABLE Geom::tGPoint Rotation(index iFace, int fromSide, int toSide) const
        {
            if (fromSide == toSide || !Rotates(iFace))
                return Geom::tGPoint::Identity();
            const auto code = toSide == 1 ? Code(iFace) : Inverse(Code(iFace));
            const auto &rotation = parameters(0, 0).generators.rotation[Generator(code)].map();
            if (uint8_t(code) <= 3)
                return rotation;
            return rotation.transpose();
        }
        DNDS_DEVICE_CALLABLE Geom::tPoint Coordinate(index iFace, int fromSide, int toSide, const Geom::tPoint &point) const
        {
            if (fromSide == toSide || Code(iFace) == FacePeriodicCode::Identity)
                return point;
            const auto code = toSide == 1 ? Code(iFace) : Inverse(Code(iFace));
            const int zone = -Generator(code);
            const auto &periodic = parameters(0, 0).generators;
            if (uint8_t(code) <= 3)
                return periodic.TransCoord(point, zone);
            return periodic.TransCoordBack(point, zone);
        }
        DNDS_DEVICE_CALLABLE Geom::tGPoint OtherRotation(index iFace, int side) const
        {
            return Rotation(iFace, 1 - side, side);
        }
        DNDS_DEVICE_CALLABLE Geom::tPoint FacePoint(index iFace, int side, const Geom::tPoint &point) const
        {
            return Coordinate(iFace, 0, side, point);
        }
        DNDS_DEVICE_CALLABLE Geom::tPoint OtherPoint(index iFace, int side, const Geom::tPoint &point) const
        {
            return Coordinate(iFace, 1 - side, side, point);
        }
    };
    static_assert(std::is_trivially_copyable_v<FacePeriodicParameters>);
    static_assert(std::is_trivially_copyable_v<FacePeriodicView>);
}
