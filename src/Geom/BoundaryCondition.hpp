#pragma once
#include "Geometric.hpp"
#include "DNDS/MPI.hpp"
#include <map>
#include <functional>
#include <unordered_map>

namespace DNDS::Geom
{
    class BCName_2_ID
    {
        std::map<std::string, t_index> name_2_id;

        void clear() { name_2_id.clear(); }

        t_index operator[](const std::string &name)
        {
            auto found = name_2_id.find(name);
            if (found == name_2_id.end())
                return name_2_id[name] = static_cast<t_index>(name_2_id.size());
            else
                return (*found).second;
        }
    };

    static const t_index BC_ID_INTERNAL = 0;    // do not change
    static const t_index BC_ID_PERIODIC_1 = -1; // * periodic-s do not count as external face
    static const t_index BC_ID_PERIODIC_2 = -2; // but should be identified also
    static const t_index BC_ID_PERIODIC_3 = -3;

    static const t_index BC_ID_PERIODIC_1_DONOR = -4;
    static const t_index BC_ID_PERIODIC_2_DONOR = -5;
    static const t_index BC_ID_PERIODIC_3_DONOR = -6;

    static const t_index BC_ID_NULL = 1;
    static const t_index BC_ID_DEFAULT_WALL = 2;
    static const t_index BC_ID_DEFAULT_FAR = 3;
    static const t_index BC_ID_DEFAULT_WALL_INVIS = 4;

    static const t_index BC_ID_DEFAULT_SPECIAL_DMR_FAR = 11;
    static const t_index BC_ID_DEFAULT_SPECIAL_RT_FAR = 12;
    static const t_index BC_ID_DEFAULT_SPECIAL_IV_FAR = 13;
    static const t_index BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR = 14;
    static const t_index BC_ID_DEFAULT_MAX = 20;

    using t_FBCName_2_ID = std::function<t_index(const std::string &)>;

    static const t_FBCName_2_ID FBC_Name_2_ID_Default = [](const std::string &name) -> t_index
    {
        if (name == "PERIODIC_1")
            return BC_ID_PERIODIC_1;
        if (name == "PERIODIC_2")
            return BC_ID_PERIODIC_2;
        if (name == "PERIODIC_3")
            return BC_ID_PERIODIC_3;
        if (name == "PERIODIC_1_DONOR")
            return BC_ID_PERIODIC_1_DONOR;
        if (name == "PERIODIC_2_DONOR")
            return BC_ID_PERIODIC_2_DONOR;
        if (name == "PERIODIC_3_DONOR")
            return BC_ID_PERIODIC_3_DONOR;
        if (name == "WALL" || name == "bc-4")
            return BC_ID_DEFAULT_WALL;
        if (name == "FAR" || name == "bc-2")
            return BC_ID_DEFAULT_FAR;
        if (name == "WALL_INVIS" || name == "bc-3")
            return BC_ID_DEFAULT_WALL_INVIS;
        if (name == "bc-DMRFar")
            return BC_ID_DEFAULT_SPECIAL_DMR_FAR;
        if (name == "bc-IVFar")
            return BC_ID_DEFAULT_SPECIAL_IV_FAR;
        if (name == "bc-RTFar")
            return BC_ID_DEFAULT_SPECIAL_RT_FAR;
        if (name == "bc-2DRiemannFar")
            return BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR;
        return BC_ID_NULL;
    };

    inline auto GetFaceName2IDDefault()
    {
        std::unordered_map<std::string, t_index> ret = {
            {"PERIODIC_1", BC_ID_PERIODIC_1},
            {"PERIODIC_2", BC_ID_PERIODIC_2},
            {"PERIODIC_3", BC_ID_PERIODIC_3},
            {"PERIODIC_1_DONOR", BC_ID_PERIODIC_1_DONOR},
            {"PERIODIC_2_DONOR", BC_ID_PERIODIC_2_DONOR},
            {"PERIODIC_3_DONOR", BC_ID_PERIODIC_3_DONOR},
            {"WALL", BC_ID_DEFAULT_WALL},
            {"bc-4", BC_ID_DEFAULT_WALL},
            {"FAR", BC_ID_DEFAULT_FAR},
            {"bc-2", BC_ID_DEFAULT_FAR},
            {"WALL_INVIS", BC_ID_DEFAULT_WALL_INVIS},
            {"bc-3", BC_ID_DEFAULT_WALL_INVIS},
            {"bc-DMRFar", BC_ID_DEFAULT_SPECIAL_DMR_FAR},
            {"bc-IVFar", BC_ID_DEFAULT_SPECIAL_IV_FAR},
            {"bc-RTFar", BC_ID_DEFAULT_SPECIAL_RT_FAR},
            {"bc-2DRiemannFar", BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR}};
        return ret;
    }

    inline bool FaceIDIsExternalBC(t_index id)
    {
        return id > 0;
    }

    inline bool FaceIDIsInternal(t_index id)
    {
        return id <= 0;
    }

    inline bool FaceIDIsTrueInternal(t_index id)
    {
        return id == 0;
    }

    inline bool FaceIDIsPeriodic(t_index id)
    {
        return id == BC_ID_PERIODIC_1 ||
               id == BC_ID_PERIODIC_2 ||
               id == BC_ID_PERIODIC_3 ||
               id == BC_ID_PERIODIC_1_DONOR ||
               id == BC_ID_PERIODIC_2_DONOR ||
               id == BC_ID_PERIODIC_3_DONOR;
    }

    inline bool FaceIDIsPeriodicMain(t_index id)
    {
        return id == BC_ID_PERIODIC_1 ||
               id == BC_ID_PERIODIC_2 ||
               id == BC_ID_PERIODIC_3;
    }

    inline bool FaceIDIsPeriodicDonor(t_index id)
    {
        return id == BC_ID_PERIODIC_1_DONOR ||
               id == BC_ID_PERIODIC_2_DONOR ||
               id == BC_ID_PERIODIC_3_DONOR;
    }

} // namespace Geom
