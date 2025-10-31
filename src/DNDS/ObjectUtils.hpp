#pragma once

#include "Defines.hpp"
#include <tuple>
// #include <boost/preprocessor.hpp>

namespace DNDS
{
    template <typename TList, typename F>
    void for_each_member_list(TList &&obj_member_list, F &&f)
    {
        std::apply([&](auto &...members)
                   { (f(members), ...); }, obj_member_list);
    }

    template <typename T>
    struct MemberRef
    {
        T &ref;
        const char *name;
    };

    template <typename T>
    MemberRef(T &, const char *) -> MemberRef<T>;

#define DNDS_MAKE_1_MEMBER_REF(x) \
    MemberRef { x, #x }
}