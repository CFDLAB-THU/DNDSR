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

    template <typename Class, typename TList, typename F>
    void for_each_member_ptr_list(Class &obj, TList &&obj_member_ptr_list, F &&f)
    {
        std::apply([&](auto &...member_ptr)
                   { (f(member_ptr.name, obj.*(member_ptr.ptr)), ...); }, obj_member_ptr_list);
    }

    template <typename TList, typename F>
    void for_each_member_ptr_list_raw(TList &&obj_member_ptr_list, F &&f)
    {
        std::apply([&](auto &...members)
                   { (f(members), ...); }, obj_member_ptr_list);
    }

    template <typename Class, typename T>
    struct MemberPtr
    {
        using t_member_ptr = T Class::*;
        t_member_ptr ptr;
        const char *name;
    };

    template <typename Class, typename T>
    MemberPtr(T Class::*, const char *) -> MemberPtr<Class, T>;

#define DNDS_MAKE_1_MEMBER_PTR(Class, member) \
    MemberPtr { &Class::member, #member }

#define DNDS_MAKE_1_MEMBER_PTR_SELF(member) \
    MemberPtr { &t_self::member, #member }
}