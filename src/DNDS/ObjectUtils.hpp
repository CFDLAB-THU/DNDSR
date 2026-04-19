#pragma once
/// @file ObjectUtils.hpp
/// @brief Tiny reflection-style helpers (@ref DNDS::MemberRef "MemberRef", @ref DNDS::MemberPtr "MemberPtr") and
/// `for_each_member_*` visitors used by config / serializer code to iterate
/// struct members with their names.

#include "Defines.hpp"
#include <tuple>
// #include <boost/preprocessor.hpp>

namespace DNDS
{
    /// @brief Invoke `f(member)` for every element of a `std::tuple` of members.
    /// @details Companion to @ref DNDS::MemberRef "MemberRef" / @ref DNDS::MemberPtr "MemberPtr" below; used to build
    /// auto-generated JSON / diagnostic dumps of struct contents.
    template <typename TList, typename F>
    void for_each_member_list(TList &&obj_member_list, F &&f)
    {
        std::apply([&](auto &...members)
                   { (f(members), ...); }, obj_member_list);
    }

    /// @brief Simple `{reference, name}` bundle for a struct member.
    /// @details Produced by @ref DNDS_MAKE_1_MEMBER_REF from a variable name;
    /// consumed by `for_each_member_list`.
    template <typename T>
    struct MemberRef
    {
        T &ref;           ///< Reference to the member.
        const char *name; ///< Compile-time-known member name.
    };

    template <typename T>
    MemberRef(T &, const char *) -> MemberRef<T>;

/// @brief Construct a @ref DNDS::MemberRef "MemberRef" capturing `x` and its stringified name.
#define DNDS_MAKE_1_MEMBER_REF(x) \
    MemberRef { x, #x }

    /// @brief Invoke `f(name, obj.*ptr)` for every member in a list of @ref DNDS::MemberPtr "MemberPtr".
    template <typename Class, typename TList, typename F>
    void for_each_member_ptr_list(Class &obj, TList &&obj_member_ptr_list, F &&f)
    {
        std::apply([&](auto &...member_ptr)
                   { (f(member_ptr.name, obj.*(member_ptr.ptr)), ...); }, obj_member_ptr_list);
    }

    /// @brief Low-level variant that passes each @ref DNDS::MemberPtr "MemberPtr" object through
    /// to `f` directly (for callers that need access to both `ptr` and `name`).
    template <typename TList, typename F>
    void for_each_member_ptr_list_raw(TList &&obj_member_ptr_list, F &&f)
    {
        std::apply([&](auto &...members)
                   { (f(members), ...); }, obj_member_ptr_list);
    }

    /// @brief Pointer-to-member wrapper with a symbolic name; the pointer-based
    /// cousin of @ref DNDS::MemberRef "MemberRef" used when the object is known only at visit time.
    template <typename Class, typename T>
    struct MemberPtr
    {
        using t_member_ptr = T Class::*;
        t_member_ptr ptr;  ///< Pointer-to-member.
        const char *name;  ///< Compile-time-known member name.
    };

    template <typename Class, typename T>
    MemberPtr(T Class::*, const char *) -> MemberPtr<Class, T>;

/// @brief Build a @ref DNDS::MemberPtr "MemberPtr" for `Class::member`.
#define DNDS_MAKE_1_MEMBER_PTR(Class, member) \
    MemberPtr { &Class::member, #member }

/// @brief Like @ref DNDS_MAKE_1_MEMBER_PTR but uses the surrounding `t_self` alias,
/// common in DNDSR member definitions (`using t_self = ...;`).
#define DNDS_MAKE_1_MEMBER_PTR_SELF(member) \
    MemberPtr { &t_self::member, #member }
}