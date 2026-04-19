#pragma once
/// @file ScopedValueAlternator.hpp
/// @brief RAII helper that temporarily overrides a variable and restores its
/// original value when the scope exits.

namespace DNDS
{
    /**
     * @brief RAII sentinel: on construction replaces `ref` with a new value
     * and on destruction restores the original value.
     *
     * @details Useful for surgically changing a global / member flag for the
     * duration of a function without adding explicit try/catch cleanup:
     * ```cpp
     * {
     *   ScopedValueAlternator<bool> g(DNDS::Debug::isDebugging, true);
     *   runDiagnostics();
     * } // isDebugging restored here, even on exception
     * ```
     *
     * Move/copy are not declared; an instance must stay in the scope that
     * owns the reference.
     *
     * @tparam T  Any assignable type.
     */
    template <class T>
    class ScopedValueAlternator
    {
        T &ref_;
        T old_tmp_;

    public:
        /// @brief Save the current value of `ref` and overwrite it with `vAssign`.
        template <class TAssign>
        ScopedValueAlternator(T &ref, TAssign &&vAssign) : ref_(ref), old_tmp_(ref_)
        {
            ref_ = vAssign;
        }

        ~ScopedValueAlternator()
        {
            ref_ = old_tmp_;
        }
    };
}