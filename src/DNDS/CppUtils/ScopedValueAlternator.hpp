#pragma once

namespace DNDS
{
    template <class T>
    class ScopedValueAlternator
    {
        T &ref_;
        T old_tmp_;

    public:
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