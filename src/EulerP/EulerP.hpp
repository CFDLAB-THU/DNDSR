#pragma once
#include "DNDS/ArrayDOF.hpp"
#include "DNDS/Defines.hpp"
#include "Eigen/Core"

namespace DNDS::EulerP
{
    static constexpr int nVarsFlow = 5;
    static constexpr int I4 = 4;
    using TU = Eigen::Vector<real, nVarsFlow>;
    using TDiffU = Eigen::Matrix<real, 3, nVarsFlow>;

    using TUFull = Eigen::Vector<real, Eigen::Dynamic>;
    using TDiffUFull = Eigen::Matrix<real, 3, Eigen::Dynamic>;

    using TUFullMap = Eigen::Map<TUFull>;
    using TDiffUFullMap = Eigen::Map<TDiffUFull>;

    using TUDof = ArrayDof<nVarsFlow, 1>;
    using TUGrad = ArrayDof<3, nVarsFlow>;

    using TUScalar = ArrayDof<1, 1>;
    using TUScalarGrad = ArrayDof<3, 1>;

    using TUScalar2 = ArrayDof<2, 1>;

    using TUVec = ArrayDof<3, 1>;
    using TUVecGrad = ArrayDof<3, 3>;

    template <class TU>
    DNDS_DEVICE_CALLABLE DNDS_FORCEINLINE auto U123(TU &&v)
    {
        return v.template block<3, 1>(1, 0);
    }

    template <class TU>
    DNDS_DEVICE_CALLABLE DNDS_FORCEINLINE auto U012(TU &&v)
    {
        return v.template block<3, 1>(0, 0);
    }

    template <class TDerived>
    class EvaluatorArgBase
    {
    public:
        void WaitAllPull(DeviceBackend B)
        {
            auto *dThis = static_cast<TDerived *>(this);

            auto wait_all = [&](std::string name, auto &v)
            {
                auto do_wait = [B, name](auto &vv)
                {
                    DNDS_check_throw_info(vv->father && vv->son, name + " needs father and son");
                    //! this assertion should be provided by ArrayTransformer
                    DNDS_check_throw_info(vv->trans.father.get() == vv->father.get(), name + " needs father == trans.father");
                    DNDS_check_throw_info(vv->trans.son.get() == vv->son.get(), name + " needs son == trans.son");
                    DNDS_check_throw_info(vv->father->device() == B,
                                          name +
                                              " father on " + device_backend_name(vv->father->device()) +
                                              " wait on " + device_backend_name(B));
                    DNDS_check_throw_info(vv->son->device() == B,
                                          name +
                                              " son on " + device_backend_name(vv->father->device()) +
                                              " wait on " + device_backend_name(B));

                    vv->trans.waitPersistentPull(B);
                };
                if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                    do_wait(v);
                else
                    for (size_t i = 0; i < v.size(); i++)
                        do_wait(v[i]);
            };
            for_each_member_ptr_list(*dThis, TDerived::member_list(), wait_all);
        }
    };
}