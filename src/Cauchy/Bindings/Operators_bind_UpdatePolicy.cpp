#include "Cauchy/Bindings/Operators_bind.hpp"
#include "Cauchy/Implicit/NonlinearSolve.hpp"
#include "DNDS/MPI/Failure.hpp"

#include <thread>

namespace DNDS::Cauchy::Bindings
{
    namespace py = pybind11;

    namespace
    {
        // Simulation owns the sessions and interface publications. Only the
        // MPI controller thread may use this handle, during this policy call.
        class PythonGroup
        {
            using Handle = std::reference_wrapper<NonlinearSolveGroup>;
            std::weak_ptr<Handle> _group;
            std::thread::id _thread = std::this_thread::get_id();

        public:
            explicit PythonGroup(const std::shared_ptr<Handle> &group) : _group(group) {}
            NonlinearSolveGroup &Get() const
            {
                DNDS_check_throw_info(std::this_thread::get_id() == _thread,
                                      "nonlinear group belongs to the MPI controller thread");
                auto group = _group.lock();
                DNDS_check_throw_info(group, "Python nonlinear group has expired");
                return group->get();
            }
            NonlinearSession &Domain(size_t i) const { return Get().domains.at(i).get(); }
        };

        class PythonUpdatePolicy final : public UpdatePolicy
        {
            UpdatePolicyDescriptor _descriptor;

        public:
            explicit PythonUpdatePolicy(UpdatePolicyDescriptor descriptor) : _descriptor(std::move(descriptor)) {}
            const UpdatePolicyDescriptor &Descriptor() const override { return _descriptor; }
            void Solve(NonlinearSolveGroup &group) const override
            {
                py::gil_scoped_acquire acquire;
                try
                {
                    auto lifetime = std::make_shared<std::reference_wrapper<NonlinearSolveGroup>>(group);
                    auto policies = py::module_::import("DNDSR.Cauchy.implicit");
                    policies.attr(_descriptor.coupledStages == 1 ? "_solve_single_stage_group"
                                                                 : "_solve_ditr_domain_group")(PythonGroup(lifetime));
                }
                catch (...)
                {
                    MPI::AbortCaught(group.communication ? group.communication->Communicator() : MPI_COMM_NULL,
                                     "registered Python nonlinear policy failed");
                }
            }
        };
    }

    void BindUpdatePolicies(py::module_ &m)
    {
        using Released = py::call_guard<py::gil_scoped_release>;
        py::class_<PythonGroup>(m, "_NonlinearDomainGroup")
            .def_property_readonly("local_domains",
                                   [](const PythonGroup &g)
        {
            return g.Get().domains.size();
        })
            .def("begin_iteration",
                 [](const PythonGroup &g)
        {
            return g.Get().BeginIteration();
        }, Released{})
            .def("refresh_interfaces",
                 [](const PythonGroup &g, int stage, index iteration)
        {
            g.Get().RefreshInterfaces(stage, iteration);
        }, Released{})
            .def("finish_interfaces",
                 [](const PythonGroup &g)
        {
            g.Get().FinishInterfaces();
        }, Released{})
            .def("observe",
                 [](const PythonGroup &g, size_t domain)
        {
            g.Domain(domain).Observe();
        }, Released{})
            .def("local_terminal",
                 [](const PythonGroup &g, size_t domain)
        {
            return g.Domain(domain).LocalTerminal();
        })
            .def("advance_stage",
                 [](const PythonGroup &g, size_t domain, int stage)
        {
            g.Domain(domain).AdvanceStage(stage);
        }, Released{})
            .def("finish_iteration", [](const PythonGroup &g)
        {
            return g.Get().FinishIteration();
        }, Released{});

        using Observation = NonlinearDetail::ResidualObservation;
        for (auto descriptor :
             {UpdatePolicyDescriptor{"pythonSingleStageImplicit", 1, Observation::BeforeUpdate, false},
              UpdatePolicyDescriptor{"pythonDITRSequential2Stage", 2, Observation::AfterUpdate, true}})
            UpdatePolicyRegistry::Register(descriptor, [descriptor]
            {
                return std::make_unique<PythonUpdatePolicy>(descriptor);
            });
    }
}
