#include "Cauchy/Bindings/Operators_bind.hpp"
#include "Cauchy/Bindings/OperatorsBindInternal.hpp"
#include "Cauchy/State/QueuedEquationOperations.hpp"
#include "Solver/TimeIntegration/DITRCoefficients.hpp"
#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include <pybind11/stl.h>

namespace DNDS::Cauchy::Bindings
{
    py::object ShareOptional(const std::optional<R::Array<real>> &array)
    {
        return array ? py::cast(R::PythonArray::Share(*array)) : py::none();
    }
    std::map<std::string, Field> PhysicalFields(const EquationDofs &state)
    {
        std::map<std::string, Field> result;
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            result.emplace(state.Fields().Name(iField), state.PhysicalField(iField));
        return result;
    }
    void BindOperatorState(py::module_ &m)
    {
        py::class_<EquationDofs>(
            m, "EquationDofs",
            "Logical flow/RANS/species binding over shared Runtime fields. Copies retain allocations and immutable metadata. Clone copies owned values; refresh ghosts afterward. No implicit regrouping or payload flattening.")
            .def_property_readonly("components", &EquationDofs::ComponentNumber)
            .def_property_readonly("space", &EquationDofs::Space)
            .def_property_readonly(
                "fields", &PhysicalFields,
                "Named physical DOF fields sharing owned/ghost arrays. Physical names and logical block offsets are separate.")
            .def_property_readonly("field_names",
                                   [](const EquationDofs &state)
        {
            std::vector<std::string> result;
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                result.push_back(state.Fields().Name(iField));
            return result;
        }, "Physical communication order; do not substitute sorted dictionary order.")
            .def_property_readonly("blocks",
                                   [](const EquationDofs &state)
        {
            std::map<std::string, std::vector<std::string>> result;
            for (const auto &block : state.Schema().blocks)
                for (const auto &component : block.components)
                    result[block.name].push_back(component.name);
            return result;
        })
            .def("locate_block",
                 [](const EquationDofs &state, const std::string &name)
        {
            const auto location = state.LocateBlock(name);
            return std::make_tuple(state.Fields().Name(location.physicalField), location.firstComponent,
                                   location.componentNumber);
        }, py::arg("name"), "Physical field name, component offset and width. Metadata grants no access permission.")
            .def(
                "resolve_components",
                [](const EquationDofs &state, const std::vector<std::string> &names)
        {
            return state.Schema().ResolveComponents(names);
        }, py::arg("names"),
                "Resolve unique logical component names or block.component names before numerical execution; empty selects all.")
            .def("clone", &EquationDofs::Clone, py::call_guard<py::gil_scoped_release>(), Local)
            .def(
                "zeros_like", &EquationDofOperations::NewVector, py::call_guard<py::gil_scoped_release>(),
                "Allocate independent zeroed storage with the same logical binding/layout. Zero residual scratch need not be an admissible physical state.")
            .def_static(
                "from_fields",
                [](const EquationDofs &prototype, const std::vector<std::pair<std::string, Field>> &fields,
                   const std::vector<std::tuple<std::string, std::string, index>> &bindings)
        {
            Algebra::DofBundle bundle(prototype.Space());
            for (const auto &[name, field] : fields)
                bundle.Add(name, field);
            std::vector<EquationDofBinding> mapping;
            for (const auto &[block, field, first] : bindings)
                mapping.push_back({block, field, first});
            return EquationDofs(prototype.Schema(), std::move(bundle), std::move(mapping));
        },
                py::arg("prototype"), py::arg("fields"), py::arg("bindings"),
                "Share new physical fields with the prototype schema/row space. Ordered (name,field) and (block,field,offset) lists must cover all components without gaps/overlap. No value copy or new numerical capability.")
            .def("copy_from", py::overload_cast<EquationDofs &, const EquationDofs &>(&EquationDofOperations::Copy),
                 py::arg("source"), py::call_guard<py::gil_scoped_release>(), Local)
            .def("fill", py::overload_cast<EquationDofs &, real>(&EquationDofOperations::Fill), py::arg("value"),
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("scale", py::overload_cast<EquationDofs &, real>(&EquationDofOperations::Scale), py::arg("value"),
                 py::call_guard<py::gil_scoped_release>(), Local)
            .def("axpy", py::overload_cast<EquationDofs &, real, const EquationDofs &>(&EquationDofOperations::Axpy),
                 py::arg("alpha"), py::arg("source"), py::call_guard<py::gil_scoped_release>(), Local)
            .def("linear_combination",
                 py::overload_cast<EquationDofs &, real, const EquationDofs &, real, const EquationDofs &>(
                     &EquationDofOperations::LinearCombination),
                 py::arg("alpha"), py::arg("x"), py::arg("beta"), py::arg("y"),
                 py::call_guard<py::gil_scoped_release>(), Local);
        using Ops = QueuedEquationOperations;
        py::class_<Ops>(
            m, "EquationOperations",
            "Owned equation algebra on one Runtime stream. Record a stream event before cross-stream use or host observation.")
            .def(py::init<R::ExecutionStream>(), py::arg("stream"))
            .def("axpy", &Ops::Axpy, py::arg("out"), py::arg("alpha"), py::arg("x"),
                 py::call_guard<py::gil_scoped_release>())
            .def("mass_axpy", &Ops::PhysicalAxpy, py::arg("out"), py::arg("alpha"), py::arg("state"),
                 py::call_guard<py::gil_scoped_release>(), "out += alpha*M*state. M masks the ACM pressure component.")
            .def("mass_combination", &Ops::PhysicalLinearCombination, py::arg("out"), py::arg("a"), py::arg("x"),
                 py::arg("b"), py::arg("y"), py::call_guard<py::gil_scoped_release>(), "out = a*M*x + b*M*y.")
            .def("mass_state_and_rate", &Ops::PhysicalStateAndRate, py::arg("out"), py::arg("a"), py::arg("state"),
                 py::arg("b"), py::arg("rate"), py::call_guard<py::gil_scoped_release>(),
                 "out = a*M*state + b*rate. Every rate component participates.");
        m.def("_ditr_weights", [](real fraction, real theta, const std::string &variant)
        {
            DNDS_check_throw_info(variant == "U2R1" || variant == "U2R2", "unknown DITR variant");
            ODE::DITRSettings settings{fraction, theta,
                                       variant == "U2R1" ? ODE::DITRVariant::U2R1 : ODE::DITRVariant::U2R2};
            settings.Validate();
            const ODE::DITRCoefficients weights(fraction, settings.variant);
            return std::make_pair(weights.interpolation, weights.integration);
        });
        py::class_<AuxiliaryRequest>(m, "AuxiliaryRequest")
            .def(py::init<>())
            .def_readwrite("cell_turbulent_viscosity", &AuxiliaryRequest::cellTurbulentViscosity)
            .def_readwrite("mean_thermodynamics", &AuxiliaryRequest::meanThermodynamics)
            .def_readwrite("face_spectral_radius", &AuxiliaryRequest::faceSpectralRadius)
            .def_readwrite("face_wave_jump", &AuxiliaryRequest::faceWaveJump)
            .def_readwrite("face_pressure_minimum", &AuxiliaryRequest::facePressureMinimum)
            .def_readwrite("cell_spectral_sum", &AuxiliaryRequest::cellSpectralSum)
            .def_readwrite("cell_wave_jump", &AuxiliaryRequest::cellWaveJump)
            .def_readwrite("cell_pressure_minimum", &AuxiliaryRequest::cellPressureMinimum)
            .def_property("derived_quantities",
                          [](const AuxiliaryRequest &request)
        {
            std::vector<DerivedQuantity> result;
            for (int q = 0; q < DerivedQuantityNumber; ++q)
                if (request.derived.Has(DerivedQuantity(q)))
                    result.push_back(DerivedQuantity(q));
            return result;
        }, [](AuxiliaryRequest &request, const std::vector<DerivedQuantity> &quantities)
        {
            DerivedRequest derived;
            for (const auto quantity : quantities)
            {
                const int q = int(quantity);
                DNDS_check_throw_info(q >= 0 && q < DerivedQuantityNumber, "unknown derived quantity");
                derived.enabled[q] = true;
            }
            request.derived = derived;
        });
        py::class_<AuxiliaryValues>(
            m, "AuxiliaryValues",
            "Optional explicit outputs. Debug arrays share allocations; external writes invalidate publication. No automatic refresh/exchange.")
            .def_property_readonly("cell_turbulent_viscosity",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.cellTurbulentViscosity);
        })
            .def_property_readonly("mean_thermodynamics",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.meanThermodynamics);
        })
            .def_property_readonly("face_spectral_radius",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.faceSpectralRadius);
        })
            .def_property_readonly("face_wave_jump",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.faceWaveJump);
        })
            .def_property_readonly("face_pressure_minimum",
                                   [](const AuxiliaryValues &v)
        {
            return ShareOptional(v.facePressureMinimum);
        })
            .def_readonly("cell_spectral_sum", &AuxiliaryValues::cellSpectralSum)
            .def_readonly("cell_wave_jump", &AuxiliaryValues::cellWaveJump)
            .def_readonly("cell_pressure_minimum", &AuxiliaryValues::cellPressureMinimum)
            .def_readonly("derived_owned", &AuxiliaryValues::derivedOwned)
            .def_readonly("derived_ghost", &AuxiliaryValues::derivedGhost);
        py::class_<MeanGradients>(m, "MeanGradients",
                                  "Bound explicit gradient scratch; external writes invalidate auxiliary provenance.")
            .def_readonly("fields", &MeanGradients::fields);
        py::class_<Workspace>(
            m, "Workspace",
            "Native FV workspace tied to exact input allocations. Serialize use and drain work before release.");
        py::class_<MeanWorkspace>(m, "MeanWorkspace",
                                  "Separate zero-reconstruction physical-step scratch; never the ordinary VR guess.");
    }
}
