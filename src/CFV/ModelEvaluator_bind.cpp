#include "ModelEvaluator_bind.hpp"

namespace DNDS::CFV
{
    void pybind11_ModelEvaluator_define(py::module_ &m)
    {
        auto ModelEvaluator_ = tPy_ModelEvaluator(m, "ModelEvaluator");
        ModelEvaluator_
            .def(
                py::init(
                    [](ssp<Geom::UnstructuredMesh> mesh,
                       ssp<CFV::VariationalReconstruction<ModelEvaluator::dim>> vfv,
                       py::object settings)
                    {
                        ModelSettings defaultSettings;
                        nlohmann::ordered_json defaultJson;
                        to_json(defaultJson, defaultSettings);
                        nlohmann::json settings_json = settings;
                        defaultJson.merge_patch(settings_json);
                        from_json(defaultJson, defaultSettings);

                        return std::make_shared<ModelEvaluator>(mesh, vfv, defaultSettings);
                    }),
                py::arg("mesh"), py::arg("vfv"), py::arg("settings"));

        ModelEvaluator_
            .def(
                "ParseSettings",
                [](ModelEvaluator &self, py::object settings)
                {
                    ModelSettings defaultSettings;
                    nlohmann::ordered_json defaultJson;
                    to_json(defaultJson, defaultSettings);
                    nlohmann::json settings_json = settings;
                    defaultJson.merge_patch(settings_json);
                    from_json(defaultJson, defaultSettings);
                    self.get_settings() = defaultJson;
                });

        auto EvaluateRHSOptions_ = py::class_<ModelEvaluator::EvaluateRHSOptions>(ModelEvaluator_, "EvaluateRHSOptions");
        EvaluateRHSOptions_
            .def(py::init())
            .def_readwrite("direct2ndRec", &ModelEvaluator::EvaluateRHSOptions::direct2ndRec)
            .def_readwrite("direct2ndRec1stConv", &ModelEvaluator::EvaluateRHSOptions::direct2ndRec1stConv);
        ModelEvaluator_
            .def("EvaluateRHS", &ModelEvaluator::EvaluateRHS,
                 py::arg("rhs"), py::arg("u"), py::arg("uRec"), py::arg("t"), py::arg("options") = ModelEvaluator::EvaluateRHSOptions{})
            .def("get_FBoundary", &ModelEvaluator::get_FBoundary, py::arg("t"));

        ModelEvaluator_
            .def("DoReconstructionIter", &ModelEvaluator::DoReconstructionIter,
                 py::arg("uRec"), py::arg("uRecNew"), py::arg("u"),
                 py::arg("t"),
                 py::arg("putIntoNew") = false, py::arg("recordInc") = false, py::arg("uRecIsZero") = false);
    }
}