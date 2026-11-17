#include "Cauchy/Input/InitialField.hpp"
#include "DNDS/Config/ExprtkWrapper.hpp"

namespace DNDS::Cauchy
{
    struct InitialFieldEvaluator::Data
    {
        struct Box
        {
            Vector3 lower, upper;
            std::vector<real> value;
        };
        struct Plane
        {
            Vector3 normal;
            real offset;
            std::vector<real> value;
        };
        struct Expression
        {
            Physics::StateInput::Form form;
            bool physical;
            std::unique_ptr<DNDS::ExprtkWrapperEvaluator> program;
        };

        PhysicalStateConverter converter;
        index width = 0;
        std::vector<Box> boxes;
        std::vector<Plane> planes;
        std::vector<Expression> expressions;

        Data(const EquationSystem &equation, const Physics::ReferenceScales &scales,
             Physics::Chemistry::ChemicalWorkerPool::Lease worker)
            : converter(equation, scales, std::move(worker))
        {
        }
    };

    InitialFieldEvaluator::InitialFieldEvaluator(const InitialFieldSettings &settings, const EquationSystem &equation,
                                                 const Physics::ReferenceScales &scales,
                                                 Physics::Chemistry::ChemicalWorkerPool::Lease worker)
        : _data(std::make_unique<Data>(equation, scales, std::move(worker)))
    {
        for (const auto &block : equation.Schema().blocks)
            _data->width += index(block.components.size());
        for (const auto &box : settings.boxInitializers)
        {
            Data::Box entry{{box.x0, box.y0, box.z0}, {box.x1, box.y1, box.z1}, {}};
            for (int d = 0; d < 3; ++d)
                DNDS_check_throw_info(std::isfinite(entry.lower[d]) && std::isfinite(entry.upper[d]) && entry.lower[d] < entry.upper[d],
                                      "initial box requires finite increasing bounds in all three coordinates");
            entry.value = _data->converter.Conservative(Physics::StateInput::Parse(box.v));
            _data->boxes.push_back(std::move(entry));
        }
        for (const auto &plane : settings.planeInitializers)
        {
            DNDS_check_throw_info(std::isfinite(plane.a) && std::isfinite(plane.b) && std::isfinite(plane.c) && std::isfinite(plane.h) &&
                                      (plane.a != 0 || plane.b != 0 || plane.c != 0),
                                  "initial plane requires a finite nonzero normal and offset");
            _data->planes.push_back({{plane.a, plane.b, plane.c}, plane.h, _data->converter.Conservative(Physics::StateInput::Parse(plane.v))});
        }
        for (const auto &expression : settings.exprtkInitializers)
        {
            // Parse the representation once; this zero vector is only a type
            // descriptor, never a physical state or missing-value fallback.
            const auto descriptor = Physics::StateInput::Parse({{"type", expression.stateType}, {"state", std::vector<real>(_data->width, 0)}});
            auto program = std::make_unique<DNDS::ExprtkWrapperEvaluator>();
            program->AddScalar("inRegion");
            program->AddScalar("iCell");
            program->AddScalar("t");
            program->AddVector("x", 3);
            program->AddVector("UExprtk", int(_data->width));
            std::string source;
            for (const auto &line : expression.exprs)
                source += line + "\n";
            DNDS_check_throw_info(!source.empty(), "empty initial expression program");
            program->Compile(source);
            _data->expressions.push_back({descriptor.form, descriptor.physical, std::move(program)});
        }
    }
    InitialFieldEvaluator::~InitialFieldEvaluator() = default;
    std::vector<real> InitialFieldEvaluator::Evaluate(const Vector3 &point, real time, index iCell, const std::vector<real> &background)
    {
        DNDS_check_throw_info(index(background.size()) == _data->width, "initial field background differs from equation width");
        auto values = background;
        for (const auto &box : _data->boxes)
            if (point[0] > box.lower[0] && point[0] < box.upper[0] && point[1] > box.lower[1] && point[1] < box.upper[1] && point[2] > box.lower[2] && point[2] < box.upper[2])
                values = box.value;
        for (const auto &plane : _data->planes)
            if (plane.normal[0] * point[0] + plane.normal[1] * point[1] + plane.normal[2] * point[2] + plane.offset > 0)
                values = plane.value;
        for (const auto &expression : _data->expressions)
        {
            auto input = _data->converter.Represent(values, expression.form, expression.physical);
            auto &program = *expression.program;
            program.Var("inRegion") = 0;
            program.Var("iCell") = real(iCell);
            program.Var("t") = time;
            for (int d = 0; d < 3; ++d)
                program.VarVec("x", d) = point[d];
            for (index j = 0; j < _data->width; ++j)
                program.VarVec("UExprtk", int(j)) = input.values[j];
            DNDS_check_throw_info(program.Evaluate() == 0, "initial expression must return zero");
            DNDS_check_throw_info(std::isfinite(program.Var("inRegion")), "initial expression region flag is not finite");
            if (program.Var("inRegion") != 0)
            {
                for (index j = 0; j < _data->width; ++j)
                    input.values[j] = program.VarVec("UExprtk", int(j));
                values = _data->converter.Conservative(std::move(input));
            }
        }
        return values;
    }
}
