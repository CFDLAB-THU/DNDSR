#include "Physics/StateInput.hpp"
#include <algorithm>

namespace DNDS::Physics
{
    StateInput StateInput::Parse(const nlohmann::ordered_json &input)
    {
        StateInput result{Form::Sensible, false, {}};
        std::string type = "consSensible";
        if (input.is_array())
            result.values = input.get<std::vector<real>>();
        else
        {
            DNDS_check_throw_info(input.is_object() && input.size() == 2 && input.contains("type") && input.contains("state"),
                                  "state input requires {type,state} or a consSensible array");
            type = input.at("type").get<std::string>();
            result.values = input.at("state").get<std::vector<real>>();
        }
        if (type.size() > 4 && type.substr(type.size() - 4) == "_phy")
        {
            result.physical = true;
            type.resize(type.size() - 4);
        }
        if (type == "cons")
            result.form = Form::Conservative;
        else if (type == "consSensible")
            result.form = Form::Sensible;
        else if (type == "primRhoP")
            result.form = Form::RhoP;
        else if (type == "primRhoT")
            result.form = Form::RhoT;
        else if (type == "primTP")
            result.form = Form::TP;
        else if (type == "totalPT")
            result.form = Form::TotalPT;
        else if (type == "primUP")
            result.form = Form::VelocityPressure;
        else
            DNDS_check_throw_info(false, "unsupported physical state type: " + type);
        DNDS_check_throw_info(result.values.size() >= (result.form == Form::VelocityPressure ? 4u : 5u) &&
                                  std::all_of(result.values.begin(), result.values.end(), [](real v)
                                              { return std::isfinite(v); }),
                              "physical state requires finite values and a complete flow block");
        return result;
    }
    nlohmann::ordered_json StateInput::Schema()
    {
        using Json = nlohmann::ordered_json;
        const Json values{{"type", "array"}, {"items", {{"type", "number"}}}, {"minItems", 5}};
        const auto tagged = [](Json types, Json state)
        {
            return Json{{"type", "object"}, {"required", {"type", "state"}}, {"additionalProperties", false}, {"properties", {{"type", {{"enum", types}}}, {"state", state}}}};
        };
        auto velocityPressure = values;
        velocityPressure["minItems"] = 4;
        auto object = tagged({"cons", "consSensible", "primRhoP", "primRhoT", "primTP", "totalPT",
                              "cons_phy", "consSensible_phy", "primRhoP_phy", "primRhoT_phy",
                              "primTP_phy", "totalPT_phy", "primUP", "primUP_phy"},
                             velocityPressure);
        // The partial-config validator selects oneOf by JSON type. Use one
        // object branch for tagged vectors. Parse enforces the same width
        // condition that external JSON Schema consumers evaluate below.
        object["if"] = {{"properties", {{"type", {{"enum", {"primUP", "primUP_phy"}}}}}}};
        object["else"] = {{"properties", {{"state", values}}}};
        return {{"oneOf", Json::array({Json{{"type", "null"}}, values, object})}};
    }
    void StateInput::ToCodeUnits(const ReferenceScales &scales, const std::vector<real> &scalarScales)
    {
        scales.Validate();
        const index flowWidth = form == Form::VelocityPressure ? 4 : 5;
        DNDS_check_throw_info(index(values.size()) == flowWidth + index(scalarScales.size()), "state input component count differs from equation");
        if (!physical)
            return;
        const bool conservative = form == Form::Conservative || form == Form::Sensible;
        if (form == Form::VelocityPressure)
        {
            for (int d = 0; d < 3; ++d)
                values[d] /= scales.U0;
            values[3] /= scales.Pressure();
        }
        else if (form == Form::TotalPT)
        {
            values[0] /= scales.Pressure();
            values[1] /= scales.T0;
            // Direction is dimensionless. Its magnitude is not a speed.
        }
        else
        {
            values[0] /= form == Form::TP ? scales.T0 : scales.rho0;
            for (int d = 1; d <= 3; ++d)
                values[d] /= conservative ? scales.Momentum() : scales.U0;
            values[4] /= form == Form::RhoT ? scales.T0 : scales.Pressure();
        }
        for (index j = 0; j < index(scalarScales.size()); ++j)
        {
            DNDS_check_throw_info(std::isfinite(scalarScales[j]) && scalarScales[j] > 0, "invalid scalar input unit scale");
            values[flowWidth + j] /= scalarScales[j] * (conservative ? scales.rho0 : 1);
        }
        physical = false;
    }
    void StateInput::ToPhysicalUnits(const ReferenceScales &scales, const std::vector<real> &scalarScales)
    {
        scales.Validate();
        DNDS_check_throw_info(values.size() == (form == Form::VelocityPressure ? 4 : 5) + scalarScales.size(),
                              "state input component count differs from equation");
        if (physical)
            return;
        // Use the same factors as input conversion, without duplicating the
        // dimensional policy for each form and appended equation component.
        StateInput factors{form, true, std::vector<real>(values.size(), 1)};
        factors.ToCodeUnits(scales, scalarScales);
        for (index j = 0; j < index(values.size()); ++j)
            values[j] /= factors.values[j];
        physical = true;
    }
}
