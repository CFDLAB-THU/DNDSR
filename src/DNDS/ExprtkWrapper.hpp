#pragma once
/// @file ExprtkWrapper.hpp
/// @brief Runtime mathematical expression compiler and evaluator wrapping exprtk.

#include <map>
#include "Defines.hpp"

namespace DNDS
{
    /// @brief Scalar name -> value map fed into an #ExprtkWrapperEvaluator.
    using tExprVars = std::map<std::string, real>;
    /// @brief Vector name -> Eigen dense vector map fed into an #ExprtkWrapperEvaluator.
    using tExprVarVecs = std::map<std::string, Eigen::Vector<real, Eigen::Dynamic>>;

    /**
     * @brief Thin RAII wrapper over the exprtk expression library.
     *
     * @details Compiles a user-supplied math expression once (with named
     * scalar and vector variables) and evaluates it many times against
     * variable values set through #Var / #VarVec. Used for user-defined
     * initial conditions and boundary functions read from config JSON
     * (see `docs/guides/array_usage.md` and CFV config files).
     *
     * Typical usage:
     * ```cpp
     * ExprtkWrapperEvaluator e;
     * e.AddScalar("t");
     * e.AddVector("x", 3);
     * e.Compile("sin(x[0]) * cos(t)");
     * e.Var("t") = 0.5;
     * e.VarVec("x", 0) = 0.1;
     * real val = e.Evaluate();
     * ```
     *
     * The type-erased `void*` members hide the exprtk symbol_table / expression
     * / parser types so exprtk headers stay out of the public API.
     */
    class ExprtkWrapperEvaluator
    {
        void *_ptr_st = nullptr;
        void *_ptr_exp = nullptr;
        void *_ptr_parser = nullptr;
        tExprVars _vars;
        tExprVarVecs _varVecs;
        bool _compiled = false;

    public:
        /// @brief Register a scalar variable. `init` is accepted for API
        /// symmetry but currently ignored (scalars default to 0).
        /// @note Calling any `Add*` invalidates a previously compiled expression.
        void AddScalar(const std::string &name, real init = 0)
        {
            Clear();
            _vars[name] = 0;
        }

        /// @brief Register a dense vector variable named `name` of length `size`.
        void AddVector(const std::string &name, int size)
        {
            Clear();
            _varVecs[name].resize(size);
        }

        /// @brief Mutable reference to a scalar variable's current value.
        real &Var(const std::string &name) { return _vars.at(name); }
        /// @brief Mutable reference to element `i` of a vector variable.
        real &VarVec(const std::string &name, int i) { return _varVecs.at(name)(i); }
        /// @brief Length of a registered vector variable.
        index VarVecSize(const std::string &name) { return _varVecs.at(name).size(); }

        /// @brief Whether #Compile has been called and the expression parsed successfully.
        [[nodiscard]] bool Compiled() const
        {
            return _compiled;
        }

        /// @brief Compile `expr`. Throws (via #DNDS_check_throw) on parse error.
        void Compile(const std::string &expr);

        /// @brief Evaluate the compiled expression with the current variable values.
        real Evaluate();

        /// @brief Release the compiled expression and parser. Must be called
        /// before re-binding variables.
        void Clear();

        ~ExprtkWrapperEvaluator() { Clear(); }
    };
}