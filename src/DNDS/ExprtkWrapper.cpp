/// @file ExprtkWrapper.cpp
/// @brief exprtk-backed implementations of @ref DNDS::ExprtkWrapperEvaluator "ExprtkWrapperEvaluator"'s Compile,
/// Evaluate, and Clear methods. Kept out-of-line so that exprtk's heavy
/// templates stay confined to one translation unit.

#include "ExprtkWrapper.hpp"
#include "ExprtkPCH.hpp"
#include <exprtk.hpp>

namespace DNDS
{
    using symbol_table_t = exprtk::symbol_table<real>;
    using expression_t = exprtk::expression<real>;
    using parser_t = exprtk::parser<real>;

    void ExprtkWrapperEvaluator::Compile(const std::string &expr)
    {
        this->Clear();

        auto *pst = new symbol_table_t;
        symbol_table_t &st = *pst;
        _ptr_st = static_cast<void *>(pst);

        st.add_infinity();
        st.add_pi();
        // TODO: add thread safety?
        static exprtk::rtl::io::println<real> println;
        st.add_function("println", println);

        for (auto &[k, v] : _vars)
            st.add_variable(k, v);

        for (auto &[k, v] : _varVecs)
            st.add_vector(k, v.data(), v.size());

        auto *pexp = new expression_t;
        expression_t &exp = *pexp;
        _ptr_exp = static_cast<void *>(pexp);

        exp.register_symbol_table(st);

        auto *pparser = new parser_t;
        parser_t &parser = *pparser;
        _ptr_parser = static_cast<void *>(pparser);

        auto compile_ok = parser.compile(expr, exp);
        std::string error_info = parser.error() + "\n";
        for (size_t i = 0; i < parser.error_count(); i++)
        {
            auto err = parser.get_error(i);
            error_info.append(
                fmt::format("Error [{}], at [{}]\n", i, err.token.position) +
                fmt::format("\tType: {}\n", exprtk::parser_error::to_str(err.mode)) +
                fmt::format("\tMsg: {}\n", err.diagnostic));
        }
        DNDS_check_throw_info(
            compile_ok,
            "exprtk compiling of === \n" +
                expr +
                "\n=== failed: " + error_info);
        _compiled = true;
    }

    real ExprtkWrapperEvaluator::Evaluate()
    {
        DNDS_assert(this->Compiled());
        DNDS_assert(_ptr_exp);
        return static_cast<expression_t *>(_ptr_exp)->value();
    }

    void ExprtkWrapperEvaluator::Clear()
    {
        if (_ptr_parser)
        {
            delete static_cast<parser_t *>(_ptr_parser);
            _ptr_parser = nullptr;
        }
        if (_ptr_exp)
        {
            delete static_cast<expression_t *>(_ptr_exp);
            _ptr_exp = nullptr;
        }
        if (_ptr_st)
        {
            delete static_cast<symbol_table_t *>(_ptr_st);
            _ptr_st = nullptr;
        }
        _compiled = false;
    }
}