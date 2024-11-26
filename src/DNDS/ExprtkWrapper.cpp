#include "ExprtkWrapper.hpp"
#include "ExprtkPCH.hpp"
#include <exprtk.hpp>

namespace DNDS
{
    typedef exprtk::symbol_table<real> symbol_table_t;
    typedef exprtk::expression<real> expression_t;
    typedef exprtk::parser<real> parser_t;

    void ExprtkWrapperEvaluator::Compile(const std::string &expr)
    {
        this->Clear();

        auto pst = new symbol_table_t;
        symbol_table_t &st = *pst;
        _ptr_st = static_cast<void *>(pst);

        st.add_infinity();
        st.add_pi();

        for (auto &[k, v] : _vars)
            st.add_variable(k, v);

        for (auto &[k, v] : _varVecs)
            st.add_vector(k, v.data(), v.size());

        auto pexp = new expression_t;
        expression_t &exp = *pexp;
        _ptr_exp = static_cast<void *>(pexp);

        exp.register_symbol_table(st);

        auto pparser = new parser_t;
        parser_t &parser = *pparser;
        _ptr_parser = static_cast<void *>(pparser);

        auto compile_ok = parser.compile(expr, exp);
        DNDS_assert_info(
            compile_ok,
            "exprtk compiling of === \n" +
                expr +
                "\n=== failed"); 
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