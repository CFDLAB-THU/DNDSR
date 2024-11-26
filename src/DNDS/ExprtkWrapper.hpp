#pragma once
#include <map>
#include "Defines.hpp"

namespace DNDS
{
    using tExprVars = std::map<std::string, real>;
    using tExprVarVecs = std::map<std::string, Eigen::Vector<real, Eigen::Dynamic>>;

    class ExprtkWrapperEvaluator
    {
        void *_ptr_st = nullptr;
        void *_ptr_exp = nullptr;
        void *_ptr_parser = nullptr;
        tExprVars _vars;
        tExprVarVecs _varVecs;
        bool _compiled = false;

    public:
        void AddScalar(const std::string &name, real init = 0)
        {
            Clear();
            _vars[name] = 0;
        }

        void AddVector(const std::string &name, int size)
        {
            Clear();
            _varVecs[name].resize(size);
        }

        real &Var(const std::string &name) { return _vars.at(name); }
        real &VarVec(const std::string &name, int i) { return _varVecs.at(name)(i); }
        index VarVecSize(const std::string &name) { return _varVecs.at(name).size(); }

        bool Compiled() const
        {
            return _compiled;
        }

        void Compile(const std::string &expr);

        real Evaluate();

        void Clear();

        ~ExprtkWrapperEvaluator() { Clear(); }
    };
}