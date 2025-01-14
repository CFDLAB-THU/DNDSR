#pragma once

#include <iostream>
#include <iomanip>

#include "Defines.hpp"

namespace DNDS
{

    struct LogSimpleDIValue
    {
        int64_t i{UnInitIndex}; // UnInitIndex for double
        double d{0};

        template <class T>
        std::enable_if_t<std::is_integral_v<T>, LogSimpleDIValue &> operator=(T v)
        {
            i = v;
            if (i == UnInitIndex)
                d = UnInitReal;
            return *this;
        }

        template <class T>
        std::enable_if_t<!std::is_integral_v<T>, LogSimpleDIValue &> operator=(T v)
        {
            d = v;
            i = UnInitIndex;
            return *this;
        }

        friend std::ostream &operator<<(std::ostream &o, const LogSimpleDIValue &v)
        {
            if(v.i == UnInitIndex)
                o << v.d;
            else 
                o << v.i;
            return o;
        }
    };

    class CsvLog
    {
        std::vector<std::string> titles;
        std::unique_ptr<std::ostream> pOs;
        int64_t n_line{0};

        std::string delim = ",";

    public:
        template <class T_titles>
        CsvLog(T_titles &&n_titles, std::unique_ptr<std::ostream> n_pOs)
            : titles(std::forward<T_titles>(n_titles)), pOs(std::move(n_pOs)){};

        template <class TMap>
        void WriteLine(TMap &&title_to_value, int nPrecision)
        {
            if (n_line == 0)
                WriteTitle();
            (*pOs) << std::setprecision(nPrecision) << std::scientific;
            for (size_t i = 0; i < titles.size(); i++)
                (*pOs) << title_to_value[titles[i]] << ((i == (titles.size() - 1)) ? "" : ",");
            (*pOs) << std::endl;
            n_line++;
        }

        void WriteTitle()
        {
            for (size_t i = 0; i < titles.size(); i++)
                (*pOs) << titles[i] << ((i == (titles.size() - 1)) ? "" : ",");
            (*pOs) << std::endl;
        }
    };

    using tLogSimpleDIValueMap = std::map<std::string, LogSimpleDIValue>;
}