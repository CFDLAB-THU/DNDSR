#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>

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
            if (v.i == UnInitIndex)
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
        std::string log_name_bare;
        std::string log_name_suffix;

        std::string delim = ",";
        int iBlock = 0;
        int64_t n_line_max = INT64_MAX;
        void update_ofstream(const std::string &fname)
        {
            std::filesystem::path outFile{fname};
            std::filesystem::create_directories(outFile.parent_path() / ".");
            pOs = std::make_unique<std::ofstream>(fname);
            DNDS_check_throw_info(*pOs, "csv file [" + fname + "] did not open");
        }

    public:
        template <class T_titles>
        CsvLog(T_titles &&n_titles, std::unique_ptr<std::ostream> n_pOs)
            : titles(std::forward<T_titles>(n_titles)), pOs(std::move(n_pOs)) {}

        template <class T_titles>
        CsvLog(T_titles &&n_titles,
               const std::string &n_log_name_bare,
               const std::string &n_log_name_suffix = ".csv",
               int64_t n_n_line_max = 10000)
            : titles(std::forward<T_titles>(n_titles)),
              log_name_bare(n_log_name_bare),
              log_name_suffix(n_log_name_suffix),
              n_line_max(n_n_line_max)
        {
            update_ofstream(log_name_bare + n_log_name_suffix);
        }

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
            if (n_line >= n_line_max)
            {
                n_line = 0;
                iBlock++;
                DNDS_check_throw_info(log_name_bare.length(), "need to give a log name");
                update_ofstream(log_name_bare + "_" + std::to_string(iBlock) + log_name_suffix);
            }
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