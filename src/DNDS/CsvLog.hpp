#pragma once
/// @file CsvLog.hpp
/// @brief Streaming CSV writer with auto-rotation and a dual int-or-double value type.

#include <iostream>
#include <iomanip>
#include <fstream>

#include "Defines.hpp"

namespace DNDS
{

    /**
     * @brief Storage-agnostic "either int or double" value with CSV-friendly streaming.
     *
     * @details Each instance holds either an integer (`i`) or a double (`d`);
     * the unused field is set to the #UnInitIndex / #UnInitReal sentinel. The
     * assignment operator dispatches on the source type, and `operator<<`
     * prints whichever field is live. Designed for columns where some rows
     * are integer counts and others are floating residuals.
     */
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

    /**
     * @brief Append-only CSV logger with automatic file rotation after a line-count limit.
     *
     * @details One instance writes rows to a file, using a caller-provided
     * ordered list of column titles. After `n_line_max` rows the output is
     * rotated to `<bare>_<block>.suffix` to keep any single file size bounded.
     *
     * Typical use: one `CsvLog` per solver per rank-0 process, recording
     * per-iteration diagnostics with #LogSimpleDIValue entries keyed by title.
     */
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
        /// @brief Construct a logger with externally-provided titles and output stream.
        template <class T_titles>
        CsvLog(T_titles &&n_titles, std::unique_ptr<std::ostream> n_pOs)
            : titles(std::forward<T_titles>(n_titles)), pOs(std::move(n_pOs)) {}

        /// @brief Construct a logger that creates / manages its own output files.
        /// @param n_titles          Column titles (ordered).
        /// @param n_log_name_bare   Base path without suffix, e.g. "out/run".
        /// @param n_log_name_suffix Filename suffix including dot (default ".csv").
        /// @param n_n_line_max      Rotate to a new file every this many lines.
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

        /// @brief Append one CSV row by looking up each column in `title_to_value`.
        /// @tparam TMap  Map-like supporting `operator[](std::string)` returning
        ///               something streamable.
        /// @param nPrecision Decimal digits used for floating-point output.
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

        /// @brief Emit the title row (called automatically on the first #WriteLine).
        void WriteTitle()
        {
            for (size_t i = 0; i < titles.size(); i++)
                (*pOs) << titles[i] << ((i == (titles.size() - 1)) ? "" : ",");
            (*pOs) << std::endl;
        }
    };

    /// @brief Convenience alias: the `title -> value` map type passed to #CsvLog::WriteLine.
    using tLogSimpleDIValueMap = std::map<std::string, LogSimpleDIValue>;
}