#include "Defines.hpp"

// #ifdef _MSC_VER
// #define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
// #endif
#include <codecvt>
#include <boost/stacktrace.hpp>
// #include <cpptrace.hpp>

#if defined(linux) || defined(_UNIX) || defined(__linux__)
#include <unistd.h>
#define _isatty isatty
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#define NOMINMAX
#include <io.h>
#endif

extern "C" void DNDS_signal_handler(int signal)
{
    std::cerr << __DNDS_getTraceString() << "\n";
    std::cerr << "Signal " + std::to_string(signal) << std::endl;
    std::signal(signal, SIG_DFL);
    std::raise(signal);
}

namespace DNDS
{
    static bool ostreamIsTTY(std::ostream &ostream)
    {
        if (&ostream == &std::cout)
            return _isatty(fileno(stdout));
        if (&ostream == &std::cerr)
            return _isatty(fileno(stderr));
        return false;
    }

    std::ostream *logStream;

    bool useCout = true;

    std::ostream &log() { return useCout ? std::cout : *logStream; }

    bool logIsTTY() { return ostreamIsTTY(*logStream); }

    void setLogStream(std::ostream *nstream) { useCout = false, logStream = nstream; }

    std::string getStringForceWString(const std::wstring &v)
    {
        // std::vector<char> buf(v.size());
        // std::wcstombs(buf.data(), v.data(), v.size());
        // return std::string{buf.data()};
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_DEPRECATED_DECLARATIONS
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        return converter.to_bytes(v); // TODO: on windows use WideCharToMultiByte()
        DISABLE_WARNING_POP
    }
}

namespace DNDS
{
    int get_env_OMP_NUM_THREADS()
    {
        static int ret{-1};
        if (ret == -1)
        {
            const char *env = std::getenv("OMP_NUM_THREADS");
            ret = 0;
            if (env)
                try
                {
                    ret = std::stoi(env);
                }
                catch (...)
                {
                }
        }
        return ret;
    }

    int get_env_DNDS_DIST_OMP_NUM_THREADS()
    {
        static int ret{-1};
        if (ret == -1)
        {
            const char *env = std::getenv("DNDS_DIST_OMP_NUM_THREADS");
            ret = 0;
            if (env)
                try
                {
                    ret = std::stoi(env);
                }
                catch (...)
                {
                }
        }
        return ret;
    }
}

/********************************/
// workaround for cpp trace
std::string __DNDS_getTraceString()
{
    std::stringstream ss;
    ss << boost::stacktrace::stacktrace();
    return ss.str();
    // return cpptrace::generate_trace().to_string();
}