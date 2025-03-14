#include "Defines.hpp"

// #ifdef _MSC_VER
// #define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
// #endif
#include <codecvt>
#include <boost/stacktrace.hpp>
// #include <cpptrace.hpp>

#if defined(linux) || defined(_UNIX) || defined(__linux__)
#include <unistd.h>
#include <sys/ioctl.h>
#define _isatty isatty
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#define NOMINMAX
#include <io.h>
#include <windows.h>
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
    bool ostreamIsTTY(std::ostream &ostream)
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

    int get_terminal_width()
    {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
        {
            return csbi.srWindow.Right - csbi.srWindow.Left + 1;
        }
#else
        struct winsize w;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0)
        {
            return w.ws_col;
        }
#endif
        return 80; // Default width if detection fails
    }

    void print_progress(std::ostream &os, double progress)
    {
        progress = std::clamp(progress, 0.0, 1.0);
        int term_width = ostreamIsTTY(os) ? get_terminal_width() : 80;
        int bar_width = std::max(10, term_width - 10);

        int pos = static_cast<int>(bar_width * progress);

        if (ostreamIsTTY(os))
        {
            os << "\r[";
            for (int i = 0; i < bar_width; ++i)
            {
                if (i < pos)
                    os << "=";
                else if (i == pos)
                    os << ">";
                else
                    os << " ";
            }
            os << "] " << std::setw(3) << static_cast<int>(progress * 100) << "% " << std::flush;
        }
        else
        {
            os << "[" << std::string(pos, '=') << ">" << std::string(bar_width - pos, ' ')
               << "] " << std::setw(3) << static_cast<int>(progress * 100) << "%" << std::endl;
        }
    }

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

namespace DNDS
{
    std::string GetSetVersionName(const std::string &ver)
    {
        static std::string ver_name = "UNKNOWN";
        if (ver.length())
            ver_name = ver;
        return ver_name;
    }
}