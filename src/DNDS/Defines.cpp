/// @file Defines.cpp
/// @brief Implementations of symbols declared in @ref Defines.hpp: log stream
/// management, signal handler body, OpenMP helpers, version string accessor,
/// terminal / progress utilities.

#include "Defines.hpp"

// #ifdef _MSC_VER
// #define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
// #endif
#include <codecvt>
#include <cstring>
#include <fstream>
#include <boost/stacktrace.hpp>
#include <utility>
// #include <cpptrace.hpp>

#ifdef DNDS_UNIX_LIKE
#    include <unistd.h>
#    include <sys/ioctl.h>
#    define _isatty isatty
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#    define NOMINMAX
#    include <io.h>
#    include <windows.h>
#endif

extern "C" void DNDS_signal_handler(int signal)
{
    std::cerr << ::DNDS::getTraceString() << "\n";
    std::cerr << "Signal " + std::to_string(signal) << std::endl;
    std::signal(signal, SIG_DFL);
    std::raise(signal);
}

namespace DNDS
{
    class TeeStreamBuf : public std::streambuf
    {
    public:
        TeeStreamBuf(std::ostream *a, std::ostream *b, bool aIsTTY, bool bIsTTY)
            : _a(a), _b(b), _aIsTTY(aIsTTY), _bIsTTY(bIsTTY) {}

        bool anyBranchTTY() const { return _aIsTTY || _bIsTTY; }

    protected:
        int_type overflow(int_type c) override
        {
            if (c == traits_type::eof())
                return traits_type::not_eof(c);
            char ch = traits_type::to_char_type(c);
            bool wa = writeChar(_a, _aIsTTY, ch);
            bool wb = writeChar(_b, _bIsTTY, ch);
            return (!wa || !wb)
                       ? traits_type::eof()
                       : c;
        }

        int sync() override
        {
            _a->flush();
            _b->flush();
            return (_a->good() && _b->good()) ? 0 : -1;
        }

        std::streamsize xsputn(const char_type *s, std::streamsize n) override
        {
            bool hasCR = n > 0 && std::memchr(s, '\r', n) != nullptr;
            auto wa = writeBranch(_a, _aIsTTY || !hasCR, s, n);
            auto wb = writeBranch(_b, _bIsTTY || !hasCR, s, n);
            return std::min(wa, wb);
        }

    private:
        static bool writeChar(std::ostream *os, bool isTTY, char ch)
        {
            os->put(isTTY ? ch : (ch == '\r' ? '\n' : ch));
            return os->good();
        }

        static std::streamsize writeBranch(std::ostream *os, bool keepCR, const char_type *s, std::streamsize n)
        {
            if (keepCR)
            {
                os->write(s, n);
                return os->good() ? n : 0;
            }
            std::streamsize written = 0;
            for (; written < n && os->good(); written++)
                os->put(s[written] == '\r' ? '\n' : s[written]);
            return written;
        }

        std::ostream *_a;
        std::ostream *_b;
        bool _aIsTTY;
        bool _bIsTTY;
    };

    static ssp<std::ofstream> logFileStream;
    static ssp<TeeStreamBuf> logTeeBuf;

    bool ostreamIsTTY(std::ostream &ostream)
    {
        if (&ostream == &std::cout)
            return _isatty(fileno(stdout));
        if (&ostream == &std::cerr)
            return _isatty(fileno(stderr));
        if (auto *tee = dynamic_cast<TeeStreamBuf *>(ostream.rdbuf()))
            return tee->anyBranchTTY();
        return false;
    }

    ssp<std::ostream> logStream;

    bool useCout = true;

    std::ostream &log() { return useCout ? std::cout : *logStream; }

    bool logIsTTY() { return useCout ? ostreamIsTTY(std::cout) : ostreamIsTTY(*logStream); }

    void setLogStream(ssp<std::ostream> nstream)
    {
        logTeeBuf.reset();
        logFileStream.reset();
        useCout = false;
        logStream = std::move(nstream);
    }

    void setLogFile(const std::string &path)
    {
        auto file = std::make_shared<std::ofstream>(path);
        DNDS_check_throw_info(file->is_open(), "failed to open log file: " + path);
        auto tee = std::make_shared<TeeStreamBuf>(&std::cout, file.get(),
                                                  _isatty(fileno(stdout)), false);
        auto stream = std::make_shared<std::ostream>(tee.get());
        logFileStream = std::move(file);
        logTeeBuf = std::move(tee);
        useCout = false;
        logStream = std::move(stream);
    }

    void setLogStreamCout()
    {
        useCout = true;
        logStream.reset();
        logTeeBuf.reset();
        logFileStream.reset();
    }

    int get_terminal_width()
    {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
        {
            return csbi.srWindow.Right - csbi.srWindow.Left + 1;
        }
#else
        struct winsize w
        {
        };
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

        auto buildBar = [&]() -> std::string
        {
            std::string bar = "[";
            for (int i = 0; i < bar_width; ++i)
            {
                if (i < pos)
                    bar += "=";
                else if (i == pos)
                    bar += ">";
                else
                    bar += " ";
            }
            bar += fmt::format("] {:3d}%", static_cast<int>(progress * 100));
            return bar;
        };

        if (ostreamIsTTY(os))
        {
            os << "\r" << buildBar() << " " << std::flush;
        }
        else
        {
            os << buildBar() << std::endl;
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
                    ret = -1;
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
                    ret = -1;
                }
        }
        return ret;
    }
}

/********************************/
// workaround for cpp trace
std::string DNDS::getTraceString()
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
        if (!ver.empty())
            ver_name = ver;
        return ver_name;
    }
}
