#ifndef MS_FUSION_COMMON_RUNTIME_LOG_H
#define MS_FUSION_COMMON_RUNTIME_LOG_H

#include <cstddef>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace ms_fusion {

enum class LogProfile {
    RELEASE,
    DEBUG,
};

struct RuntimeLogConfig {
    LogProfile profile = LogProfile::RELEASE;
    bool verbose_runtime_log = false;
    bool enable_timing_stats = false;
    bool print_config_on_startup = false;
};

// 本地日志文件配置；控制台输出始终开启。
// 每次启动在 directory 下新建一个以启动时间戳命名的文件（YYYYmmdd_HHMMSS.log），
// 不追加旧文件；单个文件超过 max_file_bytes 时按当时时间戳再开一个新文件。
// 目录里的日志文件总数超过 max_files 时删除最旧的，防止无限增长。
struct LogFileConfig {
    std::string directory = "runtime_log";  // 相对路径相对于进程当前工作目录
    std::size_t max_file_bytes = 32u << 20;
    std::size_t max_files = 10;
    double flush_interval_s = 1.0;          // 定时 flush；WARNING 及以上立即 flush
};

LogProfile parseLogProfile(const std::string& profile_str);

class RuntimeLog {
public:
    static void configure(const RuntimeLogConfig& config);
    static LogProfile profile();
    static bool verboseRuntime();
    static bool enableTimingStats();
    static bool printConfigOnStartup();

    // 开始同时写本地日志文件（不依赖配置文件，进程启动即可调用）。失败时返回 false
    // 并继续只输出控制台，不影响主流程。调用前的日志只在控制台。
    static bool startFileLogging(const LogFileConfig& config = LogFileConfig());
    // flush 并关闭日志文件，之后只输出控制台。进程退出前调用。
    static void stopFileLogging();
    // 日志目录的绝对路径；未开启文件日志时为空
    static std::string logDirectory();
};

// 日志严重级别；后端是 spdlog（只在 runtime_log.cpp 里依赖，不进公开头文件）
enum class LogSeverity {
    INFO,
    WARNING,
    ERROR,
};

namespace log_detail {

// 按线程、按输出通道（INFO 对应原 std::cout，WARNING/ERROR 对应原 std::cerr）缓存的行缓冲。
// 和原来共用 std::cout/std::cerr 的效果一致：
//  - 写出的内容以换行结尾时才提交为一条日志记录；没到行尾的片段留在缓冲里，
//    由后续语句接着拼，所以分多条语句拼成一行的老写法不需要改；
//  - std::fixed / std::setprecision 等格式状态在同一线程的后续语句里继续生效。
// 记录的源码位置取这一行第一个片段所在的位置。
struct PendingLine {
    std::ostringstream buf;
    const char* file = nullptr;
    int line = 0;
    LogSeverity severity = LogSeverity::INFO;
    bool has_content = false;
};

inline PendingLine& pendingLine(LogSeverity severity) {
    static thread_local PendingLine lines[2];
    return lines[severity == LogSeverity::INFO ? 0 : 1];
}

// 缓冲以换行结尾时提交（去掉末尾换行，后端自己加），否则继续等后续片段
void commitIfLineComplete(PendingLine& pending);

}  // namespace log_detail

class LogStream {
public:
    LogStream(bool enabled, LogSeverity severity, const char* file, int line)
        : enabled_(enabled), severity_(severity), file_(file), line_(line) {}

    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;

    ~LogStream() {
        if (!wrote_) {
            return;
        }
        try {
            log_detail::commitIfLineComplete(log_detail::pendingLine(severity_));
        } catch (...) {
            // 日志失败不能影响主流程（析构也可能发生在异常展开过程中）
        }
    }

    template <typename T>
    LogStream& operator<<(const T& value) {
        if (enabled_) {
            pending() << value;
        }
        return *this;
    }

    using Manip = std::ostream& (*)(std::ostream&);
    LogStream& operator<<(Manip manip) {
        if (enabled_) {
            pending() << manip;
        }
        return *this;
    }

private:
    std::ostringstream& pending() {
        auto& line = log_detail::pendingLine(severity_);
        if (!line.has_content) {
            line.file = file_;
            line.line = line_;
            line.has_content = true;
        }
        if (static_cast<int>(severity_) > static_cast<int>(line.severity)) {
            line.severity = severity_;
        }
        wrote_ = true;
        return line.buf;
    }

    bool enabled_;
    bool wrote_ = false;
    LogSeverity severity_;
    const char* file_;
    int line_;
};

// printf 风格日志：格式化后按同样的行缓冲规则提交（格式串里的 \n 决定行尾）
void logPrintf(LogSeverity severity, const char* file, int line, const char* format, ...)
    __attribute__((format(printf, 4, 5)));

// 有开关的日志：verbose_runtime_log / enable_timing_stats / print_config_on_startup
#define MSF_VLOG                                                                    \
    ms_fusion::LogStream(ms_fusion::RuntimeLog::verboseRuntime(),                   \
                         ms_fusion::LogSeverity::INFO, __FILE__, __LINE__)
#define MSF_TLOG                                                                    \
    ms_fusion::LogStream(ms_fusion::RuntimeLog::enableTimingStats(),                \
                         ms_fusion::LogSeverity::INFO, __FILE__, __LINE__)
#define MSF_CLOG                                                                    \
    ms_fusion::LogStream(ms_fusion::RuntimeLog::printConfigOnStartup(),             \
                         ms_fusion::LogSeverity::INFO, __FILE__, __LINE__)

// 无开关的日志：一般信息 / 告警 / 错误
#define MSF_ILOG ms_fusion::LogStream(true, ms_fusion::LogSeverity::INFO, __FILE__, __LINE__)
#define MSF_WLOG ms_fusion::LogStream(true, ms_fusion::LogSeverity::WARNING, __FILE__, __LINE__)
#define MSF_ELOG ms_fusion::LogStream(true, ms_fusion::LogSeverity::ERROR, __FILE__, __LINE__)
#define MSF_ILOGF(...) \
    ms_fusion::logPrintf(ms_fusion::LogSeverity::INFO, __FILE__, __LINE__, __VA_ARGS__)
#define MSF_WLOGF(...) \
    ms_fusion::logPrintf(ms_fusion::LogSeverity::WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define MSF_ELOGF(...) \
    ms_fusion::logPrintf(ms_fusion::LogSeverity::ERROR, __FILE__, __LINE__, __VA_ARGS__)

// 终端整行着色，只用于少数需要一眼认出的结果（reloc 修正被应用 / 被门控拒绝）。
// 用法：MSF_ILOG << kLogGreen << "..." << kLogColorReset << std::endl;
// grep/sed 按子串匹配不受首尾转义序列影响
inline constexpr const char* kLogGreen = "\033[1;32m";
inline constexpr const char* kLogRed = "\033[1;31m";
inline constexpr const char* kLogColorReset = "\033[0m";

}  // namespace ms_fusion

#endif  // MS_FUSION_COMMON_RUNTIME_LOG_H
