#include "ms_fusion/common/runtime_log.h"

#include <spdlog/logger.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <memory>
#include <regex>
#include <mutex>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(MSF_USE_AXERA_SUPERPOINT)
#include "infer_api.h"
#endif

namespace ms_fusion {

namespace {

RuntimeLogConfig g_config;

// 只用自己命名的 logger，不碰 spdlog 默认 logger / 全局 shutdown：同进程里 ROS 的
// rcl_logging_spdlog 用的是同一份系统 spdlog。
constexpr char kLoggerName[] = "ms_fusion";
// 控制台只输出正文（和原来 std::cout 一致，现有日志分析不受影响），WARNING/ERROR 着色；
// 文件每行是 "[时间] 正文"，WARNING/ERROR 在正文前多一个 [warn]/[error]（%* 见 LevelTagFlag）。
constexpr char kConsolePattern[] = "%^%v%$";
constexpr char kFilePattern[] = "[%Y-%m-%d %H:%M:%S.%f] %*%v";

// 多进程共用一个日志文件：设置了这个环境变量（值为 YYYYmmdd_HHMMSS）时不按本进程的启动时间
// 新建文件，而是追加写 <directory>/<值>.log。launch 给 ms_fusion_node 和 ms_fusion_loop_node
// 设成同一个值，两个进程的日志就写进同一个文件。
constexpr char kSessionEnv[] = "MSF_LOG_SESSION";

// 文件里 INFO 不标级别（正文自带 [tracking]/[timing] 这类模块标签），WARNING/ERROR 标出来方便 grep
class LevelTagFlag final : public spdlog::custom_flag_formatter {
public:
    void format(const spdlog::details::log_msg& msg, const std::tm&,
                spdlog::memory_buf_t& dest) override {
        const char* tag = nullptr;
        switch (msg.level) {
            case spdlog::level::warn:
                tag = "[warn] ";
                break;
            case spdlog::level::err:
                tag = "[error] ";
                break;
            case spdlog::level::critical:
                tag = "[critical] ";
                break;
            default:
                return;
        }
        dest.append(tag, tag + std::strlen(tag));
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return std::make_unique<LevelTagFlag>();
    }
};

std::mutex g_logger_mutex;
std::shared_ptr<spdlog::logger> g_logger;  // g_logger_mutex 保护
std::string g_log_directory;               // g_logger_mutex 保护

// 按时间戳命名的日志文件 sink：启动时新建文件，超过大小上限时再开一个，
// 每开一个新文件就把目录里最旧的日志删到只剩 max_files 个。
// 文件以 O_APPEND 打开，每条记录一次 write(2) 直接落盘、不经用户态缓冲，
// 多个进程写同一个文件时整行不会被拆开交错。
class TimestampedFileSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
    // session_stamp 非空时多进程共用 <session_stamp>.log（见 kSessionEnv），否则按当前时间新建
    TimestampedFileSink(std::filesystem::path directory, std::size_t max_file_bytes,
                        std::size_t max_files, std::string session_stamp)
        : directory_(std::move(directory)),
          max_file_bytes_(max_file_bytes),
          max_files_(std::max<std::size_t>(1, max_files)),
          session_stamp_(std::move(session_stamp)) {
        std::filesystem::create_directories(directory_);
        openNewFile();
    }

    ~TimestampedFileSink() override { closeFile(); }

    std::string currentFile() const { return path_.filename().string(); }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        // 用 fstat 取文件实际大小（共享文件时其他进程也在写）
        const std::size_t size = currentSize();
        if (size > 0 && size + formatted.size() > max_file_bytes_) {
            closeFile();
            openNewFile();
        }
        writeAll(formatted.data(), formatted.size());
    }

    void flush_() override {}  // 每条记录都已直接 write(2)

private:
    // 解析 YYYYmmdd_HHMMSS[_N].log，得到排序键（时间戳，序号）；不是日志文件名返回 false
    static bool parseLogFileName(const std::string& name, std::pair<std::string, long>* key) {
        static const std::regex kPattern(R"((\d{8}_\d{6})(?:_(\d+))?\.log)");
        std::smatch match;
        if (!std::regex_match(name, match, kPattern)) {
            return false;
        }
        key->first = match[1].str();
        key->second = match[2].matched ? std::stol(match[2].str()) : 0;
        return true;
    }

    void openNewFile() {
        std::string stamp;
        long seq = 0;
        const auto make_path = [&](long n) {
            return directory_ / (stamp + (n == 0 ? "" : "_" + std::to_string(n)) + ".log");
        };
        std::filesystem::path path;
        if (!session_stamp_.empty()) {
            // 共享文件：第 k 次滚动固定写 <session>_k.log，已存在就追加。各进程各自发现
            // 文件超限后算出的是同一个新文件名，滚动后仍然写在一起。
            stamp = session_stamp_;
            seq = last_seq_ + 1;
            path = make_path(seq);
        } else {
            const std::time_t now = std::time(nullptr);
            std::tm tm{};
            localtime_r(&now, &tm);
            char buf[32];
            std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
            stamp = buf;
            // 同一秒内滚动或重启时加序号 _N。序号在本进程内只增不减，被清理掉的旧文件名
            // 也不会复用，保证文件名顺序就是写入顺序、不追加到已有文件。
            seq = stamp == last_stamp_ ? last_seq_ + 1 : 0;
            path = make_path(seq);
            while (std::filesystem::exists(path)) {
                path = make_path(++seq);
            }
        }
        last_stamp_ = stamp;
        last_seq_ = seq;
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (fd_ < 0) {
            spdlog::throw_spdlog_ex("failed opening log file " + path.string(), errno);
        }
        path_ = path;
        removeOldFiles(path);
    }

    void closeFile() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    std::size_t currentSize() const {
        struct stat st {};
        if (fd_ < 0 || ::fstat(fd_, &st) != 0) {
            return 0;
        }
        return static_cast<std::size_t>(st.st_size);
    }

    void writeAll(const char* data, std::size_t size) {
        while (size > 0) {
            const ssize_t n = ::write(fd_, data, size);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                spdlog::throw_spdlog_ex("failed writing log file " + path_.string(), errno);
            }
            data += n;
            size -= static_cast<std::size_t>(n);
        }
    }

    void removeOldFiles(const std::filesystem::path& keep) {
        std::vector<std::pair<std::pair<std::string, long>, std::filesystem::path>> files;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(directory_, ec)) {
            std::pair<std::string, long> key;
            if (entry.is_regular_file(ec) &&
                parseLogFileName(entry.path().filename().string(), &key)) {
                files.emplace_back(key, entry.path());
            }
        }
        if (files.size() <= max_files_) {
            return;
        }
        // 按（时间戳，序号）从旧到新排序，删最旧的
        std::sort(files.begin(), files.end());
        std::size_t to_remove = files.size() - max_files_;
        for (const auto& [key, file] : files) {
            if (to_remove == 0) {
                break;
            }
            if (file == keep) {
                continue;
            }
            std::filesystem::remove(file, ec);
            --to_remove;
        }
    }

    std::filesystem::path directory_;
    std::size_t max_file_bytes_;
    std::size_t max_files_;
    std::string session_stamp_;
    int fd_ = -1;
    std::filesystem::path path_;
    std::string last_stamp_;
    long last_seq_ = -1;
};

std::shared_ptr<spdlog::sinks::sink> consoleSink() {
    static const auto sink = [] {
        auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console->set_pattern(kConsolePattern);
        // INFO 保持终端默认色（与原 cout 一致），只有 WARN/ERROR 上色
        console->set_color(spdlog::level::info, console->reset);
        return console;
    }();
    return sink;
}

std::shared_ptr<spdlog::logger> makeConsoleLogger() {
    auto logger = std::make_shared<spdlog::logger>(kLoggerName, consoleSink());
    logger->set_level(spdlog::level::trace);
    return logger;
}

std::shared_ptr<spdlog::logger> currentLogger() {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    if (!g_logger) {
        g_logger = makeConsoleLogger();
    }
    return g_logger;
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

}  // namespace

LogProfile parseLogProfile(const std::string& profile_str) {
    const std::string normalized = toLower(profile_str);
    if (normalized == "debug" || normalized == "dev") {
        return LogProfile::DEBUG;
    }
    return LogProfile::RELEASE;
}

void RuntimeLog::configure(const RuntimeLogConfig& config) {
    g_config = config;
#if defined(MSF_USE_AXERA_SUPERPOINT)
    ax_infer_log::sync_run_timing_enabled() = config.verbose_runtime_log;
#endif
}

LogProfile RuntimeLog::profile() {
    return g_config.profile;
}

bool RuntimeLog::verboseRuntime() {
    return g_config.verbose_runtime_log;
}

bool RuntimeLog::enableTimingStats() {
    return g_config.enable_timing_stats;
}

bool RuntimeLog::printConfigOnStartup() {
    return g_config.print_config_on_startup;
}

bool RuntimeLog::startFileLogging(const LogFileConfig& config) {
    try {
        const std::filesystem::path directory =
            std::filesystem::absolute(std::filesystem::path(config.directory));
        std::string session_stamp;
        if (const char* env = std::getenv(kSessionEnv); env != nullptr) {
            static const std::regex kStampPattern(R"(\d{8}_\d{6})");
            if (std::regex_match(env, kStampPattern)) {
                session_stamp = env;
            }
        }
        auto file_sink = std::make_shared<TimestampedFileSink>(
            directory, config.max_file_bytes, config.max_files, session_stamp);
        auto formatter = std::make_unique<spdlog::pattern_formatter>();
        formatter->add_flag<LevelTagFlag>('*').set_pattern(kFilePattern);
        file_sink->set_formatter(std::move(formatter));
        auto logger = std::make_shared<spdlog::logger>(
            kLoggerName, spdlog::sinks_init_list{consoleSink(), file_sink});
        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::warn);
        {
            std::lock_guard<std::mutex> lock(g_logger_mutex);
            if (g_logger) {
                g_logger->flush();
            }
            // 注册到 registry 才能被 flush_every 的后台线程定时 flush
            spdlog::drop(kLoggerName);
            spdlog::register_logger(logger);
            g_logger = logger;
            g_log_directory = directory.string();
        }
        const auto interval_s = static_cast<long>(std::max(1.0, std::round(config.flush_interval_s)));
        spdlog::flush_every(std::chrono::seconds(interval_s));
        currentLogger()->info("[log] log file: {}", file_sink->currentFile());
        return true;
    } catch (const std::exception& e) {
        currentLogger()->error("[log] failed to open log file under {}: {}", config.directory,
                               e.what());
        return false;
    }
}

std::string RuntimeLog::logDirectory() {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    return g_log_directory;
}

void RuntimeLog::stopFileLogging() {
    try {
        std::lock_guard<std::mutex> lock(g_logger_mutex);
        if (g_logger) {
            g_logger->flush();
        }
        spdlog::drop(kLoggerName);
        g_logger = makeConsoleLogger();
        g_log_directory.clear();
    } catch (...) {
        // 日志失败不能影响退出流程
    }
}

namespace log_detail {

void commitIfLineComplete(PendingLine& pending) {
    std::string text = pending.buf.str();
    if (text.empty()) {
        // 只设置了格式（std::fixed 等）的语句：没有内容可提交，格式状态保留
        pending.has_content = false;
        return;
    }
    if (text.back() != '\n') {
        return;
    }
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    const char* file = pending.file != nullptr ? pending.file : __FILE__;
    const int line = pending.line;
    const LogSeverity severity = pending.severity;
    // 只清内容，保留格式标志，和原来共用 std::cout 的表现一致
    pending.buf.str(std::string());
    pending.buf.clear();
    pending.has_content = false;
    pending.severity = LogSeverity::INFO;
    if (text.empty()) {
        return;  // 只有换行的语句（如单独的 << std::endl）不产生空记录
    }
    spdlog::level::level_enum level = spdlog::level::info;
    if (severity == LogSeverity::WARNING) {
        level = spdlog::level::warn;
    } else if (severity == LogSeverity::ERROR) {
        level = spdlog::level::err;
    }
    // 一条语句里的多行（如 [timing] 整块报告）拆成逐行记录，文件里每行都带时间
    const auto logger = currentLogger();
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = text.find('\n', begin);
        if (end == std::string::npos) {
            end = text.size();
        }
        std::size_t stop = end;
        if (stop > begin && text[stop - 1] == '\r') {
            --stop;
        }
        if (stop > begin) {
            logger->log(spdlog::source_loc{file, line, ""}, level,
                        spdlog::string_view_t(text.data() + begin, stop - begin));
        }
        begin = end + 1;
    }
}

}  // namespace log_detail

void logPrintf(LogSeverity severity, const char* file, int line, const char* format, ...) {
    try {
        va_list args;
        va_start(args, format);
        va_list args_copy;
        va_copy(args_copy, args);
        const int size = std::vsnprintf(nullptr, 0, format, args_copy);
        va_end(args_copy);
        if (size < 0) {
            va_end(args);
            return;
        }
        std::vector<char> buffer(static_cast<size_t>(size) + 1);
        std::vsnprintf(buffer.data(), buffer.size(), format, args);
        va_end(args);
        LogStream(true, severity, file, line) << buffer.data();
    } catch (...) {
        // 日志失败不能影响主流程
    }
}

}  // namespace ms_fusion
