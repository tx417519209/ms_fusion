#ifndef MS_FUSION_ASSERT_H
#define MS_FUSION_ASSERT_H

#include "ms_fusion/common/runtime_log.h"

#include <cstdlib>

namespace ms_fusion {

#define UNUSED(x) (void)(x)

#define MS_FUSION_ATTRIBUTE_NORETURN __attribute__((noreturn))

inline MS_FUSION_ATTRIBUTE_NORETURN void assertionFailed(char const* expr, 
                                                         char const* function,
                                                         char const* file, 
                                                         long line) {
    MSF_ELOG << "***** Assertion (" << expr << ") failed in " << function << ":\n"
             << file << ':' << line << ":" << std::endl;
    std::abort();
}

inline MS_FUSION_ATTRIBUTE_NORETURN void assertionFailedMsg(char const* expr, 
                                                            char const* msg,
                                                            char const* function, 
                                                            char const* file,
                                                            long line) {
    MSF_ELOG << "***** Assertion (" << expr << ") failed in " << function << ":\n"
             << file << ':' << line << ": " << msg << std::endl;
    std::abort();
}

} // namespace ms_fusion

#define MS_FUSION_LIKELY(x) __builtin_expect(x, 1)

#if defined(MS_FUSION_DISABLE_ASSERTS)

#define MS_FUSION_ASSERT(expr) ((void)0)

#define MS_FUSION_ASSERT_MSG(expr, msg) ((void)0)

#define MS_FUSION_ASSERT_STREAM(expr, msg) ((void)0)

#else

#define MS_FUSION_ASSERT(expr)  \
    (MS_FUSION_LIKELY(!!(expr)) \
        ? ((void)0)             \
        : ::ms_fusion::assertionFailed(#expr, __PRETTY_FUNCTION__, __FILE__, __LINE__))

#define MS_FUSION_ASSERT_MSG(expr, msg) \
    (MS_FUSION_LIKELY(!!(expr))         \
        ? ((void)0)                     \
        : ::ms_fusion::assertionFailedMsg(#expr, msg, __PRETTY_FUNCTION__, __FILE__, __LINE__))

#define MS_FUSION_ASSERT_STREAM(expr, msg) \
    (MS_FUSION_LIKELY(!!(expr))            \
        ? ((void)0)                        \
        : ([&] { MSF_ELOG << msg << std::endl; }(), /* 日志须在 abort 前提交 */ \
            ::ms_fusion::assertionFailed(#expr, __PRETTY_FUNCTION__, __FILE__, __LINE__)))

#endif

#endif // MS_FUSION__ASSERT_H