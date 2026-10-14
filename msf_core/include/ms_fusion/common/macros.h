#ifndef MS_FUSION_COMMON_MACROS_H
#define MS_FUSION_COMMON_MACROS_H

#include <mutex>
#include <memory>
#include <iostream>

namespace ms_fusion {

// 宏定义一个类实现单例需要做的一些工作。
// 每个单例类最好将其构造函数和虚析构函数设置为private
#define SINGLETON_DECLARE(theclass) \
public:                             \
    static theclass* SnglPtr();     \
    static void FreeSingleton();    \
                                    \
private:                            \
    static theclass* m_s##theclass; \
    static std::mutex m_mutex##theclass;

#define SINGLETON_IMPLEMENT(theclass)                        \
    theclass* theclass::m_s##theclass = NULL;                \
    std::mutex theclass::m_mutex##theclass;                  \
    theclass* theclass::SnglPtr()                            \
    {                                                        \
        std::lock_guard<std::mutex> lock(m_mutex##theclass); \
        if (NULL == m_s##theclass) {                         \
            m_s##theclass = new (std::nothrow) theclass;     \
        }                                                    \
        return m_s##theclass;                                \
    }                                                        \
    void theclass::FreeSingleton()                           \
    {                                                        \
        if (NULL != m_s##theclass) {                         \
            delete m_s##theclass;                            \
            m_s##theclass = NULL;                            \
        }                                                    \
    }

} // namespace ms_fusion

#endif // MS_FUSION_COMMON_MACROS_H