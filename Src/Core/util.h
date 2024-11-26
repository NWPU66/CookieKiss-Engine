#pragma once

// c

// cpp
#include <chrono>
#include <iostream>
#include <string>

// 3rdparty

// users

namespace cookieKissEngine {

/**
 * @brief 计时器类
 *
 */
class Timer {
public:
    Timer() : Timer("") {}
    explicit Timer(std::string message)
        : m_message(std::move(message)), m_startTime(std::chrono::system_clock::now())
    {
    }

    ~Timer()
    {
        auto endTime = std::chrono::system_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_startTime);
        std::cout << "Timer(): " << m_message << " Time:  " << duration.count() / 1000.0f << " s\n";
    }

private:
    std::string                                        m_message;
    std::chrono::time_point<std::chrono::system_clock> m_startTime;
};

}  // namespace cookieKissEngine
