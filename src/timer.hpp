#include <chrono>
#include <cuda_runtime.h>

#include "utilities.hpp"


namespace gmf {

class Timer {
public:
    Timer() {}

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual float duration() = 0;
};

class TimerCPU : public Timer {
public:
    virtual void start() override;
    virtual void stop() override;
    virtual float duration() override;
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_stop;
};

class TimerGPU : public Timer {
public:
    TimerGPU();
    ~TimerGPU();
    
    virtual void start() override;
    virtual void stop() override;
    virtual float duration() override;

private:
    cudaEvent_t m_start, m_stop;
};

} // namespace gmf
