#include "timer.hpp"

#include <chrono>
#include <cuda_runtime.h>

#include "utilities.hpp"


namespace gmf {

void TimerCPU::start() {
    m_start = std::chrono::high_resolution_clock::now();
}

void TimerCPU::stop() {
    m_stop = std::chrono::high_resolution_clock::now();
}

float TimerCPU::duration() {
    std::chrono::duration<float, std::milli> duration = m_stop - m_start;
    return duration.count();
}

TimerGPU::TimerGPU() {
    cudaCheck(cudaEventCreate(&m_start));
    cudaCheck(cudaEventCreate(&m_stop));
}

TimerGPU::~TimerGPU() {
    cudaCheck(cudaEventDestroy(m_start));
    cudaCheck(cudaEventDestroy(m_stop));
}
    
void TimerGPU::start() {
    cudaCheck(cudaEventRecord(m_start));
}

void TimerGPU::stop() {
    cudaCheck(cudaEventRecord(m_stop));
}

float TimerGPU::duration() {
    cudaCheck(cudaEventSynchronize(m_stop));
    float milliseconds;
    cudaCheck(cudaEventElapsedTime(&milliseconds, m_start, m_stop));
    return milliseconds;
}

} // namespace gmf
