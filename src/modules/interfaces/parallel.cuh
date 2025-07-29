#pragma once

class Parallel {
public:
    Parallel() {}
    Parallel(uint gpu_threads, uint cpu_threads = 1)
            : m_max_threads_per_block(gpu_threads), m_omp_threads(cpu_threads) {}

    void set_gpu_threads(uint threads) { m_max_threads_per_block = threads; }
    void set_cpu_threads(uint threads) { m_omp_threads = threads; }
    
protected:
    uint m_max_threads_per_block = 1;
    uint m_omp_threads = 1;
};
