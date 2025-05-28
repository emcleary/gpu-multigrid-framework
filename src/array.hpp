#pragma once

#include <cassert>
#include <cstddef> // size_t
#include <memory>

#include <cuda_runtime.h>

#include "memory.hpp"

namespace gmf {

class ArrayRaw {
protected:
    ArrayRaw() : m_N(0) {}
    ArrayRaw(const size_t N) : m_N(N) {}
    size_t m_N;
    double* m_ptr;

public:
    __host__ __device__
    inline double& operator[](int i) {
        assert(0 <= i && i < m_N);
        return m_ptr[i];
    }

    __host__ __device__
    inline const double& operator[](int i) const {
        assert(0 <= i && i < m_N);
        return m_ptr[i];
    }

    __host__ __device__
    inline double& front() {
        return m_ptr[0];
    }
    
    __host__ __device__
    inline const double& front() const {
        return m_ptr[0];
    }
    __host__ __device__
    inline double& back() {
        return m_ptr[m_N - 1];
    }
    
    __host__ __device__
    inline const double& back() const {
        return m_ptr[m_N - 1];
    }
    
    __host__ __device__
    inline double* data() {
        return m_ptr;
    }
    
    __host__ __device__
    inline const double* data() const {
        return m_ptr;
    }
    
    __host__ __device__
    size_t size() const { return m_N; }
};


class Array : public ArrayRaw {
public:
    Array() : ArrayRaw(), m_sptr(nullptr) {};
    
    Array(const size_t N) : ArrayRaw(N),
                            m_sptr( std::shared_ptr<double[]>(managed_allocator(N), managed_deleter()) ) {
        this->m_ptr = m_sptr.get();
    }

    Array(Array& array) : ArrayRaw(array.size()), m_sptr(array.get()) {
        this->m_ptr = m_sptr.get();
    }

    Array(Array&& array) : ArrayRaw(array.size()), m_sptr(array.get()) {
        this->m_ptr = m_sptr.get();
    }

    Array& operator=(const Array& other) {
        m_sptr.reset();
        m_sptr = other.m_sptr;
        this->m_ptr = m_sptr.get();
        this->m_N = other.m_N;
        return *this;
    }
    
    std::shared_ptr<double[]> get() { return m_sptr; }

private:
    std::shared_ptr<double[]> m_sptr;
};

} // namespace gmf
