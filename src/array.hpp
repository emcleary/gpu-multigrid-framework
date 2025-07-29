#pragma once

#include <iostream>

#include <cassert>
#include <cstddef> // size_t
#include <memory>

#include <cuda_runtime.h>

#include "memory.hpp"

namespace pmf {

class ArrayRaw {
protected:
    ArrayRaw() : m_N(0) {}
    ArrayRaw(const size_t N) : m_N(N), m_nrows(1), m_ncols(N) {}
    ArrayRaw(const size_t M, const size_t N) : m_N(M * N), m_nrows(M), m_ncols(N) {}
    size_t m_N;
    size_t m_nrows, m_ncols;
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
    inline double& operator()(int i, int j) {
        assert(0 <= i && i < m_nrows);
        assert(0 <= j && j < m_ncols);
        return m_ptr[j + i * m_ncols];
    }

    __host__ __device__
    inline const double& operator()(int i, int j) const {
        assert(0 <= i && i < m_nrows);
        assert(0 <= j && j < m_ncols);
        return m_ptr[j + i * m_ncols];
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
    
    __host__ __device__
    size_t get_nrows() const { return m_nrows; }
    
    __host__ __device__
    size_t get_ncols() const { return m_ncols; }
};


class Array : public ArrayRaw {
public:
    Array() : ArrayRaw(), m_sptr(nullptr) {};

    Array(const size_t N) : ArrayRaw(N),
                            m_sptr( std::shared_ptr<double[]>(managed_allocator(N), managed_deleter()) ) {
        this->m_ptr = m_sptr.get();
    }

    Array(const size_t M, const size_t N) : ArrayRaw(M, N),
                            m_sptr( std::shared_ptr<double[]>(managed_allocator(M * N), managed_deleter()) ) {
        this->m_ptr = m_sptr.get();
    }

    Array(Array& array) : ArrayRaw(array.get_nrows(), array.get_ncols()), m_sptr(array.get()) {
        this->m_ptr = m_sptr.get();
    }

    Array(Array&& array) : ArrayRaw(array.get_nrows(), array.get_ncols()), m_sptr(array.get()) {
        this->m_ptr = m_sptr.get();
    }

    Array& operator=(const Array& other) {
        m_sptr.reset();
        m_sptr = other.m_sptr;
        this->m_ptr = m_sptr.get();
        this->m_N = other.m_N;
        this->m_nrows = other.m_nrows;
        this->m_ncols = other.m_ncols;
        return *this;
    }

    std::shared_ptr<double[]> get() { return m_sptr; }

private:
    std::shared_ptr<double[]> m_sptr;
};

} // namespace pmf
