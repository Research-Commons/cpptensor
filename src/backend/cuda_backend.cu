#include "cpptensor/backend/cuda_backend.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h" // adjust include path for KernelRegistry/OpType/DeviceType
#include <vector>
#include <iostream>
#include <stdexcept>
#include <optional>
#include <cuda.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do {                                \
    cudaError_t err = (call);                                \
    if (err != cudaSuccess) {                                \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA call failed");        \
    }                                                        \
} while(0)

extern "C" __global__
void add_kernel_broadcast(const float* a, const float* b, float* out,
                          size_t total,
                          const size_t* strideA_eff,
                          const size_t* strideB_eff,
                          const size_t* strideOut,
                          const size_t* out_sh,
                          int n_dims) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;

    size_t idxA = 0;
    size_t idxB = 0;
    // map flat pos -> coordinates and then to input indices using effective strides
    for (int d = 0; d < n_dims; ++d) {
        size_t coord = (pos / strideOut[d]) % out_sh[d];
        idxA += coord * strideA_eff[d];
        idxB += coord * strideB_eff[d];
    }
    out[pos] = a[idxA] + b[idxB];
}

extern "C" __global__
void mul_kernel_broadcast(const float* a, const float* b, float* out,
                          size_t total,
                          const size_t* strideA_eff,
                          const size_t* strideB_eff,
                          const size_t* strideOut,
                          const size_t* out_sh,
                          int n_dims) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total) return;

    size_t idxA = 0;
    size_t idxB = 0;
    // map flat pos -> coordinates and then to input indices using effective strides
    for (int d = 0; d < n_dims; ++d) {
        size_t coord = (pos / strideOut[d]) % out_sh[d];
        idxA += coord * strideA_eff[d];
        idxB += coord * strideB_eff[d];
    }
    out[pos] = a[idxA] * b[idxB];
}


namespace cpptensor {

    void CUDA::addKernel(const Tensor& A, const Tensor& B, Tensor& out) {
        if (!out.is_contiguous()) {
            throw std::runtime_error(
                "CUDA::addKernel currently requires contiguous output tensors");
        }

        // Temporary contiguous fallback for non-contiguous CUDA views.
        // CUDA add/mul kernels below currently assume compact row-major inputs.
        std::optional<Tensor> a_compact;
        std::optional<Tensor> b_compact;
        const Tensor& a_tensor = A.is_contiguous() ? A : (a_compact = A.contiguous(), *a_compact);
        const Tensor& b_tensor = B.is_contiguous() ? B : (b_compact = B.contiguous(), *b_compact);

        // Prepare shapes and strides (same logic as CPU)
        const auto& a_sh = a_tensor.shape();
        const auto& b_sh = b_tensor.shape();
        const auto& out_sh = out.shape();
        int n = static_cast<int>(out_sh.size());

        // Compute padded shapes (align right)
        std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
        size_t na = a_sh.size(), nb = b_sh.size();
        for (int i = 0; i < n; ++i) {
            a_pad[i] = (i < n - (int)na ? 1 : a_sh[i - (n - (int)na)]);
            b_pad[i] = (i < n - (int)nb ? 1 : b_sh[i - (n - (int)nb)]);
        }

        // Compute strides for A,B,out (C-order)
        std::vector<size_t> strideA(n), strideB(n), strideOut(n);
        if (n > 0) {
            strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
        }
        for (int i = n - 2; i >= 0; --i) {
            strideA[i] = strideA[i+1] * a_pad[i+1];
            strideB[i] = strideB[i+1] * b_pad[i+1];
            strideOut[i] = strideOut[i+1] * out_sh[i+1];
        }

        // Effective strides: zero for broadcasted dimensions
        std::vector<size_t> strideA_eff(n), strideB_eff(n);
        for (int i = 0; i < n; ++i) {
            strideA_eff[i] = (a_pad[i] == 1) ? 0 : strideA[i];
            strideB_eff[i] = (b_pad[i] == 1) ? 0 : strideB[i];
        }

        // Total number of output elements
        size_t total = 1;
        for (size_t d : out_sh) total *= d;
        if (total == 0) return;

        const float* d_a = a_tensor.impl()->backend_data_ptr(DeviceType::CUDA);
        const float* d_b = b_tensor.impl()->backend_data_ptr(DeviceType::CUDA);
        float* d_out = out.impl()->backend_data_ptr(DeviceType::CUDA);
        size_t *d_strideA_eff = nullptr, *d_strideB_eff = nullptr, *d_strideOut = nullptr, *d_out_sh = nullptr;

        // copy stride arrays and output shape/strides
        CUDA_CHECK(cudaMalloc(&d_strideA_eff, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_strideB_eff, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_strideOut, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_out_sh, n * sizeof(size_t)));

        CUDA_CHECK(cudaMemcpy(d_strideA_eff, strideA_eff.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strideB_eff, strideB_eff.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strideOut, strideOut.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_sh, out_sh.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));

        // Launch kernel
        const int block = 256;
        int grid = static_cast<int>((total + block - 1) / block);
        add_kernel_broadcast<<<grid, block>>>(d_a, d_b, d_out, total,
                                              d_strideA_eff, d_strideB_eff, d_strideOut, d_out_sh, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free device memory
        CUDA_CHECK(cudaFree(d_strideA_eff));
        CUDA_CHECK(cudaFree(d_strideB_eff));
        CUDA_CHECK(cudaFree(d_strideOut));
        CUDA_CHECK(cudaFree(d_out_sh));
    }

    void CUDA::mulKernel(const Tensor& A, const Tensor& B, Tensor& out) {
        if (!out.is_contiguous()) {
            throw std::runtime_error(
                "CUDA::mulKernel currently requires contiguous output tensors");
        }

        // Temporary contiguous fallback for non-contiguous CUDA views.
        // CUDA add/mul kernels below currently assume compact row-major inputs.
        std::optional<Tensor> a_compact;
        std::optional<Tensor> b_compact;
        const Tensor& a_tensor = A.is_contiguous() ? A : (a_compact = A.contiguous(), *a_compact);
        const Tensor& b_tensor = B.is_contiguous() ? B : (b_compact = B.contiguous(), *b_compact);

        // Prepare shapes and strides (same logic as CPU)
        const auto& a_sh = a_tensor.shape();
        const auto& b_sh = b_tensor.shape();
        const auto& out_sh = out.shape();
        int n = static_cast<int>(out_sh.size());

        // Compute padded shapes (align right)
        std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
        size_t na = a_sh.size(), nb = b_sh.size();
        for (int i = 0; i < n; ++i) {
            a_pad[i] = (i < n - (int)na ? 1 : a_sh[i - (n - (int)na)]);
            b_pad[i] = (i < n - (int)nb ? 1 : b_sh[i - (n - (int)nb)]);
        }

        // Compute strides for A,B,out (C-order)
        std::vector<size_t> strideA(n), strideB(n), strideOut(n);
        if (n > 0) {
            strideA[n-1] = strideB[n-1] = strideOut[n-1] = 1;
        }
        for (int i = n - 2; i >= 0; --i) {
            strideA[i] = strideA[i+1] * a_pad[i+1];
            strideB[i] = strideB[i+1] * b_pad[i+1];
            strideOut[i] = strideOut[i+1] * out_sh[i+1];
        }

        // Effective strides: zero for broadcasted dimensions
        std::vector<size_t> strideA_eff(n), strideB_eff(n);
        for (int i = 0; i < n; ++i) {
            strideA_eff[i] = (a_pad[i] == 1) ? 0 : strideA[i];
            strideB_eff[i] = (b_pad[i] == 1) ? 0 : strideB[i];
        }

        // Total number of output elements
        size_t total = 1;
        for (size_t d : out_sh) total *= d;
        if (total == 0) return;

        const float* d_a = a_tensor.impl()->backend_data_ptr(DeviceType::CUDA);
        const float* d_b = b_tensor.impl()->backend_data_ptr(DeviceType::CUDA);
        float* d_out = out.impl()->backend_data_ptr(DeviceType::CUDA);
        size_t *d_strideA_eff = nullptr, *d_strideB_eff = nullptr, *d_strideOut = nullptr, *d_out_sh = nullptr;

        // copy stride arrays and output shape/strides
        CUDA_CHECK(cudaMalloc(&d_strideA_eff, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_strideB_eff, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_strideOut, n * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_out_sh, n * sizeof(size_t)));

        CUDA_CHECK(cudaMemcpy(d_strideA_eff, strideA_eff.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strideB_eff, strideB_eff.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strideOut, strideOut.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_sh, out_sh.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));

        // Launch kernel
        const int block = 256;
        int grid = static_cast<int>((total + block - 1) / block);
        mul_kernel_broadcast<<<grid, block>>>(d_a, d_b, d_out, total,
                                              d_strideA_eff, d_strideB_eff, d_strideOut, d_out_sh, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free device memory
        CUDA_CHECK(cudaFree(d_strideA_eff));
        CUDA_CHECK(cudaFree(d_strideB_eff));
        CUDA_CHECK(cudaFree(d_strideOut));
        CUDA_CHECK(cudaFree(d_out_sh));
    }
} // namespace cpptensor
