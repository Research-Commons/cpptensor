#pragma once

#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor {
    class CPU {
    public:
        static void addKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void mulKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void subKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void divKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void negKernel(const Tensor& A, Tensor& Out);
        static void powKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void expKernel(const Tensor& A, Tensor& Out);
        static void logKernel(const Tensor& A, Tensor& Out);
        static void absKernel(const Tensor& A, Tensor& Out);
        static void sqrtKernel(const Tensor& A, Tensor& Out);
        static void sinKernel(const Tensor& A, Tensor& Out);
        static void cosKernel(const Tensor& A, Tensor& Out);
        static void tanKernel(const Tensor& A, Tensor& Out);
        static void sigmoidKernel(const Tensor& A, Tensor& Out);
        static void reluKernel(const Tensor& A, Tensor& Out);
        static void gemmf32kernel(const Tensor& A, const Tensor& B, Tensor& Out);
        static void dotKernel(const Tensor &A, const Tensor &B, Tensor & Out);

        // Reduction operations
        static void sumKernel(const Tensor& input, Tensor& output, int dim, bool keepdim);
        static void meanKernel(const Tensor& input, Tensor& output, int dim, bool keepdim);
        static void maxKernel(const Tensor& input, Tensor& output, int dim, bool keepdim);
        static void minKernel(const Tensor& input, Tensor& output, int dim, bool keepdim);

        // Comparison operations
        static void eqKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void neKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void gtKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void ltKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void geKernel(const Tensor& A, const Tensor& B, Tensor& out);
        static void leKernel(const Tensor& A, const Tensor& B, Tensor& out);
    };
}
