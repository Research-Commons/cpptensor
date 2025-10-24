#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"

#include <stdexcept>
#include <vector>
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
namespace cpptensor {

    Tensor matmul(const Tensor& A, const Tensor& B) {
        if (A.device_type() != B.device_type()) {
            throw std::runtime_error("matmul: device type mismatch");
        }

        if (A.shape().size() != 2 || B.shape().size() != 2) {
            throw std::runtime_error("matmul: only supports 2D matrices");
        }

        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t KB = B.shape()[0];
        size_t N = B.shape()[1];

        if (K != KB) {
            throw std::runtime_error("matmul: dimension mismatch (A.cols != B.rows)");
        }

        Tensor C = Tensor::full({M, N}, 0.0f, A.device_type());

#ifdef USE_OPENBLAS
        // ===== Use OpenBLAS SGEMM =====
        //
        // SGEMM performs: C = alpha * A * B + beta * C
        // A: MxK
        // B: KxN
        // C: MxN
        //
        // We assume row-major layout for all tensors.
        //
        // CBLAS expects leading dimensions (lda, ldb, ldc) to match the
        // number of columns in each matrix for row-major ordering.

        //openblas_set_num_threads(1);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const float* Adata = A.data().data();
        const float* Bdata = B.data().data();
        float* Cdata = C.data().data();

        cblas_sgemm(
            CblasRowMajor,    // row-major storage
            CblasNoTrans,     // A not transposed
            CblasNoTrans,     // B not transposed
            static_cast<int>(M), // rows of A and C
            static_cast<int>(N), // cols of B and C
            static_cast<int>(K), // shared dimension
            alpha,             // scaling for A * B
            Adata, static_cast<int>(K), // A, leading dimension = K
            Bdata, static_cast<int>(N), // B, leading dimension = N
            beta,              // scaling for existing C
            Cdata, static_cast<int>(N)  // C, leading dimension = N
        );
#else
        KernelRegistry::instance().getKernel(OpType::Matmul, A.device_type())(A, B, C);
#endif
        return C;
    }

} // namespace cpptensor