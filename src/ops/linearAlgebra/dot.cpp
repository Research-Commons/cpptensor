#include "cpptensor/ops/linearAlgebra/dot.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"
#include <stdexcept>

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

namespace cpptensor {

    Tensor dot(const Tensor& A, const Tensor& B) {
        if (A.device_type() != B.device_type()) {
            throw std::runtime_error("dot: device mismatch");
        }

        const auto& Ash = A.shape();
        const auto& Bsh = B.shape();
        if (Ash.size() != 1 || Bsh.size() != 1) {
            throw std::runtime_error("dot: inputs must be 1D tensors (vectors)");
        }
        if (Ash[0] != Bsh[0]) {
            throw std::runtime_error("dot: size mismatch");
        }

        const size_t n = Ash[0];

        Tensor Out = Tensor::full({}, 0.0f, A.device_type());

    #ifdef USE_OPENBLAS
            // ===== Use OpenBLAS SDOT =====
            //
            // SDOT computes the dot product of two vectors:
            // result = sum(A[i] * B[i]) for i = 0..n-1
            //
            // Parameters:
            // - n: number of elements
            // - x: pointer to first vector
            // - incx: stride within x (1 for contiguous)
            // - y: pointer to second vector
            // - incy: stride within y (1 for contiguous)

            const float* Adata = A.data().data();
            const float* Bdata = B.data().data();

            float result = cblas_sdot(
                static_cast<int>(n),  // number of elements
                Adata, 1,             // vector A, stride 1
                Bdata, 1              // vector B, stride 1
            );

            Out.data().data()[0] = result;
    #else
            KernelRegistry::instance().getKernel(OpType::Dot, A.device_type())(A, B, Out);
    #endif

        KernelRegistry::instance().getKernel(OpType::Dot, A.device_type())(A, B, Out);
        return Out;
    }

} // namespace cpptensor
