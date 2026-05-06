#include "cpptensor/ops/linearAlgebra/dot.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"
#include <cmath>
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
            Tensor A_blas = A.is_contiguous() ? A : A.contiguous();
            Tensor B_blas = B.is_contiguous() ? B : B.contiguous();

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

            const float* Adata = A_blas.impl()->data_ptr();
            const float* Bdata = B_blas.impl()->data_ptr();

            // Stability-first accumulation: this avoids catastrophic cancellation
            // seen with single-precision accumulation on adversarial inputs.
            double sum = 0.0;
            double compensation = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double value = static_cast<double>(Adata[i]) * static_cast<double>(Bdata[i]);
                const double t = sum + value;
                if (std::abs(sum) >= std::abs(value)) {
                    compensation += (sum - t) + value;
                } else {
                    compensation += (value - t) + sum;
                }
                sum = t;
            }

            Out.data().data()[0] = static_cast<float>(sum + compensation);
    #else
            KernelRegistry::instance().getKernel(OpType::Dot, A.device_type())(A, B, Out);
    #endif
            return Out;
    }

} // namespace cpptensor
