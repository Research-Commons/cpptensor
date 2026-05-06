#include "cpptensor/ops/linearAlgebra/dot.hpp"
#include "cpptensor/dispatcher/kernelRegistry.h"
#include "cpptensor/enums/dispatcherEnum.h"
#include "cpptensor/ops/helperOps.hpp"
#include <stdexcept>

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

        Tensor Out = Tensor::full({}, 0.0f, A.device_type());

            const Tensor prepared_a = materialize_for_backend_input(A);
            const Tensor prepared_b = materialize_for_backend_input(B);
            KernelRegistry::instance().getKernel(OpType::Dot, A.device_type())(prepared_a, prepared_b, Out);
            return Out;
    }

} // namespace cpptensor
