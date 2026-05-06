#include "ops/arithmetic/pow.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"
#include "cpptensor/tensor/dtype_utils.hpp"

#include <string>

namespace cpptensor {

Tensor pow(const Tensor& base, const Tensor& exponent) {
    autograd::throw_if_requires_grad(base, exponent, "pow");
    if (base.shape() != exponent.shape()) {
        throw std::runtime_error("Shape mismatch in pow");
    }

    if (base.device_type() != exponent.device_type()) {
        throw std::runtime_error("Device mismatch in pow");
    }

    const DType promoted = promote_dtype(base.dtype(), exponent.dtype());
    if (promoted != DType::FLOAT32) {
        throw std::runtime_error(
            "pow currently supports float32 compute only; got base dtype " +
            std::string(dtype_name(base.dtype())) + " and exponent dtype " +
            std::string(dtype_name(exponent.dtype())));
    }

    Tensor out = Tensor::full(base.shape(), 0.0f, base.device_type(), DType::FLOAT32);
    const Tensor prepared_base = materialize_for_backend_input(base);
    const Tensor prepared_exponent = materialize_for_backend_input(exponent);

    KernelRegistry::instance().getKernel(OpType::Pow, base.device_type())(prepared_base, prepared_exponent, out);

    return out;
}

Tensor pow(const Tensor& base, float scalar) {
    return pow(base, Tensor::full(base.shape(), scalar, base.device_type(), base.dtype()));
}

Tensor pow(float scalar, const Tensor& exponent) {
    return pow(Tensor::full(exponent.shape(), scalar, exponent.device_type(), exponent.dtype()), exponent);
}

} // namespace cpptensor
