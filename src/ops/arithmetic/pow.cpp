#include "ops/arithmetic/pow.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

    Tensor pow(const Tensor& base, const Tensor& exponent) {
        autograd::throw_if_requires_grad(base, exponent, "pow");
        if (base.shape() != exponent.shape())
            throw std::runtime_error("Shape mismatch in pow");

        if (base.device_type() != exponent.device_type()) {
            throw std::runtime_error("Device mismatch in pow");
        }

        Tensor out = Tensor::full(base.shape(), 0.0f, base.device_type());
        const Tensor prepared_base = materialize_for_backend_input(base);
        const Tensor prepared_exponent = materialize_for_backend_input(exponent);

        KernelRegistry::instance().getKernel(OpType::Pow, base.device_type())(prepared_base, prepared_exponent, out);

        return out;
    }

    // scalar overloads
    Tensor pow(const Tensor& base, float scalar) {
        return pow(base, Tensor::full(base.shape(), scalar, base.device_type()));
    }

    Tensor pow(float scalar, const Tensor& exponent) {
        return pow(Tensor::full(exponent.shape(), scalar, exponent.device_type()), exponent);
    }

}
