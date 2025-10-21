#include "ops/pow.hpp"
#include "autograd/function.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor pow(const Tensor& base, const Tensor& exponent) {
        if (base.shape() != exponent.shape())
            throw std::runtime_error("Shape mismatch in pow");

        if (base.device_type() != exponent.device_type()) {
            throw std::runtime_error("Device mismatch in pow");
        }

        Tensor out = Tensor::full(base.shape(), 0.0f, base.requires_grad() || exponent.requires_grad(), base.device_type());

        // if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
        //     auto fn = std::make_shared<PowFunction>();
        //     fn->inputs = { base.impl_, exponent.impl_ };
        //     out.impl_->grad_fn() = fn;
        // }

        KernelRegistry::instance().getKernel(OpType::Pow, base.device_type())(base, exponent, out);

        return out;
    }

    // scalar overloads
    Tensor pow(const Tensor& base, float scalar) {
        return pow(base, Tensor::full(base.shape(), scalar, false));
    }

    Tensor pow(float scalar, const Tensor& exponent) {
        return pow(Tensor::full(exponent.shape(), scalar, false), exponent);
    }

}