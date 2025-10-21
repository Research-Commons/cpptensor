#include "ops/abs.hpp"
#include "autograd/function.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"

namespace cpptensor {

    Tensor abs(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.requires_grad());

        // if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
        //     auto fn = std::make_shared<ExpFunction>();
        //     fn->inputs = { a.impl_ };
        //     out.impl_->grad_fn() = fn;
        // }

        KernelRegistry::instance().getUnaryKernel(OpType::Abs, a.device_type())(a, out);

        return out;
    }

}