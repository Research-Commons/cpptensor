#include "ops/arithmetic/neg.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"


namespace cpptensor {

    Tensor operator-(const Tensor& a) {
        autograd::throw_if_requires_grad(a, "neg");
        Tensor out = Tensor::full(a.shape(), 0.0f, a.device_type());
        KernelRegistry::instance().getUnaryKernel(OpType::Neg, a.device_type())(a, out);
        return out;
    }

}
