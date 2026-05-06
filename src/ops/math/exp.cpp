#include "ops/math/exp.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"
#include "cpptensor/ops/helperOps.hpp"
#include "cpptensor/tensor/autograd_utils.hpp"

namespace cpptensor {

    Tensor exp(const Tensor& a) {
        autograd::throw_if_requires_grad(a, "exp");
        Tensor out = Tensor::full(a.shape(), 0.f , a.device_type());
        const Tensor input = materialize_for_backend_input(a);

        KernelRegistry::instance().getUnaryKernel(OpType::Exp, a.device_type())(input, out);

        return out;
    }

}
