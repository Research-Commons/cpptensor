#include "ops/activation/sigmoid.hpp"
#include "dispatcher/kernelRegistry.h"
#include "tensor/tensor.hpp"
#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

    Tensor sigmoid(const Tensor& a) {
        Tensor out = Tensor::full(a.shape(), 0.f , a.device_type());
        const Tensor input = materialize_for_backend_input(a);

        KernelRegistry::instance().getUnaryKernel(OpType::Sigmoid, a.device_type())(input, out);

        return out;
    }

}