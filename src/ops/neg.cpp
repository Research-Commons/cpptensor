#include "ops/neg.hpp"
#include "autograd/function.hpp"
#include "tensor/tensor.hpp"


namespace cpptensor {

    Tensor operator-(const Tensor& a) {
        std::vector<float> neg_data(a.data().size());
        const auto& src = a.data();

        for (size_t i = 0; i < src.size(); ++i)
            neg_data[i] = -src[i];

        Tensor out(a.shape(), neg_data, a.requires_grad(), a.device_type());

        // if (out.requires_grad() && out.impl_->grad_fn() == nullptr) {
        //     auto fn = std::make_shared<NegFunction>();
        //     fn->inputs = { a.impl_ };
        //     out.impl_->grad_fn() = fn;
        // }

        return out;
    }

}