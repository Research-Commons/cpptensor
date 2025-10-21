#include "ops/arithmetic/neg.hpp"
#include "tensor/tensor.hpp"


namespace cpptensor {

    Tensor operator-(const Tensor& a) {
        std::vector<float> neg_data(a.data().size());
        const auto& src = a.data();

        for (size_t i = 0; i < src.size(); ++i)
            neg_data[i] = -src[i];

        Tensor out(a.shape(), neg_data, a.device_type());

        return out;
    }

}