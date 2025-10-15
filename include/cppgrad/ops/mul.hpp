#pragma once

namespace cppgrad {
    class Tensor;

    Tensor operator*(const Tensor& a, const Tensor& b);
    Tensor operator*(const Tensor& lhs, float scalar);
    Tensor operator*(float scalar, const Tensor& rhs);

}