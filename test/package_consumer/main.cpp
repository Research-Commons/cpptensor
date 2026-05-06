#include "cpptensor/tensor/tensor.hpp"

int main() {
    auto tensor = cpptensor::Tensor::zeros({2, 2});
    return tensor.numel() == 4 ? 0 : 1;
}
