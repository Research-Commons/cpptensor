#include "cpptensor/ops/arithmetic/add.hpp"
#include "cpptensor/tensor/tensor.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

namespace {

bool approx_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) < 1e-5f;
}

} // namespace

int main() {
    try {
        cpptensor::Tensor a({2}, {1.0f, 2.0f});
        cpptensor::Tensor b({2}, {3.0f, 4.0f});

        const auto added = a + b;
        if (added.shape() != std::vector<size_t>{2}) {
            std::cerr << "unexpected add shape\n";
            return EXIT_FAILURE;
        }

        if (!approx_equal(added.data()[0], 4.0f) || !approx_equal(added.data()[1], 6.0f)) {
            std::cerr << "unexpected add data\n";
            return EXIT_FAILURE;
        }

        const auto total = added.sum();
        if (total.shape() != std::vector<size_t>{}) {
            std::cerr << "unexpected sum shape\n";
            return EXIT_FAILURE;
        }

        if (!approx_equal(total.data()[0], 10.0f)) {
            std::cerr << "unexpected sum data\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
