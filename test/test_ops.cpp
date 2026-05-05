#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cpptensor/backend/cpu_backend.h"
#include "cpptensor/backend/isa/isaDetect.hpp"
#ifdef BUILD_AVX2
#include "cpptensor/backend/isa/avx2.hpp"
#endif
#include "cpptensor/ops/arithmetic/pow.hpp"
#include "cpptensor/ops/comparison/eq.hpp"
#include "cpptensor/ops/comparison/ge.hpp"
#include "cpptensor/ops/comparison/gt.hpp"
#include "cpptensor/ops/comparison/le.hpp"
#include "cpptensor/ops/comparison/lt.hpp"
#include "cpptensor/ops/comparison/ne.hpp"
#include "cpptensor/ops/linearAlgebra/tensordot.hpp"
#include "cpptensor/ops/manipulation/cat.hpp"
#include "cpptensor/ops/manipulation/stack.hpp"
#include "cpptensor/ops/math/matmul.hpp"
#include "cpptensor/backend/backend_loader.hpp"
#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/utils/broadcastUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using Catch::Approx;

namespace {

void require_shape(const cpptensor::Tensor& tensor, std::vector<size_t> expected) {
    REQUIRE(tensor.shape() == expected);
}

void require_data(const cpptensor::Tensor& tensor, const std::vector<float>& expected) {
    REQUIRE(tensor.data().size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(tensor.data()[i] == Approx(expected[i]));
    }
}

std::vector<int> normalize_axes(std::vector<int> axes, size_t rank) {
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += static_cast<int>(rank);
        }
    }
    return axes;
}

std::vector<int> complement_axes(size_t rank, const std::vector<int>& axes) {
    std::vector<bool> contracted(rank, false);
    for (int axis : axes) {
        contracted[static_cast<size_t>(axis)] = true;
    }

    std::vector<int> result;
    result.reserve(rank - axes.size());
    for (size_t axis = 0; axis < rank; ++axis) {
        if (!contracted[axis]) {
            result.push_back(static_cast<int>(axis));
        }
    }
    return result;
}

std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 0);
    if (shape.empty()) {
        return strides;
    }

    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i) + 1] * shape[static_cast<size_t>(i) + 1];
    }
    return strides;
}

std::vector<size_t> decode_index(size_t flat, const std::vector<size_t>& shape) {
    std::vector<size_t> index(shape.size(), 0);
    if (shape.empty()) {
        return index;
    }

    auto strides = compute_strides(shape);
    for (size_t dim = 0; dim < shape.size(); ++dim) {
        if (shape[dim] == 0) {
            index[dim] = 0;
            continue;
        }
        index[dim] = flat / strides[dim];
        flat %= strides[dim];
    }
    return index;
}

size_t encode_index(const std::vector<size_t>& index, const std::vector<size_t>& strides) {
    size_t offset = 0;
    for (size_t dim = 0; dim < index.size(); ++dim) {
        offset += index[dim] * strides[dim];
    }
    return offset;
}

cpptensor::Tensor naive_tensordot(const cpptensor::Tensor& A,
                                  const cpptensor::Tensor& B,
                                  const std::vector<int>& raw_axesA,
                                  const std::vector<int>& raw_axesB) {
    auto axesA = normalize_axes(raw_axesA, A.ndim());
    auto axesB = normalize_axes(raw_axesB, B.ndim());

    auto A_rest = complement_axes(A.ndim(), axesA);
    auto B_rest = complement_axes(B.ndim(), axesB);

    std::vector<size_t> contract_shape;
    contract_shape.reserve(axesA.size());
    for (size_t i = 0; i < axesA.size(); ++i) {
        const size_t dimA = A.shape()[static_cast<size_t>(axesA[i])];
        const size_t dimB = B.shape()[static_cast<size_t>(axesB[i])];
        REQUIRE(dimA == dimB);
        contract_shape.push_back(dimA);
    }

    std::vector<size_t> out_shape;
    out_shape.reserve(A_rest.size() + B_rest.size());
    for (int axis : A_rest) {
        out_shape.push_back(A.shape()[static_cast<size_t>(axis)]);
    }
    for (int axis : B_rest) {
        out_shape.push_back(B.shape()[static_cast<size_t>(axis)]);
    }

    const auto A_strides = compute_strides(A.shape());
    const auto B_strides = compute_strides(B.shape());
    const size_t out_numel = out_shape.empty()
        ? 1
        : std::accumulate(out_shape.begin(), out_shape.end(), size_t{1}, std::multiplies<size_t>());
    const size_t contract_numel = contract_shape.empty()
        ? 1
        : std::accumulate(contract_shape.begin(), contract_shape.end(), size_t{1}, std::multiplies<size_t>());

    std::vector<float> out_data(out_numel, 0.0f);
    std::vector<size_t> a_index(A.ndim(), 0);
    std::vector<size_t> b_index(B.ndim(), 0);

    for (size_t out_flat = 0; out_flat < out_numel; ++out_flat) {
        const auto out_index = decode_index(out_flat, out_shape);
        std::fill(a_index.begin(), a_index.end(), 0);
        std::fill(b_index.begin(), b_index.end(), 0);

        size_t out_pos = 0;
        for (int axis : A_rest) {
            a_index[static_cast<size_t>(axis)] = out_index[out_pos++];
        }
        for (int axis : B_rest) {
            b_index[static_cast<size_t>(axis)] = out_index[out_pos++];
        }

        float sum = 0.0f;
        for (size_t contract_flat = 0; contract_flat < contract_numel; ++contract_flat) {
            const auto contract_index = decode_index(contract_flat, contract_shape);
            for (size_t contract_dim = 0; contract_dim < contract_shape.size(); ++contract_dim) {
                a_index[static_cast<size_t>(axesA[contract_dim])] = contract_index[contract_dim];
                b_index[static_cast<size_t>(axesB[contract_dim])] = contract_index[contract_dim];
            }

            sum += A.data()[encode_index(a_index, A_strides)] * B.data()[encode_index(b_index, B_strides)];
        }

        out_data[out_flat] = sum;
    }

    return cpptensor::Tensor(out_shape, out_data);
}

} // namespace

class ScopedCpuIsaOverride {
public:
    explicit ScopedCpuIsaOverride(const char* value) : had_previous_(std::getenv("CPPGRAD_CPU_ISA") != nullptr) {
        if (had_previous_) {
            previous_ = std::getenv("CPPGRAD_CPU_ISA");
        }
        set(value);
    }

    ~ScopedCpuIsaOverride() {
        if (had_previous_) {
            set(previous_.c_str());
        } else {
            unset();
        }
    }

private:
    static void set(const char* value) {
#ifdef _WIN32
        _putenv_s("CPPGRAD_CPU_ISA", value == nullptr ? "" : value);
#else
        if (value == nullptr) {
            unset();
        } else {
            setenv("CPPGRAD_CPU_ISA", value, 1);
        }
#endif
    }

    static void unset() {
#ifdef _WIN32
        _putenv_s("CPPGRAD_CPU_ISA", "");
#else
        unsetenv("CPPGRAD_CPU_ISA");
#endif
    }

    bool had_previous_;
    std::string previous_;
};

void require_nan_data(const cpptensor::Tensor& tensor) {
    for (float value : tensor.data()) {
        REQUIRE(std::isnan(value));
    }
}

TEST_CASE("cat concatenates tensors along existing dimensions", "[manipulation][cat]") {
    cpptensor::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor b({2, 3}, {7, 8, 9, 10, 11, 12});

    auto dim0 = cpptensor::cat({a, b}, 0);
    require_shape(dim0, {4, 3});
    require_data(dim0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    auto dim1 = cpptensor::cat({a, b}, 1);
    require_shape(dim1, {2, 6});
    require_data(dim1, {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12});

    auto neg_dim = cpptensor::cat({a, b}, -1);
    require_shape(neg_dim, {2, 6});
    require_data(neg_dim, dim1.data());
}

TEST_CASE("cat preserves device placement and rejects mixed-device inputs",
          "[manipulation][cat][device]") {
    cpptensor::Tensor cuda_a({2}, {1, 2}, DeviceType::CUDA);
    cpptensor::Tensor cuda_b({2}, {3, 4}, DeviceType::CUDA);

    auto cuda_result = cpptensor::cat({cuda_a, cuda_b}, 0);
    require_shape(cuda_result, {4});
    require_data(cuda_result, {1, 2, 3, 4});
    REQUIRE(cuda_result.device_type() == DeviceType::CUDA);

    cpptensor::Tensor cpu({2}, {5, 6}, DeviceType::CPU);
    REQUIRE_THROWS_WITH(cpptensor::cat({cuda_a, cpu}, 0),
                        Catch::Matchers::ContainsSubstring("same device"));
}

TEST_CASE("stack inserts a new dimension", "[manipulation][stack]") {
    cpptensor::Tensor a({2, 2}, {1, 2, 3, 4});
    cpptensor::Tensor b({2, 2}, {5, 6, 7, 8});

    auto dim0 = cpptensor::stack({a, b}, 0);
    require_shape(dim0, {2, 2, 2});
    require_data(dim0, {1, 2, 3, 4, 5, 6, 7, 8});

    auto dim1 = cpptensor::stack({a, b}, 1);
    require_shape(dim1, {2, 2, 2});
    require_data(dim1, {1, 2, 5, 6, 3, 4, 7, 8});

    auto dim2 = cpptensor::stack({a, b}, 2);
    require_shape(dim2, {2, 2, 2});
    require_data(dim2, {1, 5, 2, 6, 3, 7, 4, 8});
}

TEST_CASE("stack preserves device placement and rejects mixed-device inputs",
          "[manipulation][stack][device]") {
    cpptensor::Tensor cuda_a({2}, {1, 2}, DeviceType::CUDA);
    cpptensor::Tensor cuda_b({2}, {3, 4}, DeviceType::CUDA);

    auto cuda_result = cpptensor::stack({cuda_a, cuda_b}, 0);
    require_shape(cuda_result, {2, 2});
    require_data(cuda_result, {1, 2, 3, 4});
    REQUIRE(cuda_result.device_type() == DeviceType::CUDA);

    cpptensor::Tensor cpu({2}, {5, 6}, DeviceType::CPU);
    REQUIRE_THROWS_WITH(cpptensor::stack({cuda_a, cpu}, 0),
                        Catch::Matchers::ContainsSubstring("same device"));
}

TEST_CASE("squeeze can reduce singleton tensors to scalars", "[manipulation][squeeze]") {
    cpptensor::Tensor singleton({1}, std::vector<float>{42});
    auto scalar = singleton.squeeze();
    require_shape(scalar, std::vector<size_t>{});
    REQUIRE(scalar.ndim() == 0);
    require_data(scalar, {42});

    cpptensor::Tensor all_singletons({1, 1, 1}, std::vector<float>{7});
    auto squeezed = all_singletons.squeeze();
    require_shape(squeezed, std::vector<size_t>{});
    REQUIRE(squeezed.ndim() == 0);
    require_data(squeezed, {7});

    auto specific_dim = all_singletons.squeeze(1);
    require_shape(specific_dim, {1, 1});
    REQUIRE(specific_dim.ndim() == 2);
    require_data(specific_dim, {7});
}

TEST_CASE("comparison operators support tensor, scalar, and broadcast operands", "[comparison]") {
    cpptensor::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor b({2, 3}, {1, 0, 3, 10, 5, 0});

    require_data(a == b, {1, 0, 1, 0, 1, 0});
    require_data(a != b, {0, 1, 0, 1, 0, 1});
    require_data(a > b, {0, 1, 0, 0, 0, 1});
    require_data(a < b, {0, 0, 0, 1, 0, 0});
    require_data(a >= b, {1, 1, 1, 0, 1, 1});
    require_data(a <= b, {1, 0, 1, 1, 1, 0});

    require_data(a > 3.0f, {0, 0, 0, 1, 1, 1});
    require_data(3.0f < a, {0, 0, 0, 1, 1, 1});

    cpptensor::Tensor row({1, 3}, {1, 5, 10});
    require_shape(a < row, {2, 3});
    require_data(a < row, {0, 1, 1, 0, 0, 1});
}

TEST_CASE("compute_broadcast_shape rejects incompatible dimensions and preserves valid broadcasts",
          "[broadcast]") {
    REQUIRE_THROWS_WITH(cpptensor::compute_broadcast_shape({2}, {3}),
                        Catch::Matchers::ContainsSubstring("incompatible dimensions 2 and 3"));

    REQUIRE(cpptensor::compute_broadcast_shape({2, 1, 4}, {1, 3, 4}) ==
            std::vector<size_t>{2, 3, 4});
}

TEST_CASE("gemv and matmul produce the same matrix-vector result", "[matmul][gemv]") {
    cpptensor::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    cpptensor::Tensor x({3}, {1, 0, -1});

    require_shape(cpptensor::gemv(a, x), {2});
    require_data(cpptensor::gemv(a, x), {-2, -2});

    auto y = cpptensor::matmul(a, x);
    require_shape(y, {2});
    require_data(y, {-2, -2});
}

TEST_CASE("reductions handle global and dimension-specific forms", "[reduction]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

    auto global_sum = t.sum();
    require_shape(global_sum, {});
    REQUIRE(global_sum.ndim() == 0);
    require_data(global_sum, {21});

    auto global_mean = t.mean();
    require_shape(global_mean, {});
    REQUIRE(global_mean.ndim() == 0);
    require_data(global_mean, {3.5f});

    auto global_max = t.max();
    require_shape(global_max, {});
    REQUIRE(global_max.ndim() == 0);
    require_data(global_max, {6});

    auto global_min = t.min();
    require_shape(global_min, {});
    REQUIRE(global_min.ndim() == 0);
    require_data(global_min, {1});

    require_shape(t.sum(true), {1, 1});
    require_data(t.sum(true), {21});
    require_shape(t.mean(true), {1, 1});
    require_data(t.mean(true), {3.5f});
    require_shape(t.max(true), {1, 1});
    require_data(t.max(true), {6});
    require_shape(t.min(true), {1, 1});
    require_data(t.min(true), {1});

    cpptensor::Tensor v({3}, {1, 2, 3});
    require_shape(v.sum(0), {});
    require_data(v.sum(0), {6});
    require_shape(v.mean(0), {});
    require_data(v.mean(0), {2});
    require_shape(v.max(0), {});
    require_data(v.max(0), {3});
    require_shape(v.min(0), {});
    require_data(v.min(0), {1});

    require_shape(t.sum(0), {3});
    require_data(t.sum(0), {5, 7, 9});
    require_shape(t.sum(1, true), {2, 1});
    require_data(t.sum(1, true), {6, 15});

    require_shape(t.mean(1), {2});
    require_data(t.mean(1), {2, 5});

    require_shape(t.max(0), {3});
    require_data(t.max(0), {4, 5, 6});
    require_shape(t.min(-1), {2});
    require_data(t.min(-1), {1, 4});
}

TEST_CASE("contiguous materializes view values from the logical view start", "[tensor][contiguous]") {
    cpptensor::Tensor base({4}, {0, 1, 2, 3});

    auto stepped = base.slice(0, 1, 4, 2);
    auto materialized = stepped.contiguous();

    REQUIRE_FALSE(stepped.is_contiguous());
    require_shape(materialized, {2});
    require_data(materialized, {1, 3});
}

TEST_CASE("contiguous honors raw-pointer-backed view offsets", "[tensor][contiguous][from_ptr]") {
    cpptensor::Tensor owner({6}, {0, 1, 2, 3, 4, 5});
    auto subrange = cpptensor::Tensor::from_ptr(
        {4},
        owner.data().data() + 1,
        owner.impl(),
        owner.device_type());

    auto stepped = subrange.slice(0, 0, 4, 2);
    auto materialized = stepped.contiguous();

    REQUIRE_FALSE(stepped.is_contiguous());
    require_shape(materialized, {2});
    require_data(materialized, {1, 3});
}

TEST_CASE("pointer-backed views expose logical const data and reject mutable storage",
          "[tensor][data][from_ptr]") {
    cpptensor::Tensor owner({6}, {0, 1, 2, 3, 4, 5});
    auto subrange = cpptensor::Tensor::from_ptr(
        {4},
        owner.data().data() + 1,
        owner.impl(),
        owner.device_type());

    require_shape(subrange, {4});
    require_data(subrange, {1, 2, 3, 4});
    REQUIRE_THROWS_WITH(
        subrange.data(),
        Catch::Matchers::ContainsSubstring("pointer-backed views"));
}

TEST_CASE("clone deep-copies sliced views using the logical view contents", "[tensor][clone][view]") {
    cpptensor::Tensor base({4}, {0, 1, 2, 3});

    auto sliced = base.slice(0, 1, 3);
    auto cloned = sliced.clone();

    require_shape(cloned, {2});
    require_data(cloned, {1, 2});

    base.data()[1] = 99.0f;
    require_data(cloned, {1, 2});
}

TEST_CASE("clone preserves the logical order of transposed views", "[tensor][clone][view]") {
    cpptensor::Tensor matrix({2, 2}, {1, 2, 3, 4});

    auto transposed = matrix.transpose(0, 1);
    auto cloned = transposed.clone();

    require_shape(cloned, {2, 2});
    require_data(cloned, {1, 3, 2, 4});
}

TEST_CASE("contiguous copies non-contiguous view values from the logical offset",
          "[tensor][contiguous][view]") {
    cpptensor::Tensor base({4}, {0, 1, 2, 3});

    auto stepped = base.slice(0, 1, 4, 2);
    auto materialized = stepped.contiguous();

    REQUIRE_FALSE(stepped.is_contiguous());
    require_shape(materialized, {2});
    require_data(materialized, {1, 3});

    base.data()[1] = 99.0f;
    require_data(materialized, {1, 3});
}

TEST_CASE("tensordot reuses direct and transposed views for common layouts", "[tensordot]") {
    cpptensor::Tensor a({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    cpptensor::Tensor b({2, 3}, {1, 0, 1, 0, 1, 1});

    auto direct = cpptensor::tensordot(a, b, 1);
    require_shape(direct, {2, 2, 3});
    require_data(direct, {1, 2, 3, 3, 4, 7, 5, 6, 11, 7, 8, 15});

    cpptensor::Tensor c({2, 2}, {1, 2, 3, 4});
    cpptensor::Tensor d({3, 2}, {5, 6, 7, 8, 9, 10});

    auto transposed = cpptensor::tensordot(c, d, {0}, {1});
    require_shape(transposed, {2, 3});
    require_data(transposed, {23, 31, 39, 34, 46, 58});
}

TEST_CASE("tensordot matches a naive reference for arbitrary axes and offset views", "[tensordot]") {
    cpptensor::Tensor a({2, 3, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    });
    cpptensor::Tensor b({3, 2, 4}, {
        1, 0, 2, 1,
        0, 1, 3, 2,
        2, 2, 1, 0,
        1, 3, 0, 2,
        2, 1, 1, 1,
        0, 2, 2, 3
    });

    auto expected = naive_tensordot(a, b, {0, 2}, {1, 2});
    auto actual = cpptensor::tensordot(a, b, {0, 2}, {1, 2});
    require_shape(actual, expected.shape());
    require_data(actual, expected.data());

    cpptensor::Tensor base({4, 2}, {10, 11, 1, 2, 3, 4, 20, 21});
    auto sliced = base.slice(0, 1, 3);
    cpptensor::Tensor rhs({2, 3}, {5, 6, 7, 8, 9, 10});

    auto offset_expected = naive_tensordot(cpptensor::Tensor({2, 2}, {1, 2, 3, 4}), rhs, {1}, {0});
    auto offset_actual = cpptensor::tensordot(sliced, rhs, 1);
    require_shape(offset_actual, {2, 3});
    require_data(offset_actual, offset_expected.data());
}

TEST_CASE("pow preserves real-domain results for negative bases", "[arithmetic][pow]") {
    cpptensor::initialize_kernels();

    cpptensor::Tensor base({9}, {-1, -2, -3, -4, -5, -6, -7, -8, -9});
    cpptensor::Tensor even_exp({9}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    cpptensor::Tensor odd_exp({9}, {3, 3, 3, 3, 3, 3, 3, 3, 3});
    cpptensor::Tensor frac_exp({9}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});

    {
        ScopedCpuIsaOverride generic_only("generic");
        require_data(cpptensor::pow(base, even_exp), {1, 4, 9, 16, 25, 36, 49, 64, 81});
        require_data(cpptensor::pow(base, odd_exp), {-1, -8, -27, -64, -125, -216, -343, -512, -729});
        require_nan_data(cpptensor::pow(base, frac_exp));
    }

#ifdef BUILD_AVX2
    if (cpptensor::has_avx2()) {
        ScopedCpuIsaOverride avx2_only("avx2");
        require_data(cpptensor::pow(base, even_exp), {1, 4, 9, 16, 25, 36, 49, 64, 81});
        require_data(cpptensor::pow(base, odd_exp), {-1, -8, -27, -64, -125, -216, -343, -512, -729});
        require_nan_data(cpptensor::pow(base, frac_exp));
    }
#endif
}

#ifdef BUILD_AVX2
TEST_CASE("AVX2 pow handles SIMD chunks and scalar tails for negative bases", "[arithmetic][pow][avx2]") {
    if (!cpptensor::has_avx2()) {
        SUCCEED("Host CPU does not support AVX2");
        return;
    }

    cpptensor::Tensor base({9}, {-1, -2, -3, -4, -5, -6, -7, -8, -9});
    cpptensor::Tensor exponents({9}, {2, 3, 2, 3, 0.5f, 2, 3, 0.5f, 2});
    cpptensor::Tensor out = cpptensor::Tensor::full({9}, 0.0f);

    cpptensor::AVX2::pow_f32_avx2(base, exponents, out);

    REQUIRE(out.data()[0] == Approx(1.0f));
    REQUIRE(out.data()[1] == Approx(-8.0f));
    REQUIRE(out.data()[2] == Approx(9.0f));
    REQUIRE(out.data()[3] == Approx(-64.0f));
    REQUIRE(std::isnan(out.data()[4]));
    REQUIRE(out.data()[5] == Approx(36.0f));
    REQUIRE(out.data()[6] == Approx(-343.0f));
    REQUIRE(std::isnan(out.data()[7]));
    REQUIRE(out.data()[8] == Approx(81.0f));
}
#endif
