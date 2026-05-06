#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpptensor/ops/helperOps.hpp"

namespace cpptensor {

template <typename Predicate>
Tensor compare_tensors(const Tensor& a,
                       const Tensor& b,
                       Predicate&& predicate) {
    if (a.device_type() != b.device_type()) {
        throw std::runtime_error(
            "Binary op requires matching devices, got lhs=" +
            std::string(deviceTypeName(a.device_type())) +
            " and rhs=" + std::string(deviceTypeName(b.device_type())));
    }

    const auto out_shape = computeBroadcastShape(a.shape(), b.shape());
    const auto& a_data = a.data();
    const auto& b_data = b.data();
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    const size_t n = out_shape.size();

    std::vector<size_t> a_pad(n, 1), b_pad(n, 1);
    const size_t na = a_shape.size();
    const size_t nb = b_shape.size();
    for (size_t i = 0; i < n; ++i) {
        a_pad[i] = (i < n - na) ? 1 : a_shape[i - (n - na)];
        b_pad[i] = (i < n - nb) ? 1 : b_shape[i - (n - nb)];
    }

    std::vector<size_t> stride_a(n, 0), stride_b(n, 0), stride_out(n, 0);
    if (n > 0) {
        stride_a[n - 1] = 1;
        stride_b[n - 1] = 1;
        stride_out[n - 1] = 1;
    }

    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        stride_a[static_cast<size_t>(i)] = stride_a[static_cast<size_t>(i) + 1] * a_pad[static_cast<size_t>(i) + 1];
        stride_b[static_cast<size_t>(i)] = stride_b[static_cast<size_t>(i) + 1] * b_pad[static_cast<size_t>(i) + 1];
        stride_out[static_cast<size_t>(i)] = stride_out[static_cast<size_t>(i) + 1] * out_shape[static_cast<size_t>(i) + 1];
    }

    size_t total = 1;
    for (size_t dim : out_shape) {
        total *= dim;
    }

    std::vector<bool> out(total, false);
    for (size_t pos = 0; pos < total; ++pos) {
        size_t idx_a = 0;
        size_t idx_b = 0;
        for (size_t dim = 0; dim < n; ++dim) {
            const size_t i = (pos / stride_out[dim]) % out_shape[dim];
            if (a_pad[dim] != 1) idx_a += i * stride_a[dim];
            if (b_pad[dim] != 1) idx_b += i * stride_b[dim];
        }
        out[pos] = std::forward<Predicate>(predicate)(a_data[idx_a], b_data[idx_b]);
    }

    return Tensor(out_shape, out, a.device_type());
}

} // namespace cpptensor
