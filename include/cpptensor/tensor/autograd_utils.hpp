#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpptensor/tensor/tensor.hpp"

namespace cpptensor::autograd {

inline std::vector<size_t> compute_row_major_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> stride(shape.size(), 1);
    if (shape.empty()) {
        return stride;
    }

    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        stride[static_cast<size_t>(i)] = stride[static_cast<size_t>(i) + 1] * shape[static_cast<size_t>(i) + 1];
    }

    return stride;
}

inline void for_each_index(const std::vector<size_t>& shape,
                           const std::function<void(const std::vector<size_t>&)>& fn) {
    if (shape.empty()) {
        fn({});
        return;
    }

    std::vector<size_t> idx(shape.size(), 0);
    while (true) {
        fn(idx);

        int dim = static_cast<int>(shape.size()) - 1;
        while (dim >= 0) {
            const size_t d = static_cast<size_t>(dim);
            ++idx[d];
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
            --dim;
        }

        if (dim < 0) {
            break;
        }
    }
}

inline size_t linear_index(const std::vector<size_t>& idx,
                           const std::vector<size_t>& stride) {
    size_t out = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        out += idx[i] * stride[i];
    }
    return out;
}

inline std::vector<float> reduce_sum_to_shape(const std::vector<float>& grad,
                                              const std::vector<size_t>& grad_shape,
                                              const std::vector<size_t>& target_shape) {
    if (target_shape == grad_shape) {
        return grad;
    }

    const size_t target_numel = target_shape.empty() ? 1 :
        std::accumulate(target_shape.begin(), target_shape.end(), static_cast<size_t>(1), std::multiplies<>());
    std::vector<float> out(target_numel, 0.0f);

    const auto grad_stride = compute_row_major_strides(grad_shape);
    const auto target_stride = compute_row_major_strides(target_shape);

    const size_t grad_rank = grad_shape.size();
    const size_t target_rank = target_shape.size();
    if (target_rank > grad_rank) {
        throw std::runtime_error("autograd: cannot reduce gradient to higher-rank shape");
    }

    const size_t align = grad_rank - target_rank;

    for_each_index(grad_shape, [&](const std::vector<size_t>& grad_idx) {
        std::vector<size_t> target_idx(target_rank, 0);
        for (size_t i = 0; i < target_rank; ++i) {
            const size_t grad_dim = i + align;
            target_idx[i] = (target_shape[i] == 1) ? 0 : grad_idx[grad_dim];
        }

        out[linear_index(target_idx, target_stride)] += grad[linear_index(grad_idx, grad_stride)];
    });

    return out;
}

inline std::vector<float> broadcast_to_shape(const std::vector<float>& input,
                                             const std::vector<size_t>& input_shape,
                                             const std::vector<size_t>& output_shape) {
    if (input_shape == output_shape) {
        return input;
    }

    const auto input_stride = compute_row_major_strides(input_shape);
    const auto output_stride = compute_row_major_strides(output_shape);
    const size_t output_numel = output_shape.empty() ? 1 :
        std::accumulate(output_shape.begin(), output_shape.end(), static_cast<size_t>(1), std::multiplies<>());

    const size_t output_rank = output_shape.size();
    const size_t input_rank = input_shape.size();
    if (input_rank > output_rank) {
        throw std::runtime_error("autograd: cannot broadcast higher-rank input to lower-rank output");
    }

    const size_t align = output_rank - input_rank;
    std::vector<float> out(output_numel, 0.0f);

    for_each_index(output_shape, [&](const std::vector<size_t>& out_idx) {
        std::vector<size_t> in_idx(input_rank, 0);
        for (size_t i = 0; i < input_rank; ++i) {
            const size_t out_dim = i + align;
            in_idx[i] = (input_shape[i] == 1) ? 0 : out_idx[out_dim];
        }

        out[linear_index(out_idx, output_stride)] = input[linear_index(in_idx, input_stride)];
    });

    return out;
}

inline std::vector<size_t> unsqueeze_shape_at(const std::vector<size_t>& shape, size_t dim) {
    std::vector<size_t> out;
    out.reserve(shape.size() + 1);
    for (size_t i = 0; i < dim; ++i) {
        out.push_back(shape[i]);
    }
    out.push_back(1);
    for (size_t i = dim; i < shape.size(); ++i) {
        out.push_back(shape[i]);
    }
    return out;
}

inline std::vector<float> expand_reduction_grad(const std::vector<float>& grad_output,
                                                const std::vector<size_t>& grad_shape,
                                                const std::vector<size_t>& input_shape,
                                                std::optional<int> dim,
                                                bool keepdim) {
    if (!dim.has_value()) {
        const float seed = grad_output.empty() ? 0.0f : grad_output[0];
        const size_t n = input_shape.empty() ? 1 :
            std::accumulate(input_shape.begin(), input_shape.end(), static_cast<size_t>(1), std::multiplies<>());
        return std::vector<float>(n, seed);
    }

    int d = dim.value();
    if (d < 0) {
        d += static_cast<int>(input_shape.size());
    }
    if (d < 0 || d >= static_cast<int>(input_shape.size())) {
        throw std::runtime_error("autograd: reduction dim out of range during backward");
    }

    std::vector<size_t> normalized_grad_shape = grad_shape;
    if (!keepdim) {
        normalized_grad_shape = unsqueeze_shape_at(grad_shape, static_cast<size_t>(d));
    }

    return broadcast_to_shape(grad_output, normalized_grad_shape, input_shape);
}

inline void throw_if_requires_grad(const Tensor& tensor, const char* op_name) {
    if (tensor.requires_grad()) {
        throw std::runtime_error(
            std::string("autograd: operation '") + op_name +
            "' is not supported for tensors that require gradients");
    }
}

inline void throw_if_requires_grad(const Tensor& a,
                                   const Tensor& b,
                                   const char* op_name) {
    if (a.requires_grad() || b.requires_grad()) {
        throw std::runtime_error(
            std::string("autograd: operation '") + op_name +
            "' is not supported for tensors that require gradients");
    }
}

} // namespace cpptensor::autograd
