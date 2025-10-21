#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <sstream>

#include "cpptensor/tensor/tensorimpl.hpp"
#include "cpptensor/enums/dispatcherEnum.h"   // for DeviceType

namespace cpptensor {

/**
 * Public-facing Tensor class.
 * API modeled after your previous header, but using std::vector<float>
 * instead of ArrayFire's af::array.
 */
class Tensor {
public:
    // -------- Constructors --------
    // Construct from shape + values (row-major)
    Tensor(const std::vector<size_t>& shape,
           const std::vector<float>& values,
           DeviceType device = DeviceType::CPU);

    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // -------- Factory Methods --------
    static Tensor zeros(const std::vector<size_t>& shape,
                        DeviceType device = DeviceType::CPU);
    static Tensor ones(const std::vector<size_t>& shape,
                       DeviceType device = DeviceType::CPU);
    static Tensor randn(const std::vector<size_t>& shape,
                        DeviceType device = DeviceType::CPU);
    static Tensor full(const std::vector<size_t>& shape,
                       float value,
                       DeviceType device = DeviceType::CPU);

    // -------- Shape and Info --------
    std::vector<size_t> shape() const;
    size_t numel() const;
    size_t ndim() const;
    DeviceType device_type() const;

    void print() const;
    void print_pretty() const;

    // -------- Data Access --------
    const std::vector<float>& data() const;
    std::vector<float>& data();

    const std::vector<size_t>& stride() const;
    std::vector<size_t>& stride();

    std::shared_ptr<TensorImpl> impl() const;

    // -------- Reduction Ops --------
    // Tensor sum(int dim = -1, bool keepdim = false) const;
    // Tensor mean(int dim = -1, bool keepdim = false) const;
    // Tensor max(int dim = -1, bool keepdim = false) const;

    // Operator overloads (elementwise). Implemented in tensor.cpp
    friend Tensor operator+(const Tensor&, const Tensor&);
    friend Tensor operator-(const Tensor&, const Tensor&);
    friend Tensor operator*(const Tensor&, const Tensor&);
    friend Tensor operator/(const Tensor&, const Tensor&);

    friend Tensor operator+(const Tensor&, float);
    friend Tensor operator+(float, const Tensor&);
    friend Tensor operator-(const Tensor&, float);
    friend Tensor operator-(float, const Tensor&);
    friend Tensor operator*(const Tensor&, float);
    friend Tensor operator*(float, const Tensor&);
    friend Tensor operator/(const Tensor&, float);
    friend Tensor operator/(float, const Tensor&);

    friend Tensor operator-(const Tensor&);  // unary minus

protected:
    // Construct scalar or filled tensor
    Tensor(const std::vector<size_t>& shape,
           float value,
           DeviceType device = DeviceType::CPU);

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace cppgrad