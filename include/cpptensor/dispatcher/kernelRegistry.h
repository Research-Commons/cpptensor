#pragma once
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/enums/dispatcherEnum.h"
#include "cpptensor/backend/isa/isaDetect.hpp"

// This macro ensures two things:
// 1. Kernel registration code runs automatically at startup.
// 2. The object file containing it is not discarded by the linker.
#define CPPGRAD_REGISTER_BACKEND(NAME, BODY)                         \
static bool _register_##NAME##_kernels = [] {                    \
BODY;                                                        \
return true;                                                 \
}();                                                             \
extern "C" void cppgrad_force_link_##NAME() {}

//Note to self
// 1. Write kernel code
// 2. Register that kernel in backend_loader
// 3. Make sure the ops calls that kernel
namespace cpptensor {

class KernelRegistry {
public:
    using KernelFunc = std::function<void(const Tensor&, const Tensor&, Tensor&)>;
    using UnaryKernelFunc    = std::function<void(const Tensor&, Tensor&)>;

    static KernelRegistry& instance() { static KernelRegistry inst; return inst; }

    void registerKernel(OpType op, DeviceType dev, CpuIsa isa, KernelFunc fn) {
        forward_[{op, dev, isa}] = std::move(fn);
    }
    void registerKernel(OpType op, DeviceType dev, KernelFunc fn) {
        registerKernel(op, dev, CpuIsa::GENERIC, std::move(fn));
    }

    void registerUnaryKernel(OpType op, DeviceType dev, CpuIsa isa, UnaryKernelFunc fn) {
        unary_forward_[{op, dev, isa}] = std::move(fn);
    }
    void registerUnaryKernel(OpType op, DeviceType dev, UnaryKernelFunc fn) {
        registerUnaryKernel(op, dev, CpuIsa::GENERIC, std::move(fn));
    }

    // Try exact (op,dev,best_isa) then degrade to AVX2 then GENERIC
    KernelFunc getKernel(OpType op, DeviceType dev) {
        if (dev == DeviceType::CPU) {
            auto best = detect_best_cpu_isa();
            for (CpuIsa isa : {best, CpuIsa::AVX2, CpuIsa::GENERIC}) {
                auto it = forward_.find({op, dev, isa});
                if (it != forward_.end()) return it->second;
            }
        }
        // Non-CPU (e.g., CUDA) → exact, then CPU fallback
        auto exact = forward_.find({op, dev, CpuIsa::GENERIC});
        if (exact != forward_.end()) return exact->second;
        auto cpu_fallback = forward_.find({op, DeviceType::CPU, CpuIsa::GENERIC});
        if (cpu_fallback != forward_.end()) return cpu_fallback->second;
        throw std::runtime_error("No forward kernel registered for this op/device");
    }

    UnaryKernelFunc getUnaryKernel(OpType op, DeviceType dev) {
        if (dev == DeviceType::CPU) {
            auto best = detect_best_cpu_isa();
            for (CpuIsa isa : {best, CpuIsa::AVX2, CpuIsa::GENERIC}) {
                auto it = unary_forward_.find({op, dev, isa});
                if (it != unary_forward_.end()) return it->second;
            }
        }
        // Non-CPU fallback
        auto exact = unary_forward_.find({op, dev, CpuIsa::GENERIC});
        if (exact != unary_forward_.end()) return exact->second;
        auto cpu_fallback = unary_forward_.find({op, DeviceType::CPU, CpuIsa::GENERIC});
        if (cpu_fallback != unary_forward_.end()) return cpu_fallback->second;
        throw std::runtime_error("No unary kernel registered for this op/device");
    }

private:
    KernelRegistry() = default;
    std::map<DispatchKey, KernelFunc> forward_;
    std::map<DispatchKey, UnaryKernelFunc> unary_forward_;
};

} // namespace cppgrad