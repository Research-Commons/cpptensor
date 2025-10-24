#include "cpptensor/dispatcher/kernelRegistry.h"

#include "backend/isa/isaDetect.hpp"

namespace cpptensor {

    void KernelRegistry::registerKernel(OpType op, DeviceType dev, CpuIsa isa, KernelFunc fn) {
        forward_[{op, dev, isa}] = std::move(fn);
    }
    void KernelRegistry::registerKernel(OpType op, DeviceType dev, KernelFunc fn) {
        registerKernel(op, dev, CpuIsa::GENERIC, std::move(fn));
    }

    void KernelRegistry::registerUnaryKernel(OpType op, DeviceType dev, CpuIsa isa, UnaryKernelFunc fn) {
        unary_forward_[{op, dev, isa}] = std::move(fn);
    }
    void KernelRegistry::registerUnaryKernel(OpType op, DeviceType dev, UnaryKernelFunc fn) {
        registerUnaryKernel(op, dev, CpuIsa::GENERIC, std::move(fn));
    }

    // Try exact (op,dev,best_isa) then degrade to AVX2 then GENERIC
    KernelRegistry::KernelFunc KernelRegistry::getKernel(OpType op, DeviceType dev) {
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

    KernelRegistry::UnaryKernelFunc KernelRegistry::getUnaryKernel(OpType op, DeviceType dev) {
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
}
