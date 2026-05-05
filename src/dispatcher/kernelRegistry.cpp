#include "cpptensor/dispatcher/kernelRegistry.h"

#include "backend/isa/isaDetect.hpp"

#include <stdexcept>
#include <string>

namespace cpptensor {

    namespace {

        const char* op_name(OpType op) {
            switch (op) {
                case OpType::Add: return "Add";
                case OpType::Mul: return "Mul";
                case OpType::Sub: return "Sub";
                case OpType::Div: return "Div";
                case OpType::Pow: return "Pow";
                case OpType::Exp: return "Exp";
                case OpType::Log: return "Log";
                case OpType::Sqrt: return "Sqrt";
                case OpType::Sin: return "Sin";
                case OpType::Cos: return "Cos";
                case OpType::Tan: return "Tan";
                case OpType::Sigmoid: return "Sigmoid";
                case OpType::Relu: return "Relu";
                case OpType::Abs: return "Abs";
                case OpType::Sum: return "Sum";
                case OpType::Mean: return "Mean";
                case OpType::Max: return "Max";
                case OpType::Min: return "Min";
                case OpType::Eq: return "Eq";
                case OpType::Ne: return "Ne";
                case OpType::Gt: return "Gt";
                case OpType::Lt: return "Lt";
                case OpType::Ge: return "Ge";
                case OpType::Le: return "Le";
                case OpType::Matmul: return "Matmul";
                case OpType::Dot: return "Dot";
            }

            return "Unknown";
        }

        const char* device_name(DeviceType device) {
            switch (device) {
                case DeviceType::CPU: return "CPU";
                case DeviceType::CUDA: return "CUDA";
            }

            return "Unknown";
        }

        [[noreturn]] void throw_missing_kernel(const char* kernel_kind, OpType op, DeviceType dev) {
            throw std::runtime_error(
                std::string("No ") + kernel_kind + " kernel registered for op " +
                op_name(op) + " on device " + device_name(dev)
            );
        }

    } // namespace

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

    void KernelRegistry::registerReductionKernel(OpType op, DeviceType dev, CpuIsa isa, ReductionKernelFunc fn) {
        reduction_forward_[{op, dev, isa}] = std::move(fn);
    }
    void KernelRegistry::registerReductionKernel(OpType op, DeviceType dev, ReductionKernelFunc fn) {
        registerReductionKernel(op, dev, CpuIsa::GENERIC, std::move(fn));
    }

    // CPU dispatch tries best_isa then degrades to AVX2 then GENERIC.
    // Non-CPU dispatch requires an exact device match.
    KernelRegistry::KernelFunc KernelRegistry::getKernel(OpType op, DeviceType dev) {
        if (dev == DeviceType::CPU) {
            auto best = detect_best_cpu_isa();
            for (CpuIsa isa : {best, CpuIsa::AVX2, CpuIsa::GENERIC}) {
                auto it = forward_.find({op, dev, isa});
                if (it != forward_.end()) return it->second;
            }
        }

        auto exact = forward_.find({op, dev, CpuIsa::GENERIC});
        if (exact != forward_.end()) return exact->second;
        throw_missing_kernel("forward", op, dev);
    }

    KernelRegistry::UnaryKernelFunc KernelRegistry::getUnaryKernel(OpType op, DeviceType dev) {
        if (dev == DeviceType::CPU) {
            auto best = detect_best_cpu_isa();
            for (CpuIsa isa : {best, CpuIsa::AVX2, CpuIsa::GENERIC}) {
                auto it = unary_forward_.find({op, dev, isa});
                if (it != unary_forward_.end()) return it->second;
            }
        }

        auto exact = unary_forward_.find({op, dev, CpuIsa::GENERIC});
        if (exact != unary_forward_.end()) return exact->second;
        throw_missing_kernel("unary", op, dev);
    }

    KernelRegistry::ReductionKernelFunc KernelRegistry::getReductionKernel(OpType op, DeviceType dev) {
        if (dev == DeviceType::CPU) {
            auto best = detect_best_cpu_isa();
            for (CpuIsa isa : {best, CpuIsa::AVX2, CpuIsa::GENERIC}) {
                auto it = reduction_forward_.find({op, dev, isa});
                if (it != reduction_forward_.end()) return it->second;
            }
        }

        auto exact = reduction_forward_.find({op, dev, CpuIsa::GENERIC});
        if (exact != reduction_forward_.end()) return exact->second;
        throw_missing_kernel("reduction", op, dev);
    }
}
