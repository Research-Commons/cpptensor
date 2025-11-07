#pragma once
#include <functional>
#include <map>

#include "cpptensor/tensor/tensor.hpp"
#include "cpptensor/enums/dispatcherEnum.h"


//Note to self
// 1. Write kernel code
// 2. Register that kernel in backend_loader
// 3. Make sure the ops calls that kernel
namespace cpptensor {

    class KernelRegistry {
    public:
        using KernelFunc = std::function<void(const Tensor&, const Tensor&, Tensor&)>;
        using UnaryKernelFunc = std::function<void(const Tensor&, Tensor&)>;
        using ReductionKernelFunc = std::function<void(const Tensor&, Tensor&, int, bool)>;

        static KernelRegistry& instance() { static KernelRegistry inst; return inst; }

        void registerKernel(OpType op, DeviceType dev, CpuIsa isa, KernelFunc fn);
        void registerKernel(OpType op, DeviceType dev, KernelFunc fn);

        void registerUnaryKernel(OpType op, DeviceType dev, CpuIsa isa, UnaryKernelFunc fn);
        void registerUnaryKernel(OpType op, DeviceType dev, UnaryKernelFunc fn);

        void registerReductionKernel(OpType op, DeviceType dev, CpuIsa isa, ReductionKernelFunc fn);
        void registerReductionKernel(OpType op, DeviceType dev, ReductionKernelFunc fn);

        // Try exact (op,dev,best_isa) then degrade to AVX2 then GENERIC
        KernelFunc getKernel(OpType op, DeviceType dev);

        UnaryKernelFunc getUnaryKernel(OpType op, DeviceType dev);

        ReductionKernelFunc getReductionKernel(OpType op, DeviceType dev);

    private:
        KernelRegistry() = default;
        std::map<DispatchKey, KernelFunc> forward_;
        std::map<DispatchKey, UnaryKernelFunc> unary_forward_;
        std::map<DispatchKey, ReductionKernelFunc> reduction_forward_;
    };

} // namespace cpptensor