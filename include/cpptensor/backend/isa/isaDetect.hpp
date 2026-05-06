#pragma once

#include "cpptensor/enums/dispatcherEnum.h"

#include <cstdlib>
#include <cstdint>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define CPPTENSOR_HAS_X86_CPUID 1
#else
#define CPPTENSOR_HAS_X86_CPUID 0
#endif

#if CPPTENSOR_HAS_X86_CPUID
    #if defined(_MSC_VER)
    #include <immintrin.h>
    #include <intrin.h>
    #else
    #include <cpuid.h>
    #endif
#endif

namespace cpptensor {
    namespace detail {
    #if CPPTENSOR_HAS_X86_CPUID
        struct CpuIdRegs {
            unsigned int eax;
            unsigned int ebx;
            unsigned int ecx;
            unsigned int edx;
        };

        inline CpuIdRegs cpuid(unsigned int leaf, unsigned int subleaf = 0) {
    #if defined(_MSC_VER)
            int regs[4];
            __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
            return {
                static_cast<unsigned int>(regs[0]),
                static_cast<unsigned int>(regs[1]),
                static_cast<unsigned int>(regs[2]),
                static_cast<unsigned int>(regs[3])
            };
    #else
            CpuIdRegs regs {};
            __cpuid_count(leaf, subleaf, regs.eax, regs.ebx, regs.ecx, regs.edx);
            return regs;
    #endif
        }

        inline bool has_cpuid_leaf(unsigned int leaf) {
    #if defined(_MSC_VER)
            int regs[4];
            __cpuid(regs, 0);
            return static_cast<unsigned int>(regs[0]) >= leaf;
    #else
            return __get_cpuid_max(0, 0) >= leaf;
    #endif
        }

        inline std::uint64_t xgetbv(unsigned int index) {
    #if defined(_MSC_VER)
            return _xgetbv(index);
    #else
            unsigned int eax = 0;
            unsigned int edx = 0;
            __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(index));
            return (static_cast<std::uint64_t>(edx) << 32) | eax;
    #endif
        }
    #endif
    } // namespace detail

    inline bool has_avx2() {
    #if !CPPTENSOR_HAS_X86_CPUID
            return false;
    #else
            if (!detail::has_cpuid_leaf(1) || !detail::has_cpuid_leaf(7)) {
                return false;
            }

            const auto leaf1 = detail::cpuid(1);
            constexpr unsigned int OSXSAVE_BIT = 1u << 27;
            constexpr unsigned int AVX_BIT = 1u << 28;
            constexpr unsigned int FMA_BIT = 1u << 12;
            if ((leaf1.ecx & (OSXSAVE_BIT | AVX_BIT | FMA_BIT)) != (OSXSAVE_BIT | AVX_BIT | FMA_BIT)) {
                return false;
            }

            const std::uint64_t xcr0 = detail::xgetbv(0);
            constexpr std::uint64_t XMM_YMM_STATE_MASK = (1ull << 1) | (1ull << 2);
            if ((xcr0 & XMM_YMM_STATE_MASK) != XMM_YMM_STATE_MASK) {
                return false;
            }

            const auto leaf7 = detail::cpuid(7, 0);
            constexpr unsigned int AVX2_BIT = 1u << 5;
            return (leaf7.ebx & AVX2_BIT) != 0;
    #endif
        }

    inline bool has_avx512f() {
    #if !CPPTENSOR_HAS_X86_CPUID
            return false;
    #else
            if (!has_avx2()) {
                return false;
            }

            const std::uint64_t xcr0 = detail::xgetbv(0);
            constexpr std::uint64_t AVX512_STATE_MASK =
                (1ull << 1) | // XMM state
                (1ull << 2) | // YMM state
                (1ull << 5) | // opmask state
                (1ull << 6) | // ZMM_hi256 state
                (1ull << 7);  // Hi16_ZMM state
            if ((xcr0 & AVX512_STATE_MASK) != AVX512_STATE_MASK) {
                return false;
            }

            const auto leaf7 = detail::cpuid(7, 0);
            constexpr unsigned int AVX512F_BIT = 1u << 16;
            constexpr unsigned int AVX512DQ_BIT = 1u << 17;
            constexpr unsigned int AVX512BW_BIT = 1u << 30;
            constexpr unsigned int AVX512VL_BIT = 1u << 31;

            const unsigned int required_bits = AVX512F_BIT | AVX512DQ_BIT | AVX512BW_BIT | AVX512VL_BIT;
            return (leaf7.ebx & required_bits) == required_bits;
    #endif
        }

    inline CpuIsa detect_best_cpu_isa() {
        // Optional: allow override via env
        if (const char* env = std::getenv("CPPGRAD_CPU_ISA")) {
            const std::string override(env);
            if (override == "avx512") {
                if (has_avx512f()) return CpuIsa::AVX512;
                if (has_avx2()) return CpuIsa::AVX2;
                return CpuIsa::GENERIC;
            }
            if (override == "avx2") {
                if (has_avx2()) return CpuIsa::AVX2;
                return CpuIsa::GENERIC;
            }
            return CpuIsa::GENERIC;
        }
        if (has_avx512f()) return CpuIsa::AVX512;
        if (has_avx2())    return CpuIsa::AVX2;
        return CpuIsa::GENERIC;
    }
}
#undef CPPTENSOR_HAS_X86_CPUID
