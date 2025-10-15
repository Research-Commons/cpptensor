#!/usr/bin/env bash
set -e

# ==========================================================
#  cppgrad – full CPU / AVX2 / AVX512 / CUDA benchmark suite
# ==========================================================

echo "📊 Running cppgrad benchmark suite..."
echo "------------------------------------"

# 1️⃣ Ensure CPU runs at fixed performance frequency
if command -v cpupower &> /dev/null; then
    echo "⚙️  Setting CPU governor to 'performance'..."
    sudo cpupower frequency-set -g performance > /dev/null
else
    echo "⚠️  'cpupower' not found — skipping frequency fix."
fi

# 2️⃣ Disable ASLR for reproducibility
echo "🧠 Disabling ASLR temporarily..."
sudo bash -c 'echo 0 > /proc/sys/kernel/randomize_va_space'

# 3️⃣ Move to benchmark directory
cd "$(dirname "$0")/cmake-build-release/benchmarks" || {
    echo "❌ Cannot find build/benchmarks directory."
    exit 1
}

# 4️⃣ Run all benchmarks one by one and log results
run_bench() {
    local name="$1"
    local binary="./$2"
    local log="${name}.log"

    if [ -f "$binary" ]; then
        echo ""
        echo "🚀 Running ${name}..."
        echo "------------------------------------"
        "$binary" | tee "$log"
    else
        echo "⚠️  Skipping ${name}: binary not found ($binary)"
    fi
}

run_bench "CPU"      "cppgrad_bench_cpu"
run_bench "AVX2"     "cppgrad_bench_avx2"
run_bench "AVX512"   "cppgrad_bench_avx512"
run_bench "CUDA"     "cppgrad_bench_cuda"

# 5️⃣ Restore system settings
echo ""
echo "🧩 Restoring ASLR and CPU governor..."
sudo bash -c 'echo 2 > /proc/sys/kernel/randomize_va_space'
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g schedutil > /dev/null
fi

echo ""
echo "✅ All benchmarks complete."
echo "📂 Logs saved as: cpu.log, avx2.log, avx512.log, cuda.log"

