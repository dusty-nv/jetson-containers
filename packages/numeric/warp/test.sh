#!/usr/bin/env bash
set -e

MODE="${1:-core}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $MODE == "torch" || $MODE == "all" ]]; then
  USE_TORCH=1
fi

if [[ $MODE == "jax" || $MODE == "all" ]]; then
  USE_JAX=1
fi

function warp_test() {
  printf "▶️  Warp - $2 RUNNING\n\n$1\n\n"
  $1
  printf "\n✅ Warp - $2 OK\n"
}

# https://github.com/NVIDIA/warp/tree/main#getting-started
warp_test "python3 test.py" "Getting Started"

# https://github.com/NVIDIA/warp/tree/main/warp/examples
WARP_ARGS="--stage_path=/tmp/example.usd"

warp_test "python3 -m warp.examples.core.example_dem --num_frames=50 $WARP_ARGS" "Examples (dem)"
warp_test "python3 -m warp.examples.core.example_sph --num_frames=10 $WARP_ARGS" "Examples (sph)"
warp_test "python3 -m warp.examples.core.example_wave --num_frames=100 $WARP_ARGS" "Examples (wave)"
warp_test "python3 -m warp.examples.core.example_mesh --num_frames=100 $WARP_ARGS" "Examples (mesh)"
warp_test "python3 -m warp.examples.core.example_mesh_intersect $WARP_ARGS" "Examples (mesh_intersect)"
warp_test "python3 -m warp.examples.core.example_fluid --num_frames=1000 $WARP_ARGS" "Examples (fluid)"
warp_test "python3 -m warp.examples.core.example_cupy --num_frames=10 $WARP_ARGS" "Examples (cupy)"
warp_test "python3 -m warp.examples.core.example_nvdb --num_frames=10 $WARP_ARGS" "Examples (nvdb)"
warp_test "python3 -m warp.examples.core.example_raycast $WARP_ARGS" "Examples (raycast) "
warp_test "python3 -m warp.examples.core.example_raymarch $WARP_ARGS" "Examples (raymarch)"
warp_test "python3 -m warp.examples.core.example_sample_mesh --num_frames=8 $WARP_ARGS" "Examples (sample_mesh)"

if [ "$USE_TORCH" == 1 ]; then
  warp_test "python3 -m warp.examples.core.example_torch --num_frames=1000" "Examples (torch)"
fi

# https://github.com/NVIDIA/warp/blob/main/warp/examples/benchmarks/benchmark.sh
warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth warp_cpu" "Benchmarks (CPU)"
warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth warp_gpu" "Benchmarks (GPU)"

warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth numpy" "Benchmarks (numpy)"
warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth numba" "Benchmarks (numba)"
warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth cupy" "Benchmarks (cupy)"

if [ "$USE_TORCH" == 1 ]; then
  warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth torch_cpu" "Benchmarks (torch_cpu)"
  warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth torch_gpu" "Benchmarks (torch_gpu)"
fi

if [ "$USE_JAX" == 1 ]; then
  warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth jax_cpu" "Benchmarks (jax_cpu)"
  warp_test "python3 -m warp.examples.benchmarks.benchmark_cloth jax_gpu" "Benchmarks (jax_gpu)"
fi
