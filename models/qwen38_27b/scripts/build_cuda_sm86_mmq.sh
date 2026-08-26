#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
crate_dir="$(cd -- "${script_dir}/.." && pwd)"
output_dir="${1:-${crate_dir}/target/cuda-sm86-mmq}"
nvcc_command="${NVCC:-nvcc}"
nvcc_path="$(readlink -f -- "$(command -v -- "${nvcc_command}")")"
cuda_bin_dir="$(dirname -- "${nvcc_path}")"
cuda_root="$(cd -- "${cuda_bin_dir}/.." && pwd)"
cuda_include_dir="${cuda_root}/include"

test -f "${cuda_include_dir}/cuda_runtime.h"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd)"

(
  cd -- "${crate_dir}"
  "${nvcc_path}" \
    --cubin \
    --std=c++17 \
    --gpu-architecture=sm_86 \
    --use_fast_math \
    --ptxas-options=-v \
    -O3 \
    -I"${cuda_include_dir}" \
    kernels/cuda/q2q4_batched_mmq_sm86.cu \
    -o "${output_dir}/q2q4_batched_mmq_sm86.cubin"
)

sha256sum "${output_dir}/q2q4_batched_mmq_sm86.cubin" \
  > "${output_dir}/q2q4_batched_mmq_sm86.cubin.sha256"
"${cuda_bin_dir}/cuobjdump" --dump-resource-usage --dump-elf \
  "${output_dir}/q2q4_batched_mmq_sm86.cubin" \
  > "${output_dir}/q2q4_batched_mmq_sm86.cuobjdump.txt"
printf '%s\n' "${output_dir}/q2q4_batched_mmq_sm86.cubin"
