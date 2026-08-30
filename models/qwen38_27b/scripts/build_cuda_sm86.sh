#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
crate_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_dir="$(cd -- "${crate_dir}/../.." && pwd)"
source_file="${crate_dir}/kernels/cuda/q2q4_fused_matvec_sm86.cu"
mtp_manifest="${crate_dir}/vendor/cuda/tensorrt_llm_mtp/UPSTREAM.json"
output_dir="${1:-${crate_dir}/target/cuda-sm86}"
nvcc_command="${NVCC:-nvcc}"
nvcc_path="$(readlink -f -- "$(command -v -- "${nvcc_command}")")"
cuda_bin_dir="$(dirname -- "${nvcc_path}")"
cuda_root="$(cd -- "${cuda_bin_dir}/.." && pwd)"
cuda_include_dir="${cuda_root}/include"

if [[ ! -f "${cuda_include_dir}/cuda_runtime.h" ]]; then
  printf 'CUDA runtime header not found under %s\n' "${cuda_include_dir}" >&2
  exit 1
fi

python3 "${repo_dir}/training/verify_tensorrt_mtp_extraction.py" "${mtp_manifest}"

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
    "kernels/cuda/$(basename -- "${source_file}")" \
    -o "${output_dir}/q2q4_fused_matvec_sm86.cubin"
)

sha256sum "${output_dir}/q2q4_fused_matvec_sm86.cubin" \
  > "${output_dir}/q2q4_fused_matvec_sm86.cubin.sha256"

"${cuda_bin_dir}/cuobjdump" --dump-resource-usage --dump-elf \
  "${output_dir}/q2q4_fused_matvec_sm86.cubin" \
  > "${output_dir}/q2q4_fused_matvec_sm86.cuobjdump.txt"

mtp_symbol="ctox_trtllm_mtp_accept_draft_token_sm86"
mtp_symbols="${output_dir}/q2q4_fused_matvec_sm86.mtp-symbols.txt"
"${cuda_bin_dir}/cuobjdump" --dump-sass \
  "${output_dir}/q2q4_fused_matvec_sm86.cubin" \
  | awk -v expected="${mtp_symbol}" \
      '$1 == "Function" && $2 == ":" && $3 == expected { print $3 }' \
  > "${mtp_symbols}"
if [[ "$(wc -l < "${mtp_symbols}")" -ne 1 ]]; then
  printf 'combined Q2/Q4 cubin does not export exactly one %s\n' "${mtp_symbol}" >&2
  exit 1
fi

printf '%s\n' "${output_dir}/q2q4_fused_matvec_sm86.cubin"
