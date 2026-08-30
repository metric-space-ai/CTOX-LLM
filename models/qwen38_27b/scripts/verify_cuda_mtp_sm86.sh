#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
crate_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_dir="$(cd -- "${crate_dir}/../.." && pwd)"
vendor_dir="${crate_dir}/vendor/cuda/tensorrt_llm_mtp"
source_file="${vendor_dir}/mtp_accept_draft_token_sm86.cu"
manifest_file="${vendor_dir}/UPSTREAM.json"
output_dir="${1:-${crate_dir}/target/cuda-sm86-mtp-verify}"
expected_symbol="ctox_trtllm_mtp_accept_draft_token_sm86"
nvcc_command="${NVCC:-nvcc}"
nvcc_path="$(readlink -f -- "$(command -v -- "${nvcc_command}")")"
cuda_bin_dir="$(dirname -- "${nvcc_path}")"
cuobjdump_path="${cuda_bin_dir}/cuobjdump"

if [[ ! -x "${cuobjdump_path}" ]]; then
  printf 'cuobjdump is absent next to nvcc: %s\n' "${cuobjdump_path}" >&2
  exit 1
fi

python3 "${repo_dir}/training/verify_tensorrt_mtp_extraction.py" "${manifest_file}"

mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd)"
cubin="${output_dir}/mtp_accept_draft_token_sm86.cubin"
resource_dump="${output_dir}/mtp_accept_draft_token_sm86.cuobjdump.txt"
symbol_dump="${output_dir}/mtp_accept_draft_token_sm86.symbols.txt"

(
  cd -- "${crate_dir}"
  "${nvcc_path}" \
    --cubin \
    --std=c++17 \
    --gpu-architecture=sm_86 \
    --ptxas-options=-v \
    -O3 \
    "${source_file}" \
    -o "${cubin}"
)

sha256sum "${cubin}" > "${cubin}.sha256"
"${cuobjdump_path}" --dump-resource-usage --dump-elf "${cubin}" > "${resource_dump}"
"${cuobjdump_path}" --dump-sass "${cubin}" \
  | awk '$1 == "Function" && $2 == ":" { print $3 }' > "${symbol_dump}"

if [[ "$(wc -l < "${symbol_dump}")" -ne 1 ]] \
  || ! grep -Fxq -- "${expected_symbol}" "${symbol_dump}"; then
  printf 'standalone MTP cubin does not export exactly %s\n' "${expected_symbol}" >&2
  exit 1
fi

printf '%s\n' "${cubin}"
