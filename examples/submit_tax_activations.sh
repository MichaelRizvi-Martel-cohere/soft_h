#!/usr/bin/env bash
#
# Prepare or submit a Tax activation-extraction job from the Cohere login VM.

set -euo pipefail

readonly DEFAULT_CONTEXT="cw-us-east-04-prod"
readonly DEFAULT_QUEUE="post-training-smifs-queue"
readonly DEFAULT_PRIORITY="dev-medium"
readonly DEFAULT_HOOKS="block_0_block_output"
readonly DEFAULT_TENSOR_PARALLEL="4"
readonly DEFAULT_SEQUENCE_LENGTH="512"
readonly DEFAULT_OUTPUT_PREFIX="gs://cohere-dev/michael-rizvi/soft_h_tax"
readonly DEFAULT_TAX_REPO="${HOME}/repos/tax"
readonly DEFAULT_TIME_LIMIT="1h"
readonly DEFAULT_N_BINS="100"
readonly DEFAULT_SEED="0"
readonly SOFT_H_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUNTIME_PACKAGE_RELATIVE="scripts/_soft_h_runtime"

usage() {
  cat <<'EOF'
Usage:
  submit_tax_activations.sh --checkpoint URI --dataset-parquet URI \
    --max-batches N --run-name NAME [options]

Prepare a one-node, 8-H100 Tax activation-extraction job. By default, this
prints the complete job specification without submitting it.

Required:
  --checkpoint URI       Fax checkpoint under s3://us-east-01a/
  --dataset-parquet URI  Frozen C4 Parquet path/glob under gs://cohere-dev/
  --max-batches N        Number of Fax eval batches to process
  --run-name NAME        Unique run name used for the output directory

Options:
  --hooks LIST           Comma-separated hooks (default: block_0_block_output)
  --output-dir URI       Destination under gs://cohere-dev/
  --context CONTEXT      Kubernetes context (default: cw-us-east-04-prod)
  --queue QUEUE          Kueue queue (default: post-training-smifs-queue)
  --priority PRIORITY    Priority class (default: dev-medium)
  --tax-repo PATH        Tax checkout (default: ~/repos/tax)
  --online-entropy       Accumulate entropy online instead of writing NPZ shards
  --n-bins N             Number of soft bins (default: 100)
  --seed N               Soft-bin reference-point seed (default: 0)
  --submit               Authenticate and submit the displayed specification
  -h, --help             Show this help

Examples:
  # Preview only.
  examples/submit_tax_activations.sh \
    --checkpoint s3://us-east-01a/promoted-checkpoints/.../ckpt-536 \
    --dataset-parquet gs://cohere-dev/michael-rizvi/soft_h/c4/.../documents.parquet \
    --max-batches 1 \
    --run-name limpgods_ckpt536_block0

  # Submit after reviewing the preview.
  examples/submit_tax_activations.sh \
    --checkpoint s3://us-east-01a/promoted-checkpoints/.../ckpt-536 \
    --dataset-parquet gs://cohere-dev/michael-rizvi/soft_h/c4/.../documents.parquet \
    --max-batches 1 \
    --run-name limpgods_ckpt536_block0 \
    --submit
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

checkpoint=""
dataset_parquet=""
max_batches=""
run_name=""
hooks="${DEFAULT_HOOKS}"
output_dir=""
context="${DEFAULT_CONTEXT}"
queue="${DEFAULT_QUEUE}"
priority="${DEFAULT_PRIORITY}"
tax_repo="${DEFAULT_TAX_REPO}"
online_entropy=false
n_bins="${DEFAULT_N_BINS}"
seed="${DEFAULT_SEED}"
submit=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      [[ $# -ge 2 ]] || die "--checkpoint requires a value."
      checkpoint="$2"
      shift 2
      ;;
    --dataset-parquet)
      [[ $# -ge 2 ]] || die "--dataset-parquet requires a value."
      dataset_parquet="$2"
      shift 2
      ;;
    --max-batches)
      [[ $# -ge 2 ]] || die "--max-batches requires a value."
      max_batches="$2"
      shift 2
      ;;
    --run-name)
      [[ $# -ge 2 ]] || die "--run-name requires a value."
      run_name="$2"
      shift 2
      ;;
    --hooks)
      [[ $# -ge 2 ]] || die "--hooks requires a value."
      hooks="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value."
      output_dir="$2"
      shift 2
      ;;
    --context)
      [[ $# -ge 2 ]] || die "--context requires a value."
      context="$2"
      shift 2
      ;;
    --queue)
      [[ $# -ge 2 ]] || die "--queue requires a value."
      queue="$2"
      shift 2
      ;;
    --priority)
      [[ $# -ge 2 ]] || die "--priority requires a value."
      priority="$2"
      shift 2
      ;;
    --tax-repo)
      [[ $# -ge 2 ]] || die "--tax-repo requires a value."
      tax_repo="$2"
      shift 2
      ;;
    --online-entropy)
      online_entropy=true
      shift
      ;;
    --n-bins)
      [[ $# -ge 2 ]] || die "--n-bins requires a value."
      n_bins="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || die "--seed requires a value."
      seed="$2"
      shift 2
      ;;
    --submit)
      submit=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${checkpoint}" ]] || die "--checkpoint is required."
[[ -n "${dataset_parquet}" ]] || die "--dataset-parquet is required."
[[ -n "${max_batches}" ]] || die "--max-batches is required."
[[ -n "${run_name}" ]] || die "--run-name is required."
[[ "${checkpoint}" =~ ^s3://us-east-01a/[A-Za-z0-9._/-]+$ ]] ||
  die "--checkpoint must be a shell-safe path under s3://us-east-01a/."
[[ "${dataset_parquet}" =~ ^gs://cohere-dev/[A-Za-z0-9._/*-]+\.parquet$ ]] ||
  die "--dataset-parquet must be a shell-safe Parquet path under gs://cohere-dev/."
[[ "${max_batches}" =~ ^[1-9][0-9]*$ ]] || die "--max-batches must be a positive integer."
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--run-name contains unsupported characters."
[[ "${hooks}" =~ ^[A-Za-z0-9_,]+$ ]] || die "--hooks must be a comma-separated list of hook names."
[[ "${context}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--context contains unsupported characters."
[[ "${queue}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--queue contains unsupported characters."
[[ "${priority}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--priority contains unsupported characters."
[[ "${n_bins}" =~ ^[1-9][0-9]*$ ]] && (( n_bins >= 2 )) || die "--n-bins must be an integer of at least 2."
[[ "${seed}" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer."

if [[ -z "${output_dir}" ]]; then
  output_dir="${DEFAULT_OUTPUT_PREFIX}/${run_name}"
fi
[[ "${output_dir}" =~ ^gs://cohere-dev/[A-Za-z0-9._/-]+$ ]] ||
  die "--output-dir must be a shell-safe path under gs://cohere-dev/."

tax_repo="$(realpath "${tax_repo}")"
[[ -f "${tax_repo}/ops/kjobs-compute.yaml" ]] || die "Tax compute config not found under ${tax_repo}."
[[ -f "${tax_repo}/scripts/extract_activations.py" ]] || die "Tax activation extractor not found under ${tax_repo}."
package_script="${SOFT_H_REPO}/examples/package_soft_h_for_tax.py"
[[ -f "${package_script}" ]] || die "soft_h packaging script not found at ${package_script}."

kjobs_command=""
if command -v kjobs >/dev/null 2>&1; then
  kjobs_command="$(command -v kjobs)"
elif [[ -x "${HOME}/repos/kueue-jobs-cli/jobs.py" ]]; then
  kjobs_command="${HOME}/repos/kueue-jobs-cli/jobs.py"
else
  die "kjobs is not installed."
fi

extraction_args=(
  uv run --no-sync python scripts/extract_activations.py
  "--ckpt_path=${checkpoint}"
  "--output_dir=${output_dir}"
  "--hooks=${hooks}"
  "--max_batches=${max_batches}"
  "--sample_config_path=configs/sample.run.fragment"
  "--n_tensor_parallel=${DEFAULT_TENSOR_PARALLEL}"
  "--eval_data_path=${dataset_parquet}"
  "--max_sequence_length=${DEFAULT_SEQUENCE_LENGTH}"
)
soft_h_package_sha256=""
if [[ "${online_entropy}" == true ]]; then
  soft_h_package_sha256="$(
    uv run --project "${SOFT_H_REPO}" python "${package_script}" \
      --soft-h-repo "${SOFT_H_REPO}" \
      --hash-only
  )"
  extraction_args+=(
    "--online_entropy=True"
    "--n_bins=${n_bins}"
    "--seed=${seed}"
    "--soft_h_package_sha256=${soft_h_package_sha256}"
  )
  container_extraction_args=(
    env "PYTHONPATH=/app/${RUNTIME_PACKAGE_RELATIVE}"
    "${extraction_args[@]}"
  )
else
  container_extraction_args=("${extraction_args[@]}")
fi
printf -v extraction_command '%q ' "${container_extraction_args[@]}"

cat <<EOF
Tax activation extraction
  checkpoint : ${checkpoint}
  output     : ${output_dir}
  hooks      : ${hooks}
  input      : ${dataset_parquet} (Fax Drydock eval loader)
  batches    : ${max_batches}
  mode       : $([[ "${online_entropy}" == true ]] && printf 'online entropy' || printf 'raw activation export')
  bins/seed  : ${n_bins}/${seed}
  soft_h sha : ${soft_h_package_sha256:-not packaged}
  max seq    : ${DEFAULT_SEQUENCE_LENGTH}
  tensor par : ${DEFAULT_TENSOR_PARALLEL}
  FSDP       : automatic (2 ways on the requested 8 GPUs)
  context    : ${context}
  queue      : ${queue}
  priority   : ${priority}
  time limit : ${DEFAULT_TIME_LIMIT}
  worker     : 1 node x 8 GPUs (from tax/ops/kjobs-compute.yaml)
  image      : build current Tax checkout with kjobs-generated timestamp tag
  command    : ${extraction_command}
EOF

if [[ "${submit}" != true ]]; then
  printf '\nPreview only. Re-run with --submit after reviewing this specification.\n'
  exit 0
fi

runtime_package_dir=""
cleanup_runtime_package() {
  if [[ -n "${runtime_package_dir}" && -d "${runtime_package_dir}" ]]; then
    rm -r -- "${runtime_package_dir}"
  fi
}
trap cleanup_runtime_package EXIT

if [[ "${online_entropy}" == true ]]; then
  runtime_package_dir="${tax_repo}/${RUNTIME_PACKAGE_RELATIVE}"
  [[ ! -e "${runtime_package_dir}" ]] ||
    die "Refusing to overwrite stale runtime package ${runtime_package_dir}; remove it after inspecting."
  staged_sha="$(
    uv run --project "${SOFT_H_REPO}" python "${package_script}" \
      --soft-h-repo "${SOFT_H_REPO}" \
      --output-dir "${runtime_package_dir}"
  )"
  [[ "${staged_sha}" == "${soft_h_package_sha256}" ]] ||
    die "Staged soft_h package digest changed during launch."
  printf '\nStaged soft_h package %s for the Tax image build.\n' "${staged_sha}"
fi

printf '\nRunning checkpoint, mesh, tokenizer, and C4-batch preflight on CPU...\n'
(
  cd "${tax_repo}"
  if [[ "${online_entropy}" == true ]]; then
    AWS_PROFILE=caios \
      CLOUD_PROVIDER=coreweave \
      FAX_NUMBER_WORKERS=1 \
      FAX_NUMBER_GPUS_PER_WORKER=8 \
      PYTHONPATH="${runtime_package_dir}" \
      "${extraction_args[@]}" \
      --preflight_only=True
  else
    AWS_PROFILE=caios \
      CLOUD_PROVIDER=coreweave \
      FAX_NUMBER_WORKERS=1 \
      FAX_NUMBER_GPUS_PER_WORKER=8 \
      "${extraction_args[@]}" \
      --preflight_only=True
  fi
)

printf '\nAuthenticating to %s...\n' "${context}"
kubectl --context "${context}" auth whoami >/dev/null

export CONTEXT="${context}"
export QUEUE="${queue}"
export PRIORITY_CLASS="${priority}"
export CUSTOM_ARGS="worker.count=1 time_limit=${DEFAULT_TIME_LIMIT}"
export CMD="${extraction_command}"
unset IMAGE_TAG || true

cd "${tax_repo}"
"${kjobs_command}" fax submit
