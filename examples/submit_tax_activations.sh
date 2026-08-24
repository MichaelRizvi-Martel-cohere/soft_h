#!/usr/bin/env bash
#
# Prepare or submit a Tax activation-extraction job from the Cohere login VM.

set -euo pipefail

readonly DEFAULT_CONTEXT="cw-us-east-04-prod"
readonly DEFAULT_QUEUE="post-training-smifs-queue"
readonly DEFAULT_PRIORITY="dev-low"
readonly DEFAULT_HOOKS="block_0_block_output"
readonly DEFAULT_MAX_BATCHES="1"
readonly DEFAULT_N_VALIDATION_STEPS="1000"
readonly DEFAULT_TENSOR_PARALLEL="8"
readonly DEFAULT_OUTPUT_PREFIX="gs://cohere-dev/michael-rizvi/soft_h_tax"
readonly DEFAULT_TAX_REPO="${HOME}/repos/tax"
readonly DEFAULT_TIME_LIMIT="1h"

usage() {
  cat <<'EOF'
Usage:
  submit_tax_activations.sh --checkpoint URI --run-name NAME [options]

Prepare a one-node, 8-H100 Tax activation-extraction job. By default, this
prints the complete job specification without submitting it.

Required:
  --checkpoint URI       Fax checkpoint under s3://us-east-01a/
  --run-name NAME        Unique run name used for the output directory

Options:
  --hooks LIST           Comma-separated hooks (default: block_0_block_output)
  --max-batches N        Number of evaluation batches (default: 1)
  --n-validation-steps N Validation horizon used by the checkpoint loader
                         (default: 1000; extraction still uses --max-batches)
  --output-dir URI       Destination under gs://cohere-dev/
  --context CONTEXT      Kubernetes context (default: cw-us-east-04-prod)
  --queue QUEUE          Kueue queue (default: post-training-smifs-queue)
  --priority PRIORITY    Priority class (default: dev-low)
  --tax-repo PATH        Tax checkout (default: ~/repos/tax)
  --submit               Authenticate and submit the displayed specification
  -h, --help             Show this help

Examples:
  # Preview only.
  examples/submit_tax_activations.sh \
    --checkpoint s3://us-east-01a/promoted-checkpoints/.../ckpt-536 \
    --run-name limpgods_ckpt536_block0

  # Submit after reviewing the preview.
  examples/submit_tax_activations.sh \
    --checkpoint s3://us-east-01a/promoted-checkpoints/.../ckpt-536 \
    --run-name limpgods_ckpt536_block0 \
    --submit
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

checkpoint=""
run_name=""
hooks="${DEFAULT_HOOKS}"
max_batches="${DEFAULT_MAX_BATCHES}"
n_validation_steps="${DEFAULT_N_VALIDATION_STEPS}"
output_dir=""
context="${DEFAULT_CONTEXT}"
queue="${DEFAULT_QUEUE}"
priority="${DEFAULT_PRIORITY}"
tax_repo="${DEFAULT_TAX_REPO}"
submit=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      [[ $# -ge 2 ]] || die "--checkpoint requires a value."
      checkpoint="$2"
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
    --max-batches)
      [[ $# -ge 2 ]] || die "--max-batches requires a value."
      max_batches="$2"
      shift 2
      ;;
    --n-validation-steps)
      [[ $# -ge 2 ]] || die "--n-validation-steps requires a value."
      n_validation_steps="$2"
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
[[ -n "${run_name}" ]] || die "--run-name is required."
[[ "${checkpoint}" == s3://us-east-01a/* ]] || die "--checkpoint must be under s3://us-east-01a/."
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--run-name contains unsupported characters."
[[ "${hooks}" =~ ^[A-Za-z0-9_,]+$ ]] || die "--hooks must be a comma-separated list of hook names."
[[ "${max_batches}" =~ ^[1-9][0-9]*$ ]] || die "--max-batches must be a positive integer."
[[ "${n_validation_steps}" =~ ^[1-9][0-9]*$ ]] || die "--n-validation-steps must be a positive integer."
[[ "${context}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--context contains unsupported characters."
[[ "${queue}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--queue contains unsupported characters."
[[ "${priority}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--priority contains unsupported characters."

if [[ -z "${output_dir}" ]]; then
  output_dir="${DEFAULT_OUTPUT_PREFIX}/${run_name}"
fi
[[ "${output_dir}" == gs://cohere-dev/* ]] || die "--output-dir must be under gs://cohere-dev/."

tax_repo="$(realpath "${tax_repo}")"
[[ -f "${tax_repo}/ops/kjobs-compute.yaml" ]] || die "Tax compute config not found under ${tax_repo}."
[[ -f "${tax_repo}/scripts/extract_activations.py" ]] || die "Tax activation extractor not found under ${tax_repo}."

kjobs_command=""
if command -v kjobs >/dev/null 2>&1; then
  kjobs_command="$(command -v kjobs)"
elif [[ -x "${HOME}/repos/kueue-jobs-cli/jobs.py" ]]; then
  kjobs_command="${HOME}/repos/kueue-jobs-cli/jobs.py"
else
  die "kjobs is not installed."
fi

printf -v extraction_command '%q ' \
  uv run --no-sync python scripts/extract_activations.py \
  "--ckpt_path=${checkpoint}" \
  "--output_dir=${output_dir}" \
  "--hooks=${hooks}" \
  "--max_batches=${max_batches}" \
  "--sample_config_path=configs/sample.run.fragment" \
  "--patch_run_config={\"n_validation_steps\":${n_validation_steps},\"sharding.n_tensor_parallel\":${DEFAULT_TENSOR_PARALLEL}}"

head_command="${extraction_command}& extraction_pid=\$!; "
head_command+="while kill -0 \"\${extraction_pid}\" 2>/dev/null; do "
head_command+="echo '[heartbeat] activation extraction running'; sleep 30; "
head_command+="done; wait \"\${extraction_pid}\""

cat <<EOF
Tax activation extraction
  checkpoint : ${checkpoint}
  output     : ${output_dir}
  hooks      : ${hooks}
  batches    : ${max_batches}
  val steps  : ${n_validation_steps} (loader validation horizon)
  tensor par : ${DEFAULT_TENSOR_PARALLEL} (pre-load patch and sample fragment)
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

printf '\nAuthenticating to %s...\n' "${context}"
kubectl --context "${context}" auth whoami >/dev/null

export CONTEXT="${context}"
export QUEUE="${queue}"
export PRIORITY_CLASS="${priority}"
export CUSTOM_ARGS="worker.count=1 time_limit=${DEFAULT_TIME_LIMIT}"
export CMD="${head_command}"
unset IMAGE_TAG || true

cd "${tax_repo}"
"${kjobs_command}" fax submit
