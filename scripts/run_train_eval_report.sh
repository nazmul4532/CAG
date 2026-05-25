#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_train_eval_report.sh <config> <run_id> [options]

Examples:
  scripts/run_train_eval_report.sh model-b b_smoke_qwen8b_quality_002
  scripts/run_train_eval_report.sh b-improved b_improved_qwen8b_001
  scripts/run_train_eval_report.sh model-c c_smoke_qwen8b_quality_002
  scripts/run_train_eval_report.sh model-d d_smoke_qwen8b_label_aware_001
  scripts/run_train_eval_report.sh model-c c_smoke_qwen8b_quality_002 --resume
  scripts/run_train_eval_report.sh model-c c_debug_001 --max-examples 10

Options:
  --resume          Resume an existing training run.
  --skip-train      Do not run training/resume.
  --skip-eval       Do not run evaluation.
  --skip-images     Do not render PNG result images.
  --max-examples N  Limit TextAttack examples per attack/round for smoke checks.
  --attacks LIST    Comma-separated attacks. Default: pwws,textfooler,deepwordbug.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

config="$1"
run_id="$2"
shift 2

resume=0
skip_train=0
skip_eval=0
skip_images=0
max_examples=""
attacks_csv="pwws,textfooler,deepwordbug"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      resume=1
      shift
      ;;
    --skip-train)
      skip_train=1
      shift
      ;;
    --skip-eval)
      skip_eval=1
      shift
      ;;
    --skip-images)
      skip_images=1
      shift
      ;;
    --max-examples)
      max_examples="${2:-}"
      if [[ -z "$max_examples" ]]; then
        echo "--max-examples requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --attacks)
      attacks_csv="${2:-}"
      if [[ -z "$attacks_csv" ]]; then
        echo "--attacks requires a comma-separated value" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$config" in
  model-a|baseline)
    eval_dir="runs/baseline_clean_albert/${run_id}/evaluation"
    ;;
  model-b|phishing-only)
    eval_dir="runs/model_b_llm_phishing_only/${run_id}/evaluation"
    ;;
  b-improved|model-b-improved)
    eval_dir="runs/model_b_improved_llm_phishing_only/${run_id}/evaluation"
    ;;
  model-c|both-labels|cyclic|v5)
    eval_dir="runs/model_c_llm_both_labels/${run_id}/evaluation"
    ;;
  model-d|label-aware)
    eval_dir="runs/model_d_llm_both_labels_label_aware/${run_id}/evaluation"
    ;;
  *)
    echo "Unsupported config alias for automated path resolution: $config" >&2
    echo "Use model-a, model-b, b-improved, model-c, or model-d." >&2
    exit 2
    ;;
esac

IFS=',' read -r -a attacks <<< "$attacks_csv"

if [[ "$skip_train" -eq 0 ]]; then
  train_args=(run "$config" --run-id "$run_id")
  if [[ "$resume" -eq 1 ]]; then
    train_args+=(--resume)
  fi
  echo "==> Training: python mail_cag.py ${train_args[*]}"
  python mail_cag.py "${train_args[@]}"
fi

if [[ "$skip_eval" -eq 0 ]]; then
  eval_args=(evaluate "$config" --run-id "$run_id" --all-rounds --generate-adversarial --attacks "${attacks[@]}")
  if [[ -n "$max_examples" ]]; then
    eval_args+=(--max-examples "$max_examples")
  fi
  echo "==> Evaluation: python mail_cag.py ${eval_args[*]}"
  python mail_cag.py "${eval_args[@]}"
fi

if [[ "$skip_images" -eq 0 ]]; then
  echo "==> Rendering images: python scripts/render_evaluation_images.py $eval_dir"
  python scripts/render_evaluation_images.py "$eval_dir"
fi

echo "==> Done"
echo "Evaluation folder: $eval_dir"
echo "Image folder: $eval_dir/result_images"
