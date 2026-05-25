#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ATTACKS = ("pwws", "textfooler", "deepwordbug")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Mail-CAG evaluation CSVs into readable PNG images."
    )
    parser.add_argument("evaluation_dir", help="Path to a run's evaluation folder.")
    parser.add_argument(
        "--output-dir",
        default="result_images",
        help="Output folder name or path. Defaults to evaluation_dir/result_images.",
    )
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = evaluation_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = infer_title_prefix(evaluation_dir)

    for attack in ATTACKS:
        render_cross_heatmap(
            evaluation_dir / f"cross_eval_{attack}_accuracy_matrix.csv",
            output_dir / f"{attack}_eval_plus_adv_cross_accuracy_heatmap.png",
            f"{title_prefix} {attack.upper()}: Clean Eval + Round Adversarial Data",
            "Validation Dataset",
            "Model Round",
        )
        render_cross_heatmap(
            evaluation_dir / f"cross_eval_{attack}_adv_only_accuracy_matrix.csv",
            output_dir / f"{attack}_adv_only_cross_accuracy_heatmap.png",
            f"{title_prefix} {attack.upper()}: Adversarial Data Only",
            "Adversarial Dataset Source Round",
            "Model Round",
        )
        for dataset_kind, filename_part, source_label in [
            ("eval_plus_adv", "eval_plus_adv", "Clean Eval + Round Adversarial Data"),
            ("adv_only", "adv_only", "Adversarial Data Only"),
        ]:
            render_cross_metric_heatmap(
                evaluation_dir / cross_long_filename(attack, dataset_kind),
                "benign_false_positive_rate",
                output_dir / f"{attack}_{filename_part}_benign_fpr_heatmap.png",
                f"{title_prefix} {attack.upper()}: Benign False Positive Rate",
                source_label,
            )
            render_cross_metric_heatmap(
                evaluation_dir / cross_long_filename(attack, dataset_kind),
                "phishing_false_negative_rate",
                output_dir / f"{attack}_{filename_part}_phishing_fnr_heatmap.png",
                f"{title_prefix} {attack.upper()}: Phishing False Negative Rate",
                source_label,
            )
            render_cross_metric_heatmap(
                evaluation_dir / cross_long_filename(attack, dataset_kind),
                "predicted_phishing_share",
                output_dir / f"{attack}_{filename_part}_predicted_phishing_share_heatmap.png",
                f"{title_prefix} {attack.upper()}: Predicted Phishing Share",
                source_label,
            )
        render_table_df(
            own_round_summary(evaluation_dir, attack),
            output_dir / f"{attack}_own_round_summary_table.png",
            f"{title_prefix} {attack.upper()}: Own-Round Evaluation Summary",
            [
                "model_round",
                "validation_round",
                "rows",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
                "benign_false_positive_rate",
                "phishing_false_negative_rate",
                "predicted_phishing_share",
            ],
        )

    render_table_df(
        data_composition(evaluation_dir),
        output_dir / "dataset_composition_table.png",
        f"{title_prefix} Evaluation Dataset Composition",
        [
            "attack",
            "round",
            "clean_eval_rows",
            "adv_rows",
            "eval_plus_adv_rows",
            "adv_share",
            "eval_share",
        ],
    )

    render_table(
        evaluation_dir / "evaluation_matrix.csv",
        output_dir / "clean_eval_summary_table.png",
        f"{title_prefix} Clean Evaluation Summary",
        [
            "round",
            "dataset",
            "attack",
            "rows",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "benign_recall",
            "phishing_recall",
            "benign_false_positive_rate",
            "phishing_false_negative_rate",
            "predicted_phishing_share",
            "prediction_phishing_bias",
        ],
    )

    images = sorted(output_dir.glob("*.png"))
    lines = ["# Evaluation Result Images", ""]
    lines.extend(f"- `{image.name}`" for image in images)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(images)} PNG files to {output_dir}")


def render_cross_heatmap(
    csv_path: Path,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not csv_path.exists():
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path, index_col=0)
    df.index = [f"Model {index}" for index in df.index]
    df.columns = ["Clean" if str(column) == "0" else f"Round {column}" for column in df.columns]

    width = max(8, 1.35 * len(df.columns) + 2.5)
    height = max(5, 0.8 * len(df.index) + 2.2)
    plt.figure(figsize=(width, height))
    axis = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Accuracy"},
    )
    axis.set_title(title, pad=14, fontsize=14, weight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def render_cross_metric_heatmap(
    csv_path: Path,
    metric: str,
    output_path: Path,
    title: str,
    source_label: str,
) -> None:
    if not csv_path.exists():
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    if metric not in df.columns or df.empty:
        return

    matrix = df.pivot(
        index="model_round",
        columns="validation_round",
        values=metric,
    )
    matrix.index = [f"Model {index}" for index in matrix.index]
    matrix.columns = [
        "Clean" if str(column) == "0" else f"Round {column}"
        for column in matrix.columns
    ]

    width = max(8, 1.35 * len(matrix.columns) + 2.5)
    height = max(5, 0.8 * len(matrix.index) + 2.2)
    plt.figure(figsize=(width, height))
    axis = sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    axis.set_title(title, pad=14, fontsize=14, weight="bold")
    axis.set_xlabel(source_label)
    axis.set_ylabel("Model Round")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def cross_long_filename(attack: str, dataset_kind: str) -> str:
    if dataset_kind == "adv_only":
        return f"cross_eval_{attack}_adv_only_long.csv"
    return f"cross_eval_{attack}_long.csv"


def render_table(
    csv_path: Path,
    output_path: Path,
    title: str,
    columns: list[str],
) -> None:
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    render_table_df(df, output_path, title, columns)


def render_table_df(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    columns: list[str],
) -> None:
    import matplotlib.pyplot as plt

    existing_columns = [column for column in columns if column in df.columns]
    display = df[existing_columns].copy()
    if display.empty:
        display = pd.DataFrame(
            [["No rows"] + [""] * (len(existing_columns) - 1)],
            columns=existing_columns,
        )
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )

    width = max(9, min(22, 1.25 * len(display.columns) + 2))
    height = max(2.8, 0.45 * len(display) + 1.6)
    figure, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    axis.set_title(title, pad=12, fontsize=14, weight="bold")

    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#26547c")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7fb")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#d5dce8")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def own_round_summary(evaluation_dir: Path, attack: str) -> pd.DataFrame:
    path = evaluation_dir / f"cross_eval_{attack}_adv_only_long.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    summary = df[df["model_round"] == df["validation_round"]].copy()
    return summary.sort_values("model_round")


def data_composition(evaluation_dir: Path) -> pd.DataFrame:
    clean_path = evaluation_dir.parent / "clean_eval.csv"
    clean_rows = len(pd.read_csv(clean_path)) if clean_path.exists() else 0

    rows = []
    for attack in ATTACKS:
        round_dirs = sorted(evaluation_dir.glob("round_*"), key=round_sort_key)
        for round_dir in round_dirs:
            round_number = int(round_dir.name.removeprefix("round_"))
            adv_path = round_dir / f"{attack}_adversarial.csv"
            if not adv_path.exists():
                continue
            adv_rows = len(pd.read_csv(adv_path))
            total_rows = clean_rows + adv_rows
            rows.append(
                {
                    "attack": attack,
                    "round": round_number,
                    "clean_eval_rows": clean_rows,
                    "adv_rows": adv_rows,
                    "eval_plus_adv_rows": total_rows,
                    "adv_share": adv_rows / total_rows if total_rows else 0.0,
                    "eval_share": clean_rows / total_rows if total_rows else 0.0,
                }
            )
    return pd.DataFrame(rows)


def round_sort_key(path: Path) -> int:
    suffix = path.name.removeprefix("round_")
    return int(suffix) if suffix.isdigit() else -1


def infer_title_prefix(evaluation_dir: Path) -> str:
    run_root = evaluation_dir.parent
    experiment_root = run_root.parent.name
    if experiment_root == "model_b_llm_phishing_only":
        return "Model B"
    if experiment_root == "model_b_improved_llm_phishing_only":
        return "Model B Improved"
    if experiment_root == "model_c_llm_both_labels":
        return "Model C"
    if experiment_root == "model_d_llm_both_labels_label_aware":
        return "Model D"
    if experiment_root == "baseline_clean_albert":
        return "Model A"
    return run_root.name


if __name__ == "__main__":
    main()
