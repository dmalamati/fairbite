"""Evaluate Gemini's is_categorical output against human annotations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
EXPERT_DIR = EVALUATION_DIR / "expert_annotations"
LLM_DIR = EVALUATION_DIR / "sensitive_attributes_results"
OUTPUT_DIR = EVALUATION_DIR / "evaluation_results" / "categorical_evaluation"

EXPERT_PREFIX = "expert_ann_"
LLM_PREFIX = "sen_attr_"


def load_payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def payload_rows(payload: dict, label_column: str) -> pd.DataFrame:
    rows = []
    for recordset in payload.get("recordsets") or []:
        recordset_name = str(recordset.get("recordset_name") or "").strip()
        for field in recordset.get("results") or []:
            key = field.get("key")
            value = field.get("is_categorical")
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(value, bool):
                raise ValueError(
                    f"Non-Boolean {label_column} for "
                    f"{recordset_name!r}/{key!r}: {value!r}"
                )
            rows.append(
                {
                    "recordset": recordset_name,
                    "field": key.strip(),
                    label_column: value,
                }
            )
    return pd.DataFrame(
        rows, columns=["recordset", "field", label_column]
    )


def binary_metrics(frame: pd.DataFrame) -> dict:
    true = frame["human_is_categorical"]
    predicted = frame["llm_is_categorical"]

    tp = int((true & predicted).sum())
    tn = int((~true & ~predicted).sum())
    fp = int((~true & predicted).sum())
    fn = int((true & ~predicted).sum())
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2

    return {
        "n_fields": total,
        "human_categorical": int(true.sum()),
        "human_non_categorical": int((~true).sum()),
        "predicted_categorical": int(predicted.sum()),
        "predicted_non_categorical": int((~predicted).sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def matching_llm_path(expert_path: Path) -> Path:
    suffix = expert_path.name[len(EXPERT_PREFIX) :]
    return LLM_DIR / f"{LLM_PREFIX}{suffix}"


def evaluate_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    comparisons = []
    per_dataset = []

    expert_paths = sorted(EXPERT_DIR.glob(f"{EXPERT_PREFIX}*.json"))
    if not expert_paths:
        raise RuntimeError(f"No expert annotations found in {EXPERT_DIR}")

    for expert_path in expert_paths:
        llm_path = matching_llm_path(expert_path)
        if not llm_path.exists():
            raise FileNotFoundError(
                f"Missing Gemini output corresponding to {expert_path.name}"
            )

        dataset = expert_path.stem[len(EXPERT_PREFIX) :]
        human = payload_rows(
            load_payload(expert_path), "human_is_categorical"
        )
        llm = payload_rows(load_payload(llm_path), "llm_is_categorical")

        if human.duplicated(["recordset", "field"]).any():
            raise ValueError(f"Duplicate human field keys in {expert_path.name}")
        if llm.duplicated(["recordset", "field"]).any():
            raise ValueError(f"Duplicate Gemini field keys in {llm_path.name}")

        merged = human.merge(
            llm,
            on=["recordset", "field"],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        unmatched = merged[merged["_merge"] != "both"]
        if not unmatched.empty:
            raise ValueError(
                f"Human/Gemini field mismatch in {dataset}: "
                f"{len(unmatched)} unmatched rows"
            )
        merged = merged.drop(columns="_merge")
        merged.insert(0, "dataset", dataset)
        merged["correct"] = (
            merged["human_is_categorical"]
            == merged["llm_is_categorical"]
        )
        comparisons.append(merged)

        metrics = binary_metrics(merged)
        metrics["dataset"] = dataset
        per_dataset.append(metrics)

    comparison = pd.concat(comparisons, ignore_index=True)
    dataset_metrics = pd.DataFrame(per_dataset)
    summary = binary_metrics(comparison)
    summary["datasets"] = len(expert_paths)
    return comparison, dataset_metrics, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison, dataset_metrics, summary = evaluate_all()

    comparison.to_csv(OUTPUT_DIR / "field_comparison.csv", index=False)
    dataset_metrics.to_csv(OUTPUT_DIR / "dataset_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "categorical_metrics.csv", index=False
    )
    (OUTPUT_DIR / "categorical_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    confusion = pd.DataFrame(
        [[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]],
        index=["human_false", "human_true"],
        columns=["predicted_false", "predicted_true"],
    )
    confusion.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    print("Categorical-field evaluation")
    for key in (
        "datasets",
        "n_fields",
        "human_categorical",
        "human_non_categorical",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ):
        value = summary[key]
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nConfusion matrix:")
    print(confusion)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
