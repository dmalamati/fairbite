from sensitive_characteristics_search import process_single_dataset, get_sensitive_categorical_fields
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os


def _prepare_df_with_missing(df: pd.DataFrame, sensitive_cols: List[str]) -> pd.DataFrame:
    """
    Replace NaN/None with a missing token for all sensitive categorical columns.
    This way, missing values are treated as a valid category in analysis and plots.
    """
    df = df.copy()
    for col in sensitive_cols:
        df[col] = df[col].astype("object").where(df[col].notna(), "MISSING")
    return df


def _get_domain_values(df: pd.DataFrame, col: str) -> List[Any]:
    """
    Get the domain of a categorical column.
    """
    values = df[col].dropna().unique().tolist()
    return sorted(values, key=lambda x: str(x))


def _categorize_group(
    count: int,
    total: int,
    num_possible_groups: int,
    under_ratio: float = 0.5,
    over_ratio: float = 2.0,
) -> str:
    """
    Categorize a subgroup based on coverage.

    Returns one of:
      "not_represented", "under_represented",
      "over_represented", "well_represented".
    """
    if total == 0:
        return "not_represented"

    if count == 0:
        return "not_represented"

    actual_share = count / total
    equal_share = 1.0 / num_possible_groups if num_possible_groups > 0 else 0.0

    # Under-represented: very small in absolute or relative terms
    if (equal_share > 0 and actual_share < under_ratio * equal_share):
        return "under_represented"

    # Over-represented: much bigger than equal share and has enough samples
    if equal_share > 0 and actual_share > over_ratio * equal_share:
        return "over_represented"

    return "well_represented"


def _analyze_combinations_for_recordset(
    rs_name: str,
    df: pd.DataFrame,
    sensitive_cols: List[str],
    max_level: int = 2,
    under_ratio: float = 0.5,
    over_ratio: float = 2.0,
) -> Dict[str, Any]:
    """
    For one record set, analyze representation for all
    combinations of sensitive_cols up to max_level.

    Returns a dict:
    {
      "recordset_name": ...,
      "total_rows": N,
      "parameters": {...},
      "levels": {
        "1": [ ... group dicts ... ],
        "2": [ ... ],
        ...
      }
    }

    Each group dict has:
      {
        "attributes": ["race", "sex"],
        "values": ["Hispanic", "Female"],
        "count": 42,
        "proportion": 0.012,
        "equal_share": 0.0625,
        "category": "under_represented"
      }
    """
    N = len(df)
    result = {
        "recordset_name": rs_name,
        "total_rows": N,
        "sensitive_columns": sensitive_cols,
        "parameters": {
            "max_level": max_level,
            "under_ratio": under_ratio,
            "over_ratio": over_ratio,
        },
        "levels": {},
    }

    if N == 0 or not sensitive_cols:
        return result

    # Precompute domains
    domains = {col: _get_domain_values(df, col) for col in sensitive_cols}

    for level in range(1, max_level + 1):
        level_groups: List[Dict[str, Any]] = []

        for attrs in itertools.combinations(sensitive_cols, level):
            attrs = list(attrs)

            # Actual observed group counts for this combination
            group_counts_s = df.groupby(attrs, dropna=False).size()
            group_counts = group_counts_s.to_dict()

            # Domain cartesian product defines all possible groups
            dom_lists = [domains[col] for col in attrs]
            num_possible = 1
            for dl in dom_lists:
                num_possible *= max(1, len(dl))

            equal_share = (
                1.0 / num_possible if num_possible > 0 else 0.0
            )

            for value_tuple in itertools.product(*dom_lists):
                if len(attrs) == 1:
                    lookup_key = value_tuple[0]
                else:
                    lookup_key = tuple(value_tuple)

                count = group_counts.get(lookup_key, 0)

                category = _categorize_group(
                    count=count,
                    total=N,
                    num_possible_groups=num_possible,
                    under_ratio=under_ratio,
                    over_ratio=over_ratio,
                )

                group_info = {
                    "attributes": attrs,
                    "values": [str(v) for v in value_tuple],
                    "count": int(count),
                    "proportion": float(count / N) if N > 0 else 0.0,
                    "equal_share": float(equal_share),
                    "category": category,
                }
                level_groups.append(group_info)

        result["levels"][str(level)] = level_groups

    return result


def _plot_single_attribute_distributions(
    rs_name: str,
    df: pd.DataFrame,
    sensitive_cols: List[str],
    output_dir: str,
):
    """
    For each single sensitive attribute in the record set:
      - Plot a pie chart of its distribution.
      - Plot a bar chart comparing actual vs equal-share distribution.

    Saves PNG files into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    N = len(df)
    if N == 0:
        return

    for col in sensitive_cols:
        series = df[col].astype("object")
        counts = series.value_counts(dropna=False).sort_index()
        labels = [str(x) for x in counts.index.tolist()]
        values = counts.values
        total = values.sum()

        # Pie chart
        plt.figure()
        plt.title(f"{rs_name} – {col} (pie)")
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        pie_path = os.path.join(output_dir, f"{rs_name}__{col}__pie.png")
        plt.savefig(pie_path, bbox_inches="tight")
        plt.close()

        # Bar chart: actual vs equal-share
        num_categories = len(labels)
        if num_categories == 0:
            continue

        actual_props = values / total
        equal_prop = 1.0 / num_categories
        equal_props = [equal_prop] * num_categories

        x = range(num_categories)
        width = 0.4

        plt.figure()
        plt.title(f"{rs_name} – {col} (actual vs equal-share)")
        plt.bar([i - width / 2 for i in x], actual_props, width=width, label="Actual")
        plt.bar([i + width / 2 for i in x], equal_props, width=width, label="Equal-share")
        plt.xticks(list(x), labels, rotation=45, ha="right")
        plt.ylabel("Proportion of dataset")
        plt.legend()
        bar_path = os.path.join(output_dir, f"{rs_name}__{col}__bar.png")
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()


def run_representation_audit(
    dataset_report: dict,
    dfs: Dict[str, pd.DataFrame],
    sensitivity_threshold: int = 60,
    max_level: int = 2,
    under_ratio: float = 0.5,
    over_ratio: float = 2.0,
    plots_output_dir: str = "representation_plots",
) -> Dict[str, Any]:
    """
    High-level function that ties all steps of the FairBite pipeline.

    Inputs:
      - dataset_report: output of process_single_dataset (first element).
      - dfs: dict[str, pd.DataFrame], output of process_single_dataset (second element).
      - sensitivity_threshold: minimum sensitivity percentage to consider a column.
      - max_level: maximum combination size.
      - under_ratio, over_ratio: relative thresholds vs equal-share.
      - plots_output_dir: directory where plots will be saved.

    Returns:
      {
        "dataset_name": ...,
        "recordsets": [
          {
            "recordset_name": ...,
            "representation": <output of _analyze_combinations_for_recordset>
          },
          ...
        ]
      }
    """
    dataset_name = dataset_report.get("dataset_name", "Unknown dataset")

    # Get sensitive categorical fields over the given threshold
    sensitive_map = get_sensitive_categorical_fields(
        dataset_report,
        sensitivity_threshold=sensitivity_threshold,
    )

    audit = {
        "dataset_name": dataset_name,
        "parameters": {
            "sensitivity_threshold": sensitivity_threshold,
            "max_level": max_level,
            "under_ratio": under_ratio,
            "over_ratio": over_ratio,
        },
        "recordsets": [],
    }

    for rs_entry in dataset_report.get("recordsets", []):
        rs_name = rs_entry.get("recordset_name")
        if not rs_name:
            continue

        rs_df = dfs.get(rs_name)
        if rs_df is None:
            continue

        sensitive_cols = sensitive_map.get(rs_name, [])
        if not sensitive_cols:
            continue

        # Cap max_level to the number of sensitive attributes for this recordset
        effective_max_level = min(max_level, len(sensitive_cols))

        # Restrict df to sensitive columns only
        df_sens = rs_df[sensitive_cols].copy()

        df_sens = _prepare_df_with_missing(df_sens, sensitive_cols)

        rep_info = _analyze_combinations_for_recordset(
            rs_name=rs_name,
            df=df_sens,
            sensitive_cols=sensitive_cols,
            max_level=effective_max_level,
            under_ratio=under_ratio,
            over_ratio=over_ratio,
        )

        # Single-level plots
        # _plot_single_attribute_distributions(
        #     rs_name=rs_name,
        #     df=df_sens,
        #     sensitive_cols=sensitive_cols,
        #     output_dir=plots_output_dir,
        # )

        audit["recordsets"].append(
            {
                "recordset_name": rs_name,
                "representation": rep_info,
            }
        )

    return audit


def save_representation_to_csv(rep_audit: dict, csv_path: str) -> None:
    """
    Flatten the representation audit JSON into a CSV file.

    Each row corresponds to one subgroup:
      - dataset_name
      - recordset_name
      - level (1,2,...)
      - attributes (pipe-joined, e.g. "race|gender")
      - values (pipe-joined, e.g. "White|Female")
      - count
      - proportion
      - equal_share
      - category
    """
    rows = []
    dataset_name = rep_audit.get("dataset_name", "Unknown dataset")

    for rs_entry in rep_audit.get("recordsets", []):
        rs_name = rs_entry.get("recordset_name")
        rep_info = rs_entry.get("representation", {})
        levels = rep_info.get("levels", {})

        for level_str, groups in levels.items():
            try:
                level = int(level_str)
            except ValueError:
                continue

            for g in groups:
                attrs = g.get("attributes", [])
                vals = g.get("values", [])
                row = {
                    "dataset_name": dataset_name,
                    "recordset_name": rs_name,
                    "level": level,
                    "attributes": "|".join(str(a) for a in attrs),
                    "values": "|".join(str(v) for v in vals),
                    "count": int(g.get("count", 0)),
                    "proportion": float(g.get("proportion", 0.0)),
                    "equal_share": float(g.get("equal_share", 0.0)),
                    "category": str(g.get("category", "")),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved representation table to {csv_path}")


def plot_intersection_pie_from_csv(
    csv_path: str,
    recordset_name: str,
    attributes: List[str],
    dataset_name: Optional[str] = None,
    missing_label: str = "Missing",
    missing_token: str = "__MISSING__",
) -> None:
    """
    Plot a pie chart for a given combination of attributes, using the saved CSV.

    - If the requested level of intersectionality (len(attributes)) was never computed
      for this recordset, print "Haven't calculated this level of intersectionality".
    - If the level exists but that specific attribute combination isn't present,
      print a more specific message.
    - Otherwise, plot a pie chart of counts for all value-combinations of those attributes.
    """
    df = pd.read_csv(csv_path)

    # Optional: filter by dataset name if provided
    if dataset_name is not None:
        df = df[df["dataset_name"] == dataset_name]

    # Filter by recordset
    df = df[df["recordset_name"] == recordset_name]

    if df.empty:
        print(f"No data found for recordset '{recordset_name}' in {csv_path}.")
        return

    # Desired level of intersectionality
    level = len(attributes)
    attr_key = "|".join(attributes)

    df_level = df[df["level"] == level]

    if df_level.empty:
        print("Haven't calculated this level of intersectionality "
              f"(requested level={level}).")
        return

    df_combo = df_level[df_level["attributes"] == attr_key]

    if df_combo.empty:
        print(f"No subgroup data for attribute combination {attributes} "
              f"in recordset '{recordset_name}'.")
        return

    # Build labels and counts
    labels_raw = df_combo["values"].tolist()
    counts = df_combo["count"].tolist()

    # Split pipe-joined values into nicer labels
    # Example: "White|Female" -> "White, Female"
    labels = []
    for v in labels_raw:
        parts = str(v).split("|")
        pretty_parts = [
            missing_label if p == missing_token else p
            for p in parts
        ]
        labels.append(", ".join(pretty_parts))

    total = sum(counts)
    if total == 0:
        print(f"All groups for {attributes} have zero count; nothing to plot.")
        return

    # Plot pie chart
    plt.figure()
    title = f"{recordset_name} – {', '.join(attributes)} (level {level})"
    plt.title(title)
    plt.pie(counts, labels=labels, autopct="%1.1f%%")
    plt.tight_layout()
    plt.show()
