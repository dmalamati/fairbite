import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

# The imported backend initializes Gemini at import time, although this plot
# only uses Croissant loading and the local representation audit.
os.environ.setdefault("GEMINI_API_KEY", "not-used-by-compas-heatmap")

from representation_bias_audit import (
    _analyze_combinations_for_recordset,
    _prepare_df_with_missing,
)
from sensitive_characteristics_search import (
    build_recordset_dataframes,
    load_croissant_dataset,
)


# ============================================================
# SETTINGS β€” same COMPAS case-study settings used in the paper
# ============================================================

CROISSANT_URL = (
    "https://www.kaggle.com/datasets/danofer/compass/croissant/download"
)

RECORDSET = "cox-violent-parsed.csv"
ATTRIBUTES = ["sex", "age_cat", "race"]

ALPHA = 0.8
BETA = 2.0

OUTPUT_PDF = Path("compas_intersectional_audit_heatmap.pdf")
OUTPUT_PNG = Path("compas_intersectional_audit_heatmap.png")


# ============================================================
# 1. LOAD COMPAS FROM CROISSANT
# ============================================================

dataset_entry, croissant_dataset, record_sets = load_croissant_dataset(
    CROISSANT_URL
)

if croissant_dataset is None:
    raise RuntimeError(
        f"Could not load the Croissant dataset: {dataset_entry.get('error')}"
    )

dfs = build_recordset_dataframes(croissant_dataset, record_sets)

if RECORDSET not in dfs:
    raise KeyError(
        f"RecordSet '{RECORDSET}' not found.\n"
        f"Available RecordSets: {list(dfs.keys())}"
    )

df = dfs[RECORDSET].copy()

missing_columns = [c for c in ATTRIBUTES if c not in df.columns]
if missing_columns:
    raise KeyError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# 2. RUN THE LEVEL-3 SUBGROUP ANALYSIS
# ============================================================

df_sensitive = df[ATTRIBUTES].copy()
df_sensitive = _prepare_df_with_missing(df_sensitive, ATTRIBUTES)

representation = _analyze_combinations_for_recordset(
    rs_name=RECORDSET,
    df=df_sensitive,
    sensitive_cols=ATTRIBUTES,
    max_level=3,
    under_ratio=ALPHA,
    over_ratio=BETA,
)

level3_groups = representation["levels"]["3"]


# ============================================================
# 3. CONVERT LEVEL-3 OUTPUT TO A DATAFRAME
# ============================================================

category_map = {
    "not_represented": "Absent",
    "under_represented": "Below",
    "well_represented": "Within",
    "over_represented": "Above",
}

rows = []


def clean_category_value(value):
    """Convert byte values and byte-literal strings to readable labels."""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
        if len(text) >= 3 and text[0] == "b" and text[1] in {"'", '"'}:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, bytes):
                    text = parsed.decode("utf-8", errors="replace")
            except (SyntaxError, ValueError):
                pass

    return {"25 - 45": "25\u201345"}.get(text, text)


def format_percentage(proportion):
    percentage = 100 * proportion
    return f"{percentage:.2f}%" if percentage < 0.1 else f"{percentage:.1f}%"


def format_ratio(ratio):
    if 0 < ratio < 0.01:
        return "<0.01\u00d7"
    return f"{ratio:.2f}\u00d7"

for group in level3_groups:
    attrs = group["attributes"]

    # At level 3, retain the sex x age_cat x race combination.
    if set(attrs) != set(ATTRIBUTES):
        continue

    # Make the script robust to attribute order in the audit output.
    values_by_attribute = {
        attribute: clean_category_value(value)
        for attribute, value in zip(attrs, group["values"])
    }

    rows.append(
        {
            "sex": values_by_attribute["sex"],
            "age_cat": values_by_attribute["age_cat"],
            "race": values_by_attribute["race"],
            "count": group["count"],
            "proportion": group["proportion"],
            "equal_share": group["equal_share"],
            "status": category_map[group["category"]],
        }
    )

audit_df = pd.DataFrame(rows)

if audit_df.empty:
    raise RuntimeError(
        "No sex x age_cat x race groups were found in the level-3 audit."
    )


# ============================================================
# 4. USE THE BACKEND'S EQUAL-SHARE REFERENCE
# ============================================================

# Categories and equal-share values now come directly from FairBite.
audit_df["ratio_to_equal_share"] = (
    audit_df["proportion"] / audit_df["equal_share"]
)

equal_share = audit_df["equal_share"].iloc[0]
m = round(1 / equal_share)


# ============================================================
# 5. ORDER CATEGORIES FOR A CLEAN PAPER FIGURE# ============================================================

def ordered_values(values, preferred):
    values = list(pd.unique(values))
    first = [v for v in preferred if v in values]
    remaining = sorted(v for v in values if v not in first)
    return first + remaining


sex_order = ordered_values(audit_df["sex"], ["Female", "Male"])

age_order = ordered_values(
    audit_df["age_cat"],
    ["Less than 25", "25\u201345", "Greater than 45"],
)

race_order = ordered_values(
    audit_df["race"],
    [
        "African-American",
        "Caucasian",
        "Hispanic",
        "Asian",
        "Native American",
        "Native-American",
        "Other",
    ],
)


# ============================================================
# 6. DRAW FACETED HEATMAP
# ============================================================

status_codes = {"Absent": 0, "Below": 1, "Within": 2, "Above": 3}
status_labels = ["Absent", "Below", "Within", "Above"]

# FairBite-inspired colors drawn from the blue, orange, and green layers in
# the project diagram. The softer fills keep the dense labels easy to read.
category_colors = {
    "Absent": "#90A4C3",
    "Below": "#F5C6A5",
    "Within": "#A8DEB7",
    "Above": "#E97B4C",
}
cmap = ListedColormap([category_colors[label] for label in status_labels])
norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)

background_color = "#FFFFFF"
text_color = "#172033"
secondary_text_color = "#5F6B7A"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelcolor": text_color,
        "xtick.color": secondary_text_color,
        "ytick.color": secondary_text_color,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(
    1,
    len(sex_order),
    figsize=(13.2, 5.25),
    sharey=True,
    facecolor=background_color,
)

if len(sex_order) == 1:
    axes = [axes]

last_image = None

for ax, sex in zip(axes, sex_order):
    subset = audit_df[audit_df["sex"] == sex]
    status_matrix = np.full((len(age_order), len(race_order)), np.nan)
    label_matrix = np.empty(status_matrix.shape, dtype=object)

    for i, age in enumerate(age_order):
        for j, race in enumerate(race_order):
            row = subset[
                (subset["age_cat"] == age) & (subset["race"] == race)
            ]

            if row.empty:
                label_matrix[i, j] = ""
                continue

            row = row.iloc[0]
            status_matrix[i, j] = status_codes[row["status"]]

            if row["status"] == "Absent":
                label_matrix[i, j] = "Absent"
            else:
                label_matrix[i, j] = (
                    f"{format_percentage(row['proportion'])}\n"
                    f"{format_ratio(row['ratio_to_equal_share'])}"
                )

    last_image = ax.imshow(
        status_matrix,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="none",
    )
    ax.set_facecolor("white")
    ax.set_title(
        sex,
        fontsize=14,
        fontweight="bold",
        color=text_color,
        pad=13,
    )
    ax.set_xticks(np.arange(len(race_order)))
    ax.set_xticklabels(
        race_order,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )
    ax.set_yticks(np.arange(len(age_order)))
    ax.set_yticklabels(age_order, fontsize=10.5)
    ax.set_xlabel("Race", fontsize=11, labelpad=8)

    # Subtle cell borders improve readability without adding visual weight.
    ax.set_xticks(np.arange(-0.5, len(race_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(age_order), 1), minor=True)
    ax.grid(which="minor", color=background_color, linewidth=2.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(age_order)):
        for j in range(len(race_order)):
            text = label_matrix[i, j]
            if text:
                status = status_matrix[i, j]
                cell_text_color = "white" if status == 3 else text_color
                ax.text(
                    j, i, text, ha="center", va="center",
                    fontsize=9,
                    color=cell_text_color,
                    linespacing=1.3,
                )

axes[0].set_ylabel(
    "Age category",
    fontsize=11,
    labelpad=10,
)


# ============================================================
# 7. COMPACT DISCRETE LEGEND
# ============================================================

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markerfacecolor=category_colors[label],
        markeredgecolor="none",
        markersize=10,
        label=label,
    )
    for label in status_labels
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=4,
    frameon=False,
    handlelength=0.8,
    columnspacing=1.65,
    handletextpad=0.45,
    labelcolor=text_color,
    fontsize=10.5,
)


# ============================================================
# 8. TITLE AND OUTPUT
# ============================================================

plt.subplots_adjust(
    left=0.145,
    right=0.985,
    bottom=0.225,
    top=0.84,
    wspace=0.13,
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor=background_color,
    transparent=False,
)
fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor=background_color,
    transparent=False,
)

print("\nLevel-3 audit summary:")
print(audit_df["status"].value_counts())

print(f"\nEqual-share reference: {equal_share:.6f}")
print(f"Lower boundary: {ALPHA * equal_share:.6f}")
print(f"Upper boundary: {BETA * equal_share:.6f}")

print(f"\nSaved:\n  {OUTPUT_PDF}\n  {OUTPUT_PNG}")

plt.show()
