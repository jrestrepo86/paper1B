from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

# Configuration
INPUT_FILE = Path("./csv/sim02_complete_data.csv").resolve()
OUTPUT_FOLDER = Path("./csv").resolve()

MEASURES = [
    "mi",
    "htoe",
    "hangle",
    "inter_dependence",
    "inter_dependence_prop",
    "toe_var",
    "angle_var",
    "total_var",
]

# Define comparison pairs: (class1, class2, prosthesis_filter, test_type, test_group)
COMPARISON_PAIRS = [
    # Group 1 (g1): Amputee with Mech prosthesis
    ("A-A", "S-S", ["Mech"], "wilcoxon", "g1"),
    ("A-S", "S-A", ["Mech"], "wilcoxon", "g1"),
    ("A-A", "Same", ["Mech", "none"], "mannwhitneyu", "g1"),
    ("S-S", "Same", ["Mech", "none"], "mannwhitneyu", "g1"),
    ("A-S", "Contra", ["Mech", "none"], "mannwhitneyu", "g1"),
    ("S-A", "Contra", ["Mech", "none"], "mannwhitneyu", "g1"),
    # Group 2 (g2): Amputee with Ech prosthesis
    ("A-A", "S-S", ["Ech"], "wilcoxon", "g2"),
    ("A-S", "S-A", ["Ech"], "wilcoxon", "g2"),
    ("A-A", "Same", ["Ech", "none"], "mannwhitneyu", "g2"),
    ("S-S", "Same", ["Ech", "none"], "mannwhitneyu", "g2"),
    ("A-S", "Contra", ["Ech", "none"], "mannwhitneyu", "g2"),
    ("S-A", "Contra", ["Ech", "none"], "mannwhitneyu", "g2"),
    # Group 3 (g3): Healthy paired comparisons
    ("L-L", "R-R", ["none"], "wilcoxon", "g3"),
    ("L-R", "R-L", ["none"], "wilcoxon", "g3"),
    ("Same", "Contra", ["none"], "wilcoxon", "g3"),
]


def cohens_d(x, y):
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    )

    if pooled_std == 0:
        return 0.0

    return (np.mean(x) - np.mean(y)) / pooled_std


def perform_statistical_test(x: np.ndarray, y: np.ndarray, test_type: str) -> float:
    """Perform statistical test and return p-value."""
    try:
        if test_type == "wilcoxon":
            _, pval = wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
        else:  # mannwhitneyu
            _, pval = mannwhitneyu(x, y, alternative="two-sided")
        return pval if np.isfinite(pval) else 1.0
    except Exception:
        return 1.0


def get_comparison_data(
    df: pd.DataFrame,
    class1: str,
    class2: str,
    measure: str,
    test_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and prepare data for comparison."""

    # Determine which column contains the class labels
    if class1 in ["L-L", "R-R", "L-R", "R-L"]:
        class_col = "toe_angle"
    else:
        class_col = "amp_sound"

    # Get data for each class
    data1 = df[df[class_col] == class1][["subject", measure]].dropna()
    data2 = df[df[class_col] == class2][["subject", measure]].dropna()

    # Average by subject
    data1 = data1.groupby("subject")[measure].mean()
    data2 = data2.groupby("subject")[measure].mean()

    if test_type == "wilcoxon":
        # Paired test: match subjects
        paired_data = pd.DataFrame({"data1": data1, "data2": data2}).dropna()
        return paired_data["data1"].values, paired_data["data2"].values
    else:
        # Unpaired test: independent samples
        return data1.values, data2.values


def compute_pvalues():
    """Main function to compute p-values with corrections."""

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    cycles = df["cycle"].unique()
    all_results = []

    print(f"Processing {len(cycles)} cycles...")

    for cycle in cycles:
        df_cycle = df[df["cycle"] == cycle]

        # Compute raw p-values for all comparisons and measures
        for class1, class2, prosthesis_list, test_type, test_group in COMPARISON_PAIRS:
            # Filter by prosthesis
            df_filtered = df_cycle[df_cycle["prosthesis"].isin(prosthesis_list)]

            for measure in MEASURES:
                # Get data and compute p-value
                x, y = get_comparison_data(
                    df_filtered, class1, class2, measure, test_type
                )
                pvalue = perform_statistical_test(x, y, test_type)
                cohens = cohens_d(x, y)

                # Store result
                all_results.append(
                    {
                        "cycle": cycle,
                        "test_group": test_group,
                        "test_type": test_type,
                        "prosthesis": "-".join(prosthesis_list),
                        "pair": f"({class1})-({class2})",
                        "metric": measure,
                        "pvalue": pvalue,
                        "cohens_d": cohens,
                    }
                )

    # Create DataFrame with all results
    results_df = pd.DataFrame(all_results)

    print("Applying multiple testing corrections...")

    # Apply corrections within each group (test_group) separately
    corrected_results = []

    for (cycle, test_group, measure), group_data in results_df.groupby(
        ["cycle", "test_group", "metric"]
    ):
        pvalues = group_data["pvalue"].values

        # Apply Holm and Bonferroni corrections
        _, pvals_holm, _, _ = multipletests(pvalues, method="holm")
        _, pvals_bonf, _, _ = multipletests(pvalues, method="bonferroni")

        # Add corrected p-values to the group data
        group_data = group_data.copy()
        group_data["pvalue_holm"] = pvals_holm
        group_data["pvalue_bonferroni"] = pvals_bonf

        corrected_results.append(group_data)

    # Combine all results
    final_df = pd.concat(corrected_results, ignore_index=True)

    # Reorder columns
    column_order = [
        "cycle",
        "test_group",
        "test_type",
        "prosthesis",
        "pair",
        "metric",
        "pvalue",
        "pvalue_holm",
        "pvalue_bonferroni",
        "cohens_d",
    ]
    final_df = final_df[column_order]

    # Sort for readability
    final_df = final_df.sort_values(
        ["cycle", "test_group", "pair", "metric"]
    ).reset_index(drop=True)

    # Save results
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    out_csv_file = OUTPUT_FOLDER / "sim02_metrics_pvals.csv"
    final_df.to_csv(out_csv_file, index=False)

    print(f"\nResults saved to: {out_csv_file}")
    print(f"Total comparisons: {len(final_df)}")
    print(f"Cycles: {', '.join(sorted(final_df['cycle'].unique()))}")
    print(f"Test groups: {', '.join(sorted(final_df['test_group'].unique()))}")


if __name__ == "__main__":
    compute_pvalues()
