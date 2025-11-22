from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

# Configuration
INPUT_FILE = Path("./csv/sim02_complete_data.csv").resolve()
OUTPUT_FOLDER = Path("./csv").resolve()

# Toggle between parametric and non-parametric tests FOR P-VALUES
# Effect sizes always use Cohen's d (standard for compositional data)
USE_NONPARAMETRIC = False  # Set to False to use t-tests for p-values

ILR_COLUMNS = ["ilr1", "ilr2"]
CLR_COLUMNS = ["toe_clr", "angle_clr", "inter_dep_clr"]

# Define comparison pairs
COMPARISON_PAIRS = [
    # Group 3 (g3): Healthy paired comparisons
    ("L-L", "R-R", ["none"], "paired", "g3"),
    ("L-R", "R-L", ["none"], "paired", "g3"),
    ("Same", "Contra", ["none"], "paired", "g3"),
    # Group 1 (g1): Amputee with Mech prosthesis
    ("A-A", "S-S", ["Mech"], "paired", "g1"),
    ("A-S", "S-A", ["Mech"], "paired", "g1"),
    ("A-A", "Same", ["Mech", "none"], "unpaired", "g1"),
    ("S-S", "Same", ["Mech", "none"], "unpaired", "g1"),
    ("A-S", "Contra", ["Mech", "none"], "unpaired", "g1"),
    ("S-A", "Contra", ["Mech", "none"], "unpaired", "g1"),
    # Group 2 (g2): Amputee with Ech prosthesis
    ("A-A", "S-S", ["Ech"], "paired", "g2"),
    ("A-S", "S-A", ["Ech"], "paired", "g2"),
    ("A-A", "Same", ["Ech", "none"], "unpaired", "g2"),
    ("S-S", "Same", ["Ech", "none"], "unpaired", "g2"),
    ("A-S", "Contra", ["Ech", "none"], "unpaired", "g2"),
    ("S-A", "Contra", ["Ech", "none"], "unpaired", "g2"),
]


def cohens_d(x, y):
    """
    Calculate Cohen's d effect size.

    This is the standard effect size for compositional data analysis,
    used regardless of whether parametric or non-parametric tests are used for p-values.
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    )

    if pooled_std == 0:
        return 0.0

    return (np.mean(x) - np.mean(y)) / pooled_std


def get_data(df, class1, class2, is_paired):
    """
    Extract compositional data and ILR coordinates for two classes.

    For paired tests, ensures same subjects are present in both groups
    and data is properly aligned.

    Returns:
        comp1, comp2: Composition arrays (n_samples, 3)
        ilr1, ilr2: ILR coordinate arrays (n_samples, 2)
    """
    # Determine class column
    if class1 in ["L-L", "R-R", "L-R", "R-L"]:
        class_col = "toe_angle"
    else:
        class_col = "amp_sound"

    # Columns to extract
    comp_cols = ILR_COLUMNS + CLR_COLUMNS
    all_cols = ["subject"] + comp_cols

    # Get data
    data1 = df[df[class_col] == class1][all_cols].dropna()
    data2 = df[df[class_col] == class2][all_cols].dropna()

    # Average by subject (only used for Same and Contra conditions)
    data1 = data1.groupby("subject")[comp_cols].mean()
    data2 = data2.groupby("subject")[comp_cols].mean()

    if is_paired:
        # CRITICAL Keep only common subjects and ensure alignment
        common_subjects = data1.index.intersection(data2.index)

        # Sort by subject to ensure alignment
        data1 = data1.loc[common_subjects].sort_index()
        data2 = data2.loc[common_subjects].sort_index()

        # Verify alignment
        assert all(data1.index == data2.index), (
            "Subject indices don't match after sorting!"
        )

    # Split compositions and ILR
    comp1 = data1[comp_cols]
    comp2 = data2[comp_cols]

    return comp1, comp2


def omnibus_test(comp1, comp2, is_paired):
    """
    Perform multivariate test (Hotelling's T²) on ILR coordinates.

    This is the omnibus test asking: "Does the composition differ?"
    """
    comp1_ilr = comp1[ILR_COLUMNS].values
    comp2_ilr = comp2[ILR_COLUMNS].values

    try:
        if is_paired:
            result = pg.multivariate_ttest(comp1_ilr, comp2_ilr, paired=True)
        else:
            result = pg.multivariate_ttest(comp1_ilr, comp2_ilr, paired=False)
        pval = result["pval"].values[0]
        T2 = result["T2"].values[0]  # Hotelling's T² statistic

        # Calculate Mahalanobis distance (D)
        n1 = len(comp1_ilr)
        n2 = len(comp1_ilr) if not is_paired else n1

        if is_paired:
            # For paired data
            D = np.sqrt(T2 / n1)
        else:
            # For unpaired data
            D = np.sqrt(T2 * (n1 + n2 - 2) / (n1 * n2))

        return {
            "pval": pval,
            "T2": T2,
            "mahalanobis_D": D,
        }

    except Exception as e:
        print(f"Warning: Omnibus test failed - {e}")
        return {"pval": 1.0, "T2": 0.0, "mahalanobis_D": 0.0}


def posthoc_tests(comp1, comp2, is_paired, alpha=0.05, use_nonparametric=False):
    """
    Post-hoc tests to identify which proportions differ.
    Only run if omnibus test is significant (protected testing).

    IMPORTANT:
    - P-values: Use Wilcoxon/Mann-Whitney (if use_nonparametric=True) or t-tests
    - Effect sizes: ALWAYS use Cohen's d (standard for compositional data)

    Rationale: Cohen's d on ILR/CLR coordinates is standard in compositional data
    literature. Non-parametric tests are used for robust p-values with small samples,
    but effect sizes are calculated on the transformed (more normal) data.

    Returns:
        Dictionary with post-hoc results for ILR coordinates and original proportions
    """
    results = {}

    # ===== ILR Coordinate Tests =====
    # Bonferroni correction for 2 ILR coordinates
    alpha_ilr = alpha / 2

    for ilr in ILR_COLUMNS:
        comp1_ilr = comp1[ilr].values
        comp2_ilr = comp2[ilr].values
        # ILR1: inter_dependence vs. independent variations
        if use_nonparametric:
            if is_paired:
                _, p_ilr = wilcoxon(comp1_ilr, comp2_ilr, alternative="two-sided")
            else:
                _, p_ilr = mannwhitneyu(comp1_ilr, comp2_ilr, alternative="two-sided")
        else:
            if is_paired:
                _, p_ilr = ttest_rel(comp1_ilr, comp2_ilr)
            else:
                _, p_ilr = ttest_ind(comp1_ilr, comp2_ilr)

        # ALWAYS use Cohen's d for effect size (standard for compositional data)
        d_ilr = cohens_d(comp1_ilr, comp2_ilr)

        results[f"{ilr}_pval"] = p_ilr
        results[f"{ilr}_pval_bonf"] = min(p_ilr * 2, 1.0)
        results[f"{ilr}_cohens_d"] = d_ilr
        results[f"{ilr}_significant"] = p_ilr < alpha_ilr

    # ===== CLR-based tests for interpretability =====
    # Bonferroni correction for 3 proportions
    alpha_clr = alpha / 3

    for clr in CLR_COLUMNS:
        comp1_clr = comp1[clr].values
        comp2_clr = comp2[clr].values
        # P-value: parametric or non-parametric
        if use_nonparametric:
            if is_paired:
                _, p_clr = wilcoxon(comp1_clr, comp2_clr, alternative="two-sided")
            else:
                _, p_clr = mannwhitneyu(comp1_clr, comp2_clr, alternative="two-sided")
        else:
            if is_paired:
                _, p_clr = ttest_rel(comp1_clr, comp2_clr)
            else:
                _, p_clr = ttest_ind(comp1_clr, comp2_clr)

        # Effect size: ALWAYS Cohen's d (standard for compositional data)
        d_clr = cohens_d(comp1_clr, comp2_clr)

        results[f"{clr}_pval"] = p_clr
        results[f"{clr}_pval_bonf"] = min(p_clr * 3, 1.0)
        results[f"{clr}_cohens_d"] = d_clr
        results[f"{clr}_significant"] = p_clr < alpha_clr

    return results


def compute_pvalues():
    """Main function to compute compositional analysis with ILR."""

    test_type_str = (
        "Non-parametric (Wilcoxon/Mann-Whitney)"
        if USE_NONPARAMETRIC
        else "Parametric (t-tests)"
    )
    print(test_type_str)

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    # Verify ILR columns exist
    if "ilr1" not in df.columns or "ilr2" not in df.columns:
        raise ValueError(
            "ERROR: 'ilr1' and 'ilr2' columns not found in input data. "
            "Make sure to run compile_src_data.py first."
        )

    cycles = sorted(df["cycle"].unique())
    all_results = []

    print(f"Processing {len(cycles)} cycles: {', '.join(cycles)}\n")

    for cycle in cycles:
        print(f"Processing cycle: {cycle}")
        df_cycle = df[df["cycle"] == cycle]

        for class1, class2, prosthesis_list, test_type, test_group in COMPARISON_PAIRS:
            # Filter by prosthesis
            df_filtered = df_cycle[df_cycle["prosthesis"].isin(prosthesis_list)]

            # Match subjects for paired comparisons
            is_paired = test_type == "paired"

            # Get compositional data and ILR coordinates (FIXED: with subject matching)
            comp1, comp2 = get_data(df_filtered, class1, class2, is_paired)

            if len(comp1) == 0 or len(comp2) == 0:
                print(f"  Warning: No data for {class1} vs {class2}")
                continue

            # Verify equal sample sizes for paired tests
            if is_paired and len(comp1) != len(comp2):
                print(
                    f"  ERROR: Paired test but unequal sample sizes: {class1}={len(comp1)}, {class2}={len(comp2)}"
                )
                continue

            # ===== OMNIBUS TEST =====
            results_omnibus = omnibus_test(comp1, comp2, is_paired)
            p_omnibus = results_omnibus["pval"]
            T2 = results_omnibus["T2"]
            mahalanobis_d = results_omnibus["mahalanobis_D"]

            if USE_NONPARAMETRIC:
                if is_paired:
                    posthoc_test_type = "wilcoxon"
                else:
                    posthoc_test_type = "mannwhitney"
            else:
                if is_paired:
                    posthoc_test_type = "ttest_rel"
                else:
                    posthoc_test_type = "ttest_ind"

            result = {
                "cycle": cycle,
                "test_group": test_group,
                "test_type": test_type,
                "prosthesis": "-".join(prosthesis_list),
                "pair": f"({class1})-({class2})",
                "n1": len(comp1),
                "n2": len(comp2),
                "omnibus_pval": p_omnibus,
                "omnibus_significant": p_omnibus < 0.05,
                "T2": T2,
                "mahalanobis_D": mahalanobis_d,
                "posthoc_test": posthoc_test_type,
            }

            # ===== POST-HOC TESTS (only if omnibus significant) =====
            if p_omnibus < 0.05:
                posthoc = posthoc_tests(
                    comp1,
                    comp2,
                    is_paired,
                    use_nonparametric=USE_NONPARAMETRIC,
                )
                result.update(posthoc)
            else:
                # Fill with NA if omnibus not significant
                for ilr in ILR_COLUMNS:
                    result.update(
                        {
                            f"{ilr}_pval": np.nan,
                            f"{ilr}_pval_bonf": np.nan,
                            f"{ilr}_cohens_d": np.nan,
                            f"{ilr}_significant": False,
                        }
                    )
                for clr in CLR_COLUMNS:
                    result.update(
                        {
                            f"{clr}_pval": np.nan,
                            f"{clr}_pval_bonf": np.nan,
                            f"{clr}_cohens_d": np.nan,
                            f"{clr}_significant": False,
                        }
                    )

            all_results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    test_suffix = "_nonparam" if USE_NONPARAMETRIC else "_param"
    out_csv_file = OUTPUT_FOLDER / f"sim02_composition_pvals{test_suffix}.csv"
    results_df.to_csv(out_csv_file, index=False)

    return results_df


if __name__ == "__main__":
    results = compute_pvalues()
