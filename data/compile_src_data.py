from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FOLDER = Path("/home/jrestrepo/Documents/DatosToeMineAmp").resolve()
# INPUT_FOLDER = Path("/media/Datos/toe-mine-amp/sim02_rep")
OUTPUT_FOLDER = Path("./csv").resolve()


# Recursively list all CSV files
def get_sim_files(folder: Path):
    csv_files = list(folder.rglob("*.csv"))
    sim_files = []
    for f in csv_files:
        if "plot" in f.name:
            continue
        sim_files.append(f)
    return sim_files


def ilr_transform(toe_var_prop, angle_var_prop, inter_dependence_prop):
    """
    Apply Balance ILR transformation for 3-part composition.

    Structure:
    - ILR1: inter_dependence vs. (toe_var + angle_var) independent variations
    - ILR2: toe vs. angle (within independent variations)

    Args:
        composition: array of shape (n_samples, 3) with columns:
                    [toe_var_prop, angle_var_prop, inter_dependence_prop]

    Returns:
        ilr: array of shape (n_samples, 2) with ILR coordinates
    """
    toe_var_prop = toe_var_prop + 1e-10
    angle_var_prop = angle_var_prop + 1e-10
    inter_dependence_prop = inter_dependence_prop + 1e-10

    # Geometric mean of toe and angle (independent variations)
    geom_mean_independent = np.sqrt(toe_var_prop * angle_var_prop)

    # ILR1: inter_dependence vs. independent variations
    ILR1 = np.sqrt(2 / 3) * np.log(inter_dependence_prop / geom_mean_independent)

    # ILR2: toe vs. angle
    ILR2 = np.sqrt(1 / 2) * np.log(angle_var_prop / toe_var_prop)

    return ILR1, ILR2


def ilr_inverse(ilr1, ilr2):
    """
    Inverse ILR transformation to recover 3-part composition.

    Args:
        ilr1, ilr2: arrays or Series with ILR coordinates

    Returns:
        composition: array of shape (n_samples, 3) with columns:
                    [toe_var_prop, angle_var_prop, inter_dependence_prop]
    """
    # Convert to numpy arrays and stack
    ilr1 = np.array(ilr1).reshape(-1, 1)
    ilr2 = np.array(ilr2).reshape(-1, 1)
    ilr_coords = np.column_stack([ilr1, ilr2])  # FIXED: np.concat doesn't exist

    # Contrast matrix V for this specific balance structure
    # Structure: [toe, angle, inter_dependence]
    V = np.array(
        [
            [-np.sqrt(1 / 6), -np.sqrt(1 / 2)],  # toe
            [-np.sqrt(1 / 6), np.sqrt(1 / 2)],  # angle
            [np.sqrt(2 / 3), 0],  # inter_dependence
        ]
    )

    # Transform back: log_coords = ilr_coords @ V.T
    log_coords = ilr_coords @ V.T  # shape: (n_samples, 3)

    # Exponentiate and normalize (closure)
    composition = np.exp(log_coords)
    composition = composition / composition.sum(axis=1, keepdims=True)

    return composition


def clr_transform(toe_var_prop, angle_var_prop, inter_dependence_prop):
    """
    Apply CLR transformation (for post-hoc interpretation).
    Returns one coordinate per proportion.
    """

    log_toe_var = np.log(toe_var_prop + 1e-10)
    log_angle_var = np.log(angle_var_prop + 1e-10)
    log_inter_dep = np.log(inter_dependence_prop + 1e-10)

    # Geometric mean of toe and angle and inter_dependence
    geom_mean = (log_angle_var + log_inter_dep + log_toe_var) / 3

    # CLR1: toe
    toe_clr = log_toe_var - geom_mean
    # CLR2: angle
    angle_clr = log_angle_var - geom_mean
    # CLR3: inter_dependence
    inter_dep_clr = log_inter_dep - geom_mean

    return toe_clr, angle_clr, inter_dep_clr


def compile_df(files: list[Path]) -> pd.DataFrame:
    """
    Aggregates multiple realizations per subject using MEDIAN (robust to outliers).

    Returns subject-level data with one row per unique combination of meta_columns.
    """
    df_orig = pd.concat([pd.read_csv(f) for f in files], axis=0)
    meta_columns = [
        "subject",
        "condition",
        "prosthesis",
        "amp_side",
        "angle",
        "cycle",
        "toe_angle",
        "amp_sound",
    ]
    measures = [
        "mi_remine",
        "htoe",
        "hangle",
        "toe_std",
        "angle_std",
        "cycle_size",
    ]
    df = df_orig.copy()

    # FIXED: Use MEDIAN consistently for both groups (more robust to outliers)
    # Median aggregation over multiple realizations per subject

    # healthy subjects
    healthy = df[df["condition"] == "healthy"].copy()
    healthy_avg = healthy.groupby(
        meta_columns,
        as_index=False,
    )[measures].median(numeric_only=True)

    # amputee subjects
    amputee = df[df["condition"] == "amputee"].copy()
    amputee_avg = amputee.groupby(
        meta_columns,
        as_index=False,
    )[measures].median(numeric_only=True)

    dfc = pd.concat([amputee_avg, healthy_avg], ignore_index=True)

    dfc = dfc.rename(columns={"mi_remine": "mi"})

    dfc.reset_index(drop=True, inplace=True)

    # Calculate derived metrics at SUBJECT LEVEL
    dfc["inter_dependence"] = dfc["mi"]
    dfc["toe_var"] = dfc["htoe"] - dfc["inter_dependence"]
    dfc["angle_var"] = dfc["hangle"] - dfc["inter_dependence"]
    dfc["total_var"] = dfc["htoe"] + dfc["hangle"] - dfc["inter_dependence"]

    dfc["toe_var_prop"] = dfc["toe_var"] / dfc["total_var"]
    dfc["angle_var_prop"] = dfc["angle_var"] / dfc["total_var"]
    dfc["inter_dependence_prop"] = dfc["inter_dependence"] / dfc["total_var"]

    # ILR transform
    ilr1, ilr2 = ilr_transform(
        dfc["toe_var_prop"], dfc["angle_var_prop"], dfc["inter_dependence_prop"]
    )
    dfc["ilr1"] = ilr1
    dfc["ilr2"] = ilr2

    # CLR transform
    clr1, clr2, clr3 = clr_transform(
        dfc["toe_var_prop"], dfc["angle_var_prop"], dfc["inter_dependence_prop"]
    )
    dfc["toe_clr"] = clr1
    dfc["angle_clr"] = clr2
    dfc["inter_dep_clr"] = clr3

    dfc = dfc.fillna("none")

    return dfc


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates subject-level data to group-level descriptive statistics.
    """
    grouping_columns = [
        "condition",
        "prosthesis",
        "amp_side",
        "angle",
        "cycle",
        "toe_angle",
        "amp_sound",
    ]

    # Aggregate ONLY the base measures (NOT the proportions!)
    base_metrics_to_aggregate = [
        "mi",
        "htoe",
        "hangle",
        "toe_var",
        "angle_var",
        "inter_dependence",
        "total_var",
        "ilr1",
        "ilr2",
    ]

    # Use MEDIAN consistently for both groups

    # healthy subjects
    healthy = df[df["condition"] == "healthy"].copy()
    healthy_avg = healthy.groupby(
        grouping_columns,
        as_index=False,
    )[base_metrics_to_aggregate].mean(numeric_only=True)

    # amputee subjects
    amputee = df[df["condition"] == "amputee"].copy()
    amputee_avg = amputee.groupby(
        grouping_columns,
        as_index=False,
    )[base_metrics_to_aggregate].mean(numeric_only=True)

    dfc = pd.concat([amputee_avg, healthy_avg], ignore_index=True)

    # Inverse ILR transform
    proportion_median = ilr_inverse(dfc["ilr1"], dfc["ilr2"])

    # Proportions median
    dfc["toe_var_prop"] = proportion_median[:, 0]
    dfc["angle_var_prop"] = proportion_median[:, 1]
    dfc["inter_dependence_prop"] = proportion_median[:, 2]

    # CLR medians
    clr1, clr2, clr3 = clr_transform(
        dfc["toe_var_prop"], dfc["angle_var_prop"], dfc["inter_dependence_prop"]
    )
    dfc["toe_clr"] = clr1
    dfc["angle_clr"] = clr2
    dfc["inter_dep_clr"] = clr3

    dfc.reset_index(drop=True, inplace=True)

    return dfc


def compile_data_from_sim_results(input_folder: Path, output_folder: Path):
    """
    Main pipeline function:
    1. Aggregates multiple realizations per subject (compile_df)
    2. Calculates group-level descriptive statistics (calculate_metrics)
    """
    sim_files = get_sim_files(input_folder)
    df_measures = compile_df(sim_files)
    df_metrics = calculate_metrics(df_measures)

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Write subject-level data (used for statistical testing)
    df_measures.to_csv(output_folder / "sim02_complete_data.csv", index=False)
    # Get group-level descriptive metrics
    df_metrics.to_csv(output_folder / "sim02_metrics.csv", index=False)

    print(
        f"✓ Subject-level data saved: {output_folder / 'sim02_complete_data.csv'} ({len(df_measures)} rows)"
    )
    print(
        f"✓ Group-level metrics saved: {output_folder / 'sim02_metrics.csv'} ({len(df_metrics)} rows)"
    )

    # Verify proportions sum to 1 in group-level data
    df_metrics["prop_sum"] = (
        df_metrics["toe_var_prop"]
        + df_metrics["angle_var_prop"]
        + df_metrics["inter_dependence_prop"]
    )
    max_deviation = abs(df_metrics["prop_sum"] - 1.0).max()
    print(f"✓ Proportion sum check: max deviation from 1.0 = {max_deviation:.2e}")


if __name__ == "__main__":
    compile_data_from_sim_results(INPUT_FOLDER, OUTPUT_FOLDER)
