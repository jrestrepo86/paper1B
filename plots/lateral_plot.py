"""
Laterality Index (CLI) - Coordination Laterality based on Inter-dependence Proportion
Author: Analysis Script
Date: 2024

Coordination Laterality Index (CLI):
CLI = (inter_dep_prop_RR - inter_dep_prop_LL) × 100

Positive values: Right limb has higher coordination proportion
Negative values: Left limb has higher coordination proportion

Statistical testing: Paired Wilcoxon signed-rank test (H0: CLI = 0)
Effect size: Cohen's d for paired differences
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
CONFIG_FILE = Path("../config.yml")
DATA_FILE = Path("../data/csv/sim02_complete_data.csv")

# Phase order for consistent display
PHASE_ORDER = ["stance", "swing", "cycle"]
PHASE_DISPLAY = {"stance": "Stance", "swing": "Swing", "cycle": "Complete Cycle"}
METRIC = "inter_dependence_prop"  # Using inter-dependence proportion

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_config():
    """Load configuration from YAML file"""
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config


def calculate_cli(data, metric="inter_dependence_prop"):
    """
    Calculate Coordination Laterality Index (CLI)

    CLI = (inter_dep_prop_RR - inter_dep_prop_LL) × 100

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the gait data (healthy subjects only)
    metric : str
        Metric to use (default: 'inter_dependence_prop')

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: subject, phase, CLI, RR, LL
    """
    results = []

    phases = data["cycle"].unique()

    for phase in phases:
        phase_data = data[data["cycle"] == phase].copy()

        # Get R-R and L-L data
        rr_data = phase_data[phase_data["toe_angle"] == "R-R"][
            ["subject", metric]
        ].copy()
        ll_data = phase_data[phase_data["toe_angle"] == "L-L"][
            ["subject", metric]
        ].copy()

        rr_data.columns = ["subject", "RR"]
        ll_data.columns = ["subject", "LL"]

        # Merge by subject (paired data)
        comparison = pd.merge(rr_data, ll_data, on="subject", how="inner")

        # Calculate CLI as simple difference × 100
        comparison["CLI"] = (comparison["RR"] - comparison["LL"]) * 100
        comparison["phase"] = phase

        results.append(comparison[["subject", "phase", "CLI", "RR", "LL"]])

    return pd.concat(results, ignore_index=True)


def calculate_cohens_d_paired(differences):
    """
    Calculate Cohen's d for paired differences

    d = mean(differences) / std(differences)

    Parameters:
    -----------
    differences : array-like
        Paired differences

    Returns:
    --------
    float : Cohen's d
    """
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    if std_diff == 0:
        return np.nan

    return mean_diff / std_diff


def perform_wilcoxon_test(data_series):
    """
    Perform paired Wilcoxon signed-rank test against zero

    H0: median(CLI) = 0
    H1: median(CLI) ≠ 0

    Parameters:
    -----------
    data_series : pd.Series or array-like
        CLI values

    Returns:
    --------
    dict with test results
    """
    # Wilcoxon signed-rank test (paired, testing against 0)
    statistic, p_value = stats.wilcoxon(data_series, alternative="two-sided")

    # Calculate Cohen's d
    cohens_d = calculate_cohens_d_paired(data_series)

    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_size = "negligible"
    elif abs_d < 0.5:
        effect_size = "small"
    elif abs_d < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    # Significance
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "ns"

    return {
        "statistic": statistic,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size": effect_size,
        "significance": sig,
    }


def get_statistics(data_series):
    """Calculate descriptive statistics"""
    stats_dict = {
        "mean": data_series.mean(),
        "median": data_series.median(),
        "std": data_series.std(),
        "q1": data_series.quantile(0.25),
        "q3": data_series.quantile(0.75),
        "n": len(data_series),
        "n_positive": (data_series > 0).sum(),
        "n_negative": (data_series < 0).sum(),
        "pct_positive": (data_series > 0).sum() / len(data_series) * 100,
    }

    return stats_dict


def create_cli_figure(cli_data, config):
    """
    Create figure with boxplot and statistics table

    Parameters:
    -----------
    cli_data : pd.DataFrame
        DataFrame with CLI data
    config : dict
        Configuration dictionary from YAML

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    lat_config = config["laterality-plot"]
    box_config = lat_config["boxplot"]
    table_config = lat_config["table"]

    # Create subplots: boxplot in row1-col1, table in row1-col2
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Coordination Laterality Index by Phase",
            "Statistical Summary",
        ),
        column_widths=[0.40, 0.60],
        horizontal_spacing=0.1,
        specs=[[{"type": "box"}, {"type": "table"}]],
    )

    # Add boxplots and collect table data
    table_data = []

    for phase in PHASE_ORDER:
        phase_data = cli_data[cli_data["phase"] == phase]["CLI"]

        if len(phase_data) > 0:
            # Descriptive statistics
            stats_desc = get_statistics(phase_data)

            # Wilcoxon test and Cohen's d
            test_results = perform_wilcoxon_test(phase_data)

            # Add boxplot
            fig.add_trace(
                go.Box(
                    y=phase_data,
                    name=PHASE_DISPLAY[phase],
                    marker=dict(
                        color=box_config["colors"][phase],
                        size=box_config["marker-size"],
                        opacity=box_config["marker-opacity"],
                    ),
                    line=dict(
                        # color=box_config["colors"][phase],
                        color=box_config["colors"]["lines"],
                        width=box_config["line-width"],
                    ),
                    boxmean="sd",  # Show mean and std dev
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "CLI: %{y:.2f}%<br>"
                    + "<extra></extra>",
                    hoverinfo="text",
                    boxpoints="all",  # Show all points
                    jitter=box_config["jitter"],
                    pointpos=box_config["pointpos"],
                    line_width=box_config["line-width"],
                    fillcolor=box_config["colors"][phase],
                    opacity=box_config["box-fillalpha"],
                    width=box_config["box-width"],
                ),
                row=1,
                col=1,
            )

            # Collect table data
            table_data.append(
                [
                    PHASE_DISPLAY[phase],
                    f"{stats_desc['median']:.2f}",
                    f"[{stats_desc['q1']:.2f}, {stats_desc['q3']:.2f}]",
                    f"{test_results['p_value']:.4f} ({test_results['significance']})",
                    # test_results["significance"],
                    f"{test_results['cohens_d']:.3f} ({test_results['effect_size']})",
                    f"{stats_desc['pct_positive']:.1f}",
                ]
            )

    # Add horizontal line at y=0 (no laterality)
    fig.add_hline(
        y=0,
        line_dash=box_config["zeroline-dash"],
        line_color=box_config["zeroline-color"],
        line_width=box_config["zeroline-width"],
        row=1,
        col=1,
    )

    # Create statistics table
    if len(table_data) > 0:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "<b>Phase</b>",
                        "<b>Median (%)</b>",
                        "<b>IQR (%)</b>",
                        "<b>p-value</b>",
                        # "<b>Sig</b>",
                        "<b>Cohen's d</b>",
                        # "<b>Effect</b>",
                        "<b>% Positive</b>",
                    ],
                    fill_color=table_config["header-bgcolor"],
                    font=dict(
                        color=table_config["header-fontcolor"],
                        size=table_config["header-fontsize"],
                        family=lat_config["subtitle-font-family"],
                    ),
                    align="center",
                    height=table_config["cells-height"],
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[table_config["cells-bgcolor"] * len(table_data)],
                    align="center",
                    font=dict(
                        size=table_config["cells-fontsize"],
                        family=lat_config["subtitle-font-family"],
                    ),
                    height=table_config["cells-height"],
                ),
            ),
            row=1,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "Coordination Laterality Index (CLI)<br>",
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": lat_config["title-font-size"],
                "family": lat_config["title-font-family"],
                "color": lat_config["title-text-color"],
            },
        },
        height=lat_config["export-html"]["height"],
        width=lat_config["export-html"]["width"],
        font=dict(
            size=lat_config["subtitle-font-size"],
            family=lat_config["subtitle-font-family"],
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    # Update axes
    fig.update_xaxes(
        title_text="",
        title_font=dict(
            size=box_config["xaxis-title-size"],
            family=lat_config["subtitle-font-family"],
        ),
        tickfont=dict(
            size=box_config["xaxis-tick-size"],
            family=lat_config["subtitle-font-family"],
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="CLI (%)",
        title_font=dict(
            size=box_config["yaxis-title-size"],
            family=lat_config["subtitle-font-family"],
        ),
        tickfont=dict(
            size=box_config["yaxis-tick-size"],
            family=lat_config["subtitle-font-family"],
        ),
        zeroline=True,
        zerolinewidth=box_config["zeroline-width"],
        zerolinecolor="lightgray",
        gridcolor="lightgray",
        row=1,
        col=1,
    )

    # Update subplot title fonts
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(
            size=lat_config["subtitle-font-size"],
            family=lat_config["subtitle-font-family"],
            color=lat_config["subtitle-text-color"],
        )

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function"""

    print("=" * 80)
    print("COORDINATION LATERALITY INDEX (CLI) ANALYSIS")
    print("CLI = (inter_dep_prop_RR - inter_dep_prop_LL) × 100")
    print("=" * 80)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    print("   ✓ Configuration loaded")

    # Load data
    print("\n2. Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"   Loaded {len(df)} rows")

    # Filter for healthy subjects only
    print("\n3. Filtering for healthy subjects...")
    healthy = df[df["condition"] == "healthy"].copy()
    print(f"   {len(healthy)} rows for healthy subjects")
    print(f"   Unique subjects: {healthy['subject'].nunique()}")
    print(f"   Phases: {healthy['cycle'].unique().tolist()}")

    # Calculate CLI
    print(f"\n4. Calculating CLI using {METRIC}...")
    cli_data = calculate_cli(healthy, metric=METRIC)
    print(f"   Calculated CLI for {len(cli_data)} subject-phase combinations")

    # Display summary statistics with tests
    print("\n5. Statistical Analysis:")
    print("=" * 80)
    print(
        f"{'Phase':<12} {'Mean±SD':<20} {'Median [IQR]':<25} {'p-value':<12} {'d':<8} {'Effect'}"
    )
    print("-" * 80)

    for phase in PHASE_ORDER:
        phase_data = cli_data[cli_data["phase"] == phase]["CLI"]

        if len(phase_data) > 0:
            stats_desc = get_statistics(phase_data)
            test_results = perform_wilcoxon_test(phase_data)

            print(
                f"{PHASE_DISPLAY[phase]:<12} "
                f"{stats_desc['mean']:>6.2f}±{stats_desc['std']:<5.2f}       "
                f"{stats_desc['median']:>6.2f} "
                f"[{stats_desc['q1']:>6.2f}, {stats_desc['q3']:>6.2f}]   "
                f"{test_results['p_value']:>8.4f} {test_results['significance']:<3} "
                f"{test_results['cohens_d']:>6.3f}  "
                f"{test_results['effect_size']}"
            )

    print("=" * 80)
    print("\nInterpretation:")
    print("  • Positive CLI: Right limb shows higher coordination proportion")
    print("  • Negative CLI: Left limb shows higher coordination proportion")
    print("  • p-value: Wilcoxon signed-rank test (H0: CLI = 0)")
    print("  • Cohen's d: Effect size for paired differences")
    print("=" * 80)

    # Create visualization
    print("\n6. Creating visualization...")
    fig = create_cli_figure(cli_data, config)

    # Create output directory
    output_dir = Path("../figs/laterality_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save HTML
    html_file = output_dir / "coordination_laterality_index.html"
    fig.write_html(str(html_file))
    print(f"   ✓ Saved: {html_file}")

    # Save PDF
    try:
        pdf_file = output_dir / "coordination_laterality_index.pdf"
        fig.write_image(
            str(pdf_file),
            width=config["laterality-plot"]["export-pdf"]["width"],
            height=config["laterality-plot"]["export-pdf"]["height"],
            scale=config["laterality-plot"]["export-pdf"]["scale"],
        )
        print(f"   ✓ Saved: {pdf_file}")
    except Exception as e:
        print(f"   ⚠ Could not save PDF: {e}")

    # Save data
    csv_file = output_dir / "coordination_laterality_data.csv"
    cli_data.to_csv(csv_file, index=False)
    print(f"   ✓ Saved: {csv_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nFiles saved to: {output_dir}")
    print("=" * 80)

    return fig, cli_data


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    fig, cli_data = main()
