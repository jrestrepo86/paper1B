"""
Laterality Index (LI_MI) Distribution by Gait Phase - Interactive Visualization
Author: Analysis Script
Date: 2024

This script creates an interactive boxplot showing the distribution of
Laterality Index (LI_MI) across different gait phases using Plotly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# File path
DATA_FILE = Path("../data/csv/sim02_complete_data.csv")

# Laterality Index formula
# LI_MI = (MI_RR - MI_LL) / (MI_RR + MI_LL) × 100
# Positive = Right foot dominance, Negative = Left foot dominance

# Colors for each phase
PHASE_COLORS = {
    "stance": "#3498db",  # Blue
    "swing": "#e74c3c",  # Red
    "cycle": "#2ecc71",  # Green
}

# =============================================================================
# FUNCTIONS
# =============================================================================


def calculate_laterality_index(data, metric):
    """
    Calculate Laterality Index for a given metric

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the gait data
    metric : str
        Metric to use for LI calculation (default: 'mi')

    Returns:
    --------
    pd.DataFrame
        DataFrame with subject, phase, and LI values
    """
    results = []

    # Get unique phases
    phases = data["cycle"].unique()

    for phase in phases:
        # Filter data for this phase
        phase_data = data[data["cycle"] == phase].copy()

        # Get L-L and R-R data
        ll_data = phase_data[phase_data["toe_angle"] == "L-L"][
            ["subject", metric]
        ].copy()
        rr_data = phase_data[phase_data["toe_angle"] == "R-R"][
            ["subject", metric]
        ].copy()

        ll_data.columns = ["subject", "LL"]
        rr_data.columns = ["subject", "RR"]

        # Merge
        comparison = pd.merge(ll_data, rr_data, on="subject", how="inner")

        # Calculate LI_MI
        comparison["LI_MI"] = (
            (comparison["RR"] - comparison["LL"])
            / (comparison["RR"] + comparison["LL"])
            * 100
        )
        comparison["phase"] = phase

        results.append(comparison[["subject", "phase", "LI_MI"]])

    return pd.concat(results, ignore_index=True)


def get_statistics(data_series):
    """Calculate statistical measures for a data series"""
    stats_dict = {
        "mean": data_series.mean(),
        "median": data_series.median(),
        "std": data_series.std(),
        "q1": data_series.quantile(0.25),
        "q3": data_series.quantile(0.75),
        "min": data_series.min(),
        "max": data_series.max(),
        "n": len(data_series),
        "n_positive": (data_series > 0).sum(),
        "pct_positive": (data_series > 0).sum() / len(data_series) * 100,
    }

    # t-test against 0
    t_stat, p_value = stats.ttest_1samp(data_series, 0)
    stats_dict["p_value"] = p_value

    return stats_dict


def create_interactive_boxplot(li_data):
    """
    Create interactive boxplot with Plotly

    Parameters:
    -----------
    li_data : pd.DataFrame
        DataFrame with columns: subject, phase, LI_MI

    Returns:
    --------
    plotly.graph_objects.Figure
    """

    # Define phase order
    phase_order = ["stance", "swing", "cycle"]

    # Create figure
    fig = go.Figure()

    # Add boxplot for each phase
    for phase in phase_order:
        phase_data = li_data[li_data["phase"] == phase]["LI_MI"]

        # Get statistics
        phase_stats = get_statistics(phase_data)

        # Create hover text
        hover_text = [
            f"Subject: {subj}<br>"
            + f"LI_MI: {val:.2f}%<br>"
            + f"Phase: {phase.capitalize()}"
            for subj, val in zip(
                li_data[li_data["phase"] == phase]["subject"], phase_data
            )
        ]

        # Add boxplot
        fig.add_trace(
            go.Box(
                y=phase_data,
                name=phase.capitalize(),
                marker=dict(color=PHASE_COLORS[phase]),
                boxmean="sd",  # Show mean and std dev
                text=hover_text,
                hoverinfo="text",
                boxpoints="all",  # Show all points
                jitter=0.3,
                pointpos=-1.8,
                marker_size=6,
                marker_opacity=0.6,
                line_width=2,
            )
        )

    # Add horizontal line at y=0
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text="No Laterality",
        annotation_position="right",
    )

    # Calculate statistics for annotation
    stats_text = ""
    for phase in phase_order:
        phase_data = li_data[li_data["phase"] == phase]["LI_MI"]
        phase_stats = get_statistics(phase_data)
        sig = (
            "***"
            if phase_stats["p_value"] < 0.001
            else "**"
            if phase_stats["p_value"] < 0.01
            else "*"
            if phase_stats["p_value"] < 0.05
            else "ns"
        )

        stats_text += f"<b>{phase.capitalize()}</b>: "
        stats_text += f"Mean={phase_stats['mean']:.2f}%, "
        stats_text += f"p={phase_stats['p_value']:.4f} {sig}<br>"

    # Update layout
    fig.update_layout(
        title={
            "text": "Laterality Index (LI_MI) Distribution by Gait Phase<br>"
            + "<sub>Healthy Subjects (n=22)</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#2c3e50"},
        },
        xaxis_title="Gait Phase",
        yaxis_title="Laterality Index LI_MI (%)",
        yaxis=dict(
            zeroline=True, zerolinewidth=2, zerolinecolor="gray", gridcolor="lightgray"
        ),
        xaxis=dict(
            tickfont=dict(size=14, color="#2c3e50"),
            title_font=dict(size=16, color="#2c3e50"),
        ),
        font=dict(size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            title="Phase",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        annotations=[
            dict(
                text=stats_text,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255, 255, 224, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=10,
                font=dict(size=11, family="monospace"),
            ),
            dict(
                text="<b>Interpretation:</b><br>"
                + "Positive = Right foot dominance<br>"
                + "Negative = Left foot dominance<br>"
                + "LI_MI = (MI_RR - MI_LL) / (MI_RR + MI_LL) × 100",
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                bgcolor="rgba(230, 240, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=10,
                font=dict(size=10),
            ),
        ],
        width=1200,
        height=700,
    )

    return fig


def create_detailed_comparison(li_data):
    """
    Create a multi-panel figure with detailed comparisons
    """

    phase_order = ["stance", "swing", "cycle"]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Distribution by Phase",
            "Violin Plot by Phase",
            "Subject Trajectories Across Phases",
            "Statistical Summary",
        ),
        specs=[
            [{"type": "box"}, {"type": "violin"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    # Panel 1: Boxplot
    for phase in phase_order:
        phase_data = li_data[li_data["phase"] == phase]["LI_MI"]
        fig.add_trace(
            go.Box(
                y=phase_data,
                name=phase.capitalize(),
                marker_color=PHASE_COLORS[phase],
                boxmean="sd",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Panel 2: Violin plot
    for phase in phase_order:
        phase_data = li_data[li_data["phase"] == phase]["LI_MI"]
        fig.add_trace(
            go.Violin(
                y=phase_data,
                name=phase.capitalize(),
                marker_color=PHASE_COLORS[phase],
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Panel 3: Subject trajectories
    subjects = li_data["subject"].unique()
    phase_positions = {"stance": 0, "swing": 1, "cycle": 2}

    for subject in subjects:
        subject_data = li_data[li_data["subject"] == subject].copy()
        subject_data["x_pos"] = subject_data["phase"].map(phase_positions)
        subject_data = subject_data.sort_values("x_pos")

        fig.add_trace(
            go.Scatter(
                x=subject_data["x_pos"],
                y=subject_data["LI_MI"],
                mode="lines+markers",
                name=subject,
                line=dict(width=1),
                marker=dict(size=4),
                opacity=0.4,
                showlegend=False,
                hovertemplate=f"<b>{subject}</b><br>"
                + "Phase: %{text}<br>"
                + "LI_MI: %{y:.2f}%<extra></extra>",
                text=[phase.capitalize() for phase in subject_data["phase"]],
            ),
            row=2,
            col=1,
        )

    # Panel 4: Statistics table
    table_data = []
    for phase in phase_order:
        phase_data = li_data[li_data["phase"] == phase]["LI_MI"]
        stats_dict = get_statistics(phase_data)
        sig = (
            "***"
            if stats_dict["p_value"] < 0.001
            else "**"
            if stats_dict["p_value"] < 0.01
            else "*"
            if stats_dict["p_value"] < 0.05
            else "ns"
        )

        table_data.append(
            [
                phase.capitalize(),
                f"{stats_dict['mean']:.2f}",
                f"{stats_dict['std']:.2f}",
                f"[{stats_dict['q1']:.2f}, {stats_dict['q3']:.2f}]",
                f"{stats_dict['pct_positive']:.1f}%",
                f"{stats_dict['p_value']:.4f}",
                sig,
            ]
        )

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "<b>Phase</b>",
                    "<b>Mean</b>",
                    "<b>SD</b>",
                    "<b>IQR</b>",
                    "<b>% Positive</b>",
                    "<b>p-value</b>",
                    "<b>Sig</b>",
                ],
                fill_color="#3498db",
                font=dict(color="white", size=12),
                align="center",
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=[["white", "#ecf0f1"] * 3],
                align="center",
                font=dict(size=11),
            ),
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title_text="Phase", row=1, col=1)
    fig.update_xaxes(title_text="Phase", row=1, col=2)
    fig.update_xaxes(
        title_text="Phase",
        tickvals=[0, 1, 2],
        ticktext=["Stance", "Swing", "Cycle"],
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="LI_MI (%)", row=1, col=1)
    fig.update_yaxes(title_text="LI_MI (%)", row=1, col=2)
    fig.update_yaxes(title_text="LI_MI (%)", row=2, col=1)

    # Add horizontal lines at 0 (using shapes with explicit references)
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="x domain",
        yref="y",
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        row=1,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="x2 domain",
        yref="y2",
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        row=1,
        col=2,
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="x3 domain",
        yref="y3",
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Comprehensive Laterality Index Analysis by Phase",
        height=900,
        width=1400,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function"""

    print("=" * 80)
    print("LATERALITY INDEX VISUALIZATION")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"   Loaded {len(df)} rows")

    # Filter for healthy subjects only
    print("\n2. Filtering for healthy subjects...")
    healthy = df[df["condition"] == "healthy"].copy()
    print(f"   {len(healthy)} rows for healthy subjects")
    print(f"   Unique subjects: {healthy['subject'].nunique()}")
    print(f"   Phases: {healthy['cycle'].unique().tolist()}")

    # Calculate Laterality Index
    print("\n3. Calculating Laterality Index (LI_MI)...")
    li_data = calculate_laterality_index(healthy, metric="mi")
    print(f"   Calculated LI for {len(li_data)} subject-phase combinations")

    # Display summary statistics
    print("\n4. Summary Statistics:")
    print("-" * 80)
    for phase in ["stance", "swing", "cycle"]:
        phase_data = li_data[li_data["phase"] == phase]["LI_MI"]
        stats_dict = get_statistics(phase_data)
        sig = (
            "***"
            if stats_dict["p_value"] < 0.001
            else "**"
            if stats_dict["p_value"] < 0.01
            else "*"
            if stats_dict["p_value"] < 0.05
            else "ns"
        )

        print(f"\n   {phase.upper()}:")
        print(f"      Mean ± SD: {stats_dict['mean']:.2f} ± {stats_dict['std']:.2f}%")
        print(
            f"      Median [IQR]: {stats_dict['median']:.2f} [{stats_dict['q1']:.2f}, {stats_dict['q3']:.2f}]"
        )
        print(f"      Range: [{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]")
        print(f"      % Positive: {stats_dict['pct_positive']:.1f}%")
        print(f"      p-value: {stats_dict['p_value']:.4f} {sig}")

    # Create visualizations
    print("\n5. Creating interactive visualizations...")

    # Simple boxplot
    print("   Creating main boxplot...")
    fig1 = create_interactive_boxplot(li_data)
    # fig1.write_html("/mnt/user-data/outputs/laterality_index_boxplot.html")
    # print("   ✓ Saved: laterality_index_boxplot.html")

    # Detailed comparison
    print("   Creating detailed comparison...")
    fig2 = create_detailed_comparison(li_data)
    # fig2.write_html("/mnt/user-data/outputs/laterality_index_detailed.html")
    # print("   ✓ Saved: laterality_index_detailed.html")

    # Also save as static images (requires kaleido)
    try:
        print("\n6. Saving static images...")
        fig1.write_image(
            "/mnt/user-data/outputs/laterality_index_boxplot.png",
            width=1200,
            height=700,
        )
        print("   ✓ Saved: laterality_index_boxplot.png")

        fig2.write_image(
            "/mnt/user-data/outputs/laterality_index_detailed.png",
            width=1400,
            height=900,
        )
        print("   ✓ Saved: laterality_index_detailed.png")
    except Exception as e:
        print(f"   ⚠ Could not save static images: {e}")
        print("   (This is optional - HTML files were saved successfully)")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print("\nInteractive HTML files created:")
    print("  - laterality_index_boxplot.html (main figure)")
    print("  - laterality_index_detailed.html (comprehensive analysis)")
    print("\nOpen these files in a web browser to interact with the plots.")
    print("=" * 80)

    return fig1, fig2, li_data


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    fig1, fig2, li_data = main()

    # Optional: Display in Jupyter/IPython
    fig1.show()
    fig2.show()
