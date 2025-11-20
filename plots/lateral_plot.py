"""
Laterality Index (LI_MI) Distribution by Gait Phase - Interactive Visualization
Author: Analysis Script
Date: 2024

This script creates interactive boxplots showing the distribution of
Laterality Index across different gait phases using Plotly.

The Laterality Index is defined as:
L_{class1 vs class2} = (I_{class1} - I_{class2}) / (I_{class1} + I_{class2}) × 100

Where I is the mutual information between toe and angle.

Two figures are generated:
1. R-R vs L-L: Ipsilateral comparison (same side)
2. R-L vs L-R: Contralateral comparison (opposite sides)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
CONFIG_FILE = Path("../config.yml")
DATA_FILE = Path("../data/csv/sim02_complete_data.csv")

# Phase order for consistent display
PHASE_ORDER = ["stance", "swing", "cycle"]
PHASE_DISPLAY = {"stance": "Stance", "swing": "Swing", "cycle": "Complete Cycle"}
METRIC = "ilr1"

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_config():
    """Load configuration from YAML file"""
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config


def calculate_laterality_index(data, class1, class2, metric):
    """
    Calculate Laterality Index between two classes

    L = (I_class1 - I_class2) / (I_class1 + I_class2) × 100

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the gait data (healthy subjects only)
    class1 : str
        First class identifier (e.g., 'R-R')
    class2 : str
        Second class identifier (e.g., 'L-L')
    metric : str
        Metric to use for LI calculation (default: 'mi' for mutual information)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: subject, phase, LI, class1_value, class2_value
    """
    results = []

    # Get unique phases
    phases = data["cycle"].unique()

    for phase in phases:
        # Filter data for this phase
        phase_data = data[data["cycle"] == phase].copy()

        # Get class1 and class2 data
        class1_data = phase_data[phase_data["toe_angle"] == class1][
            ["subject", metric]
        ].copy()
        class2_data = phase_data[phase_data["toe_angle"] == class2][
            ["subject", metric]
        ].copy()

        class1_data.columns = ["subject", "class1_value"]
        class2_data.columns = ["subject", "class2_value"]

        # Merge
        comparison = pd.merge(class1_data, class2_data, on="subject", how="inner")

        # Calculate Laterality Index
        # LI = (I_class1 - I_class2) / (I_class1 + I_class2) × 100
        comparison["LI"] = -(
            (comparison["class1_value"] - comparison["class2_value"])
            / (comparison["class1_value"] + comparison["class2_value"])
            * 100
        )
        comparison["phase"] = phase

        results.append(
            comparison[["subject", "phase", "LI", "class1_value", "class2_value"]]
        )

    return pd.concat(results, ignore_index=True)


def get_statistics(data_series):
    """Calculate statistical measures for a data series"""
    stats_dict = {
        "mean": data_series.mean(),
        "median": data_series.median(),
        "std": data_series.std(),
        "q1": data_series.quantile(0.25),
        "q3": data_series.quantile(0.75),
        "n": len(data_series),
        "n_positive": (data_series > 0).sum(),
        "pct_positive": (data_series > 0).sum() / len(data_series) * 100,
    }

    return stats_dict


def create_laterality_figure(
    li_data, class1, class2, config, comparison_name="Ipsilateral"
):
    """
    Create figure with boxplot and statistics table

    Parameters:
    -----------
    li_data : pd.DataFrame
        DataFrame with laterality index data
    class1 : str
        First class identifier
    class2 : str
        Second class identifier
    config : dict
        Configuration dictionary from YAML
    comparison_name : str
        Name of comparison for title (e.g., 'Ipsilateral', 'Contralateral')

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    lat_config = config["laterality-plot"]
    box_config = lat_config["boxplot"]
    table_config = lat_config["table"]
    interp_config = lat_config["interpretation-box"]

    # Create subplots: boxplot in row1-col1, table in row2-col2
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            lat_config["subtitles"]["boxplot"],
            lat_config["subtitles"]["table"],
        ),
        specs=[[{"type": "box", "colspan": 1}, {"type": "table"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        column_widths=[0.5, 0.5],
    )

    # Prepare data for table
    table_data = []

    # Add boxplots for each phase
    for phase in PHASE_ORDER:
        phase_data = li_data[li_data["phase"] == phase]["LI"]
        subjects = li_data[li_data["phase"] == phase]["subject"]

        if len(phase_data) == 0:
            continue

        # Get statistics for this phase
        stats = get_statistics(phase_data)

        # Add to table data
        table_data.append(
            [
                PHASE_DISPLAY[phase],
                f"{stats['mean']:.2f}",
                f"{stats['median']:.2f}",
                f"{stats['std']:.2f}",
                f"[{stats['q1']:.2f}, {stats['q3']:.2f}]",
                f"{stats['pct_positive']:.1f}%",
            ]
        )

        # Create hover text for individual points
        hover_text = [
            f"Subject: {subj}<br>LI: {val:.2f}%<br>Phase: {PHASE_DISPLAY[phase]}"
            for subj, val in zip(subjects, phase_data)
        ]

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
                # boxmean=True,  # Show mean as a dashed line
                boxmean="sd",  # Show mean and std dev
                line=dict(
                    # color=box_config["colors"][phase],
                    color=box_config["colors"]["lines"],
                    width=box_config["line-width"],
                ),
                text=hover_text,
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
                        "<b>Mean (%)</b>",
                        "<b>Median (%)</b>",
                        "<b>SD (%)</b>",
                        "<b>IQR (%)</b>",
                        "<b>% Positive</b>",
                    ],
                    fill_color=table_config["header-bgcolor"],
                    font=dict(
                        color=table_config["header-fontcolor"],
                        size=table_config["header-fontsize"],
                        family=lat_config["subtitle-font-family"],
                    ),
                    align=table_config["header-align"],
                    height=table_config["cells-height"],
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[
                        table_config["cells-bgcolor"] * len(table_data)
                    ],  # Alternating colors
                    align=table_config["cells-align"],
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
            "text": f"{lat_config['title']} - <b>{comparison_name} ({class1} vs {class2})</b>",
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
        title_text=box_config["xaxis-title"],
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
        title_text=box_config["yaxis-title"],
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
        if annotation["text"] in [
            lat_config["subtitles"]["boxplot"],
            lat_config["subtitles"]["table"],
        ]:
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
    print("LATERALITY INDEX VISUALIZATION")
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

    # Define comparisons
    comparisons = [
        {
            "name": "Ipsilateral",
            "class1": "R-R",
            "class2": "L-L",
            "description": "Same side comparison",
        },
        {
            "name": "Contralateral",
            "class1": "R-L",
            "class2": "L-R",
            "description": "Opposite sides comparison",
        },
    ]

    # Create output directory
    output_dir = Path("../figs/laterality_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each comparison
    for comp in comparisons:
        print(
            f"\n4. Processing {comp['name']} comparison ({comp['class1']} vs {comp['class2']})..."
        )

        # Calculate Laterality Index
        li_data = calculate_laterality_index(
            healthy, comp["class1"], comp["class2"], metric=METRIC
        )
        print(f"   Calculated LI for {len(li_data)} subject-phase combinations")

        # Display summary statistics
        print(f"\n   Summary Statistics for {comp['name']}:")
        print("   " + "-" * 76)

        for phase in PHASE_ORDER:
            phase_data = li_data[li_data["phase"] == phase]["LI"]
            if len(phase_data) > 0:
                stats = get_statistics(phase_data)
                print(f"\n   {PHASE_DISPLAY[phase].upper()}:")
                print(f"      Mean ± SD: {stats['mean']:.2f} ± {stats['std']:.2f}%")
                print(
                    f"      Median [IQR]: {stats['median']:.2f} "
                    f"[{stats['q1']:.2f}, {stats['q3']:.2f}]%"
                )
                print(f"      % Positive: {stats['pct_positive']:.1f}%")

        # Create figure
        print(f"\n   Creating visualization...")
        fig = create_laterality_figure(
            li_data, comp["class1"], comp["class2"], config, comp["name"]
        )

        # Save files
        file_prefix = f"laterality_index_{comp['name'].lower()}"

        # Save HTML
        html_file = output_dir / f"{file_prefix}.html"
        fig.write_html(str(html_file))
        print(f"   ✓ Saved: {html_file}")

        # Save PDF
        try:
            pdf_file = output_dir / f"{file_prefix}.pdf"
            fig.write_image(
                str(pdf_file),
                width=config["laterality-plot"]["export-pdf"]["width"],
                height=config["laterality-plot"]["export-pdf"]["height"],
                scale=config["laterality-plot"]["export-pdf"]["scale"],
            )
            print(f"   ✓ Saved: {pdf_file}")
        except Exception as e:
            print(f"   ⚠ Could not save PDF: {e}")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nFiles saved to: {output_dir}")
    print("=" * 80)


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()
