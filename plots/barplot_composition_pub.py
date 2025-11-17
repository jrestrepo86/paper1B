from pathlib import Path

import kaleido as k
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
METRICS_FILE = Path("../data/csv/sim02_metrics.csv")
PVALS_ILR_FILE = Path("../data/csv/sim02_composition_pvals_nonparam.csv")
# PVALS_ILR_FILE = Path("../data/csv/sim02_composition_pvals_param.csv")
SUBJECT_DATA_FILE = Path("../data/csv/sim02_complete_data.csv")

ILR_COLUMNS = ["ilr2", "ilr1"]
PROPORTIONS = ["toe_var_prop", "angle_var_prop", "inter_dependence_prop"]
CLR_COLUMNS = ["toe_clr", "angle_clr", "inter_dep_clr"]

FIGS_OUTPUT_PATH = Path("../figs/composition_publication")

CYCLES = ["swing", "stance", "cycle"]

# Define colors - publication friendly (colorblind safe)
COLORS = {
    "toe_var_prop": "#0072B2",  # Blue for H(T|A)
    "angle_var_prop": "#009E73",  # Green for H(A|T)
    "inter_dependence_prop": "#D55E00",  # Orange for I(T;A)
}

VARIABLE_LABELS = {
    "toe_var_prop": "Toe Var.",
    "angle_var_prop": "Angle Var.",
    "inter_dependence_prop": "Inter-dep.",
    "total_var": "Total Var",
    "toe_clr": "Toe Var.",
    "angle_clr": "Angle Var.",
    "inter_dep_clr": "Inter-dep.",
}


def get_pvalue_color(log_pval):
    """Get color based on -log10(p-value)"""
    if log_pval >= 3:  # p < 0.001
        return "#440154"  # Purple
    elif log_pval >= 2:  # p < 0.01
        return "#31688e"  # Blue
    elif log_pval >= 1.301:  # p < 0.05
        return "#35b779"  # Green
    else:
        return "#888888"  # Gray


def get_significance_symbol(pvalue):
    """Convert p-value to significance symbol"""
    if pd.isna(pvalue):
        return "ns"
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.05:
        return "*"
    else:
        return "ns"


def get_effect_size_category(d):
    """Categorize effect size"""
    if pd.isna(d):
        return "none"
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def make_proportion_barplot(
    df: pd.DataFrame,
    condition: str | list,
    prosthesis: str | list,
    cycle: str,
    classes: list,
    fig: go.Figure,
    row: int,
    col: int,
    text_size: int = 16,  # Increased default
    bar_width: float = 0.7,  # Adjustable bar width
):
    """
    Create stacked bar plot showing proportions of H(T|A), H(A|T), and I(T;A)

    IMPROVEMENTS FOR PUBLICATION:
    - Larger text size (adjustable)
    - Better text positioning logic for small proportions
    - Cleaner labels
    """

    if isinstance(condition, str):
        condition = [condition]
    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]

    # Filter data
    dfc = df[
        (df["condition"].isin(condition))
        & (df["prosthesis"].isin(prosthesis))
        & (df["cycle"] == cycle)
    ]

    # Extract data for each class
    data_by_class = {}
    total_var = []

    for class_name in classes:
        # For amputee subjects, use amp_sound column
        if class_name in ["A-A", "S-S", "A-S", "S-A"]:
            class_data = dfc[dfc["amp_sound"] == class_name]
        # For healthy subjects, use toe_angle column
        elif class_name in ["L-L", "R-R", "L-R", "R-L"]:
            class_data = dfc[dfc["toe_angle"] == class_name]
        # For aggregate healthy classes (Same, Contra)
        else:
            class_data = dfc[dfc["toe_angle"] == class_name]

        if len(class_data) == 0:
            continue

        data_by_class[class_name] = {
            "toe_var_prop": class_data["toe_var_prop"].values[0] * 100,
            "angle_var_prop": class_data["angle_var_prop"].values[0] * 100,
            "inter_dependence_prop": class_data["inter_dependence_prop"].values[0]
            * 100,
            "toe_var": class_data["toe_var"].values[0],
            "angle_var": class_data["angle_var"].values[0],
            "inter_dependence": class_data["inter_dependence"].values[0],
        }
        total_var.append(class_data["total_var"].values[0])

    # Extract values
    toe_var_prop = [
        data_by_class[c]["toe_var_prop"] for c in classes if c in data_by_class
    ]
    angle_var_prop = [
        data_by_class[c]["angle_var_prop"] for c in classes if c in data_by_class
    ]
    inter_dependence_prop = [
        data_by_class[c]["inter_dependence_prop"] for c in classes if c in data_by_class
    ]

    toe_var = [data_by_class[c]["toe_var"] for c in classes if c in data_by_class]
    angle_var = [data_by_class[c]["angle_var"] for c in classes if c in data_by_class]
    inter_dependence = [
        data_by_class[c]["inter_dependence"] for c in classes if c in data_by_class
    ]

    valid_classes = [c for c in classes if c in data_by_class]

    # IMPROVED TEXT DISPLAY LOGIC
    # For small proportions, show only percentage or use outside positioning
    def format_bar_text(proportion, value, threshold=8):
        return f"{proportion:.2f}%<br>({value:.2f})"

    # Add stacked bars with improved text
    fig.add_trace(
        go.Bar(
            name="Toe Var",
            x=valid_classes,
            y=toe_var_prop,
            marker_color=COLORS["toe_var_prop"],
            text=[format_bar_text(p, v) for p, v in zip(toe_var_prop, toe_var)],
            textposition="inside",
            textfont=dict(color="white", size=text_size, family="Arial"),
            insidetextanchor="middle",
            showlegend=True,
            legendgroup="htoe",
            hovertemplate="<b>H(T|A)</b><br>%{y:.2f}%<br>%{customdata:.3f} nats<extra></extra>",
            customdata=toe_var,
            width=bar_width,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Bar(
            name="Angle Var",
            x=valid_classes,
            y=angle_var_prop,
            marker_color=COLORS["angle_var_prop"],
            text=[format_bar_text(p, v) for p, v in zip(angle_var_prop, angle_var)],
            textposition="inside",
            textfont=dict(color="white", size=text_size, family="Arial"),
            insidetextanchor="middle",
            showlegend=True,
            legendgroup="hangle",
            hovertemplate="<b>H(A|T)</b><br>%{y:.2f}%<br>%{customdata:.3f} nats<extra></extra>",
            customdata=angle_var,
            width=bar_width,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Bar(
            name="Inter-dep.",
            x=valid_classes,
            y=inter_dependence_prop,
            marker_color=COLORS["inter_dependence_prop"],
            text=[
                format_bar_text(p, v)
                for p, v in zip(inter_dependence_prop, inter_dependence)
            ],
            textposition="inside",
            textfont=dict(color="white", size=text_size, family="Arial"),
            insidetextanchor="middle",
            showlegend=True,
            legendgroup="coord",
            hovertemplate="<b>I(T;A)</b><br>%{y:.2f}%<br>%{customdata:.3f} nats<extra></extra>",
            customdata=inter_dependence,
            width=bar_width,
        ),
        row=row,
        col=col,
    )

    # Add H(T,A) annotations above bars - larger font
    for class_name, joint_ent in zip(valid_classes, total_var):
        fig.add_annotation(
            x=class_name,
            y=105,
            text=f"<b>{joint_ent:.2f}</b>",
            showarrow=False,
            font=dict(size=text_size, color="black", family="Arial"),
            xanchor="center",
            yanchor="bottom",
            row=row,
            col=col,
        )

    # Larger axis labels
    fig.update_yaxes(
        title_text="Composition (%)",
        title_font=dict(size=text_size + 2, family="Arial"),
        tickfont=dict(size=text_size - 2, family="Arial"),
        range=[0, 115],
        row=row,
        col=col,
    )

    fig.update_xaxes(
        tickfont=dict(size=text_size - 2, family="Arial"),
        row=row,
        col=col,
    )

    return fig


def make_omnibus_plot(
    df: pd.DataFrame,
    prosthesis: str | list,
    cycle: str,
    pairs: list,
    fig: go.Figure,
    row: int,
    col: int,
    text_size: int = 14,
):
    """Create plot showing omnibus test results"""

    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]
    else:
        prosthesis = ["-".join(prosthesis)]

    dfc = df[(df["prosthesis"].isin(prosthesis)) & (df["cycle"] == cycle)]
    pairs_reversed = list(reversed(pairs))

    x_vals = []
    colors = []
    hover_texts = []
    text_labels = []
    y_labels = []

    for class1, class2 in pairs_reversed:
        pair_str = f"({class1})-({class2})"
        comp_data = dfc[dfc["pair"] == pair_str]

        if len(comp_data) > 0:
            pval = comp_data["omnibus_pval"].values[0]
            mahalanobis_d = comp_data["mahalanobis_D"].values[0]
            cat = get_effect_size_category(mahalanobis_d)

            # Fix p-value at 1e-4 if smaller
            pval = max(pval, 1e-4)
            log_pval = -np.log10(pval)

            sig = get_significance_symbol(pval)
            x_vals.append(log_pval)
            colors.append(get_pvalue_color(log_pval))

            hover_texts.append(f"p = {pval:.4f} {sig}")
            text_labels.append(f"D={mahalanobis_d:.2f} ({cat})")
            y_labels.append(f"{class1} vs {class2}")

    y_positions = list(range(len(y_labels)))

    # Add bars
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_positions,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="inside",
            textfont=dict(color="white", size=text_size, family="Arial"),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
            width=0.7,
        ),
        row=row,
        col=col,
    )

    # Add significance lines
    for x_val, label in [(1.301, "0.05"), (2.0, "0.01"), (3.0, "0.001")]:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x_val,
            x1=x_val,
            y0=-1,
            y1=max(y_positions) + 0.46 if y_positions else 1,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=x_val,
            y=max(y_positions) + 0.5 if y_positions else 1,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=text_size - 2, color="gray", family="Arial"),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log<sub>10</sub>(p-value)",
        title_font=dict(size=text_size, family="Arial"),
        tickfont=dict(size=text_size - 2, family="Arial"),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        tickfont=dict(size=text_size - 2, family="Arial"),
        row=row,
        col=col,
    )

    return fig


def make_ilr_coordinates_plot(
    df: pd.DataFrame,
    prosthesis: str | list,
    cycle: str,
    pairs: list,
    fig: go.Figure,
    row: int,
    col: int,
    text_size: int = 14,
):
    """Create plot showing ILR coordinate post-hoc tests"""

    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]
    else:
        prosthesis = ["-".join(prosthesis)]

    dfc = df[(df["prosthesis"].isin(prosthesis)) & (df["cycle"] == cycle)]
    pairs_reversed = list(reversed(pairs))

    x_vals = []
    colors = []
    hover_texts = []
    text_labels = []
    y_labels = []
    y_positions = []

    y_pos = 0
    for class1, class2 in pairs_reversed:
        pair_str = f"({class1})-({class2})"
        comp_data = dfc[dfc["pair"] == pair_str]

        if len(comp_data) > 0:
            row_data = comp_data.iloc[0]

            if row_data["omnibus_significant"]:
                for ilr in ILR_COLUMNS:
                    pval = row_data[f"{ilr}_pval_bonf"]
                    d = row_data[f"{ilr}_cohens_d"]
                    pval = max(pval, 1e-4)
                    log_pval = -np.log10(pval)
                    sig = get_significance_symbol(pval)
                    cat = get_effect_size_category(d)

                    x_vals.append(log_pval)
                    colors.append(get_pvalue_color(log_pval))
                    hover_texts.append(
                        f"{ilr}: p={pval:.4f} {sig}<br>d={d:.2f} ({cat})"
                    )
                    text_labels.append(f"d={d:.2f} ({cat})")
                    y_labels.append(f"{class1} vs {class2}<br><b>{ilr}</b>")
                    y_positions.append(y_pos)
                    y_pos += 1

            else:
                # Omnibus not significant
                for coord in ILR_COLUMNS:
                    x_vals.append(0)
                    colors.append("#d3d3d3")
                    hover_texts.append("Omnibus NS")
                    text_labels.append("")
                    y_labels.append(f"{class1} vs {class2}<br><b>{coord}</b>")
                    y_positions.append(y_pos)
                    y_pos += 1

        y_pos += 0.5

    # Add bars
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_positions,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="inside",
            textfont=dict(color="white", size=text_size, family="Arial"),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
            width=0.7,
        ),
        row=row,
        col=col,
    )

    # Add significance lines
    for x_val, label in [(1.301, "0.05"), (2.0, "0.01"), (3.0, "0.001")]:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x_val,
            x1=x_val,
            y0=-1,
            y1=max(y_positions) + 0.46 if y_positions else 1,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=x_val,
            y=max(y_positions) + 0.5 if y_positions else 1,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=text_size - 2, color="gray", family="Arial"),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log<sub>10</sub>(p-value)",
        title_font=dict(size=text_size, family="Arial"),
        tickfont=dict(size=text_size - 2, family="Arial"),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        tickfont=dict(size=text_size - 2, family="Arial"),
        row=row,
        col=col,
    )

    return fig


def make_proportions_posthoc_plot(
    df: pd.DataFrame,
    prosthesis: str | list,
    cycle: str,
    pairs: list,
    fig: go.Figure,
    row: int,
    col: int,
    text_size: int = 14,
):
    """Create plot showing individual proportion tests"""

    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]
    else:
        prosthesis = ["-".join(prosthesis)]

    dfc = df[(df["prosthesis"].isin(prosthesis)) & (df["cycle"] == cycle)]
    pairs_reversed = list(reversed(pairs))

    x_vals = []
    colors = []
    hover_texts = []
    text_labels = []
    y_labels = []
    y_positions = []

    y_pos = 0
    for class1, class2 in pairs_reversed:
        pair_str = f"({class1})-({class2})"
        comp_data = dfc[dfc["pair"] == pair_str]

        if len(comp_data) > 0:
            row_data = comp_data.iloc[0]

            if row_data["omnibus_significant"]:
                for prop in CLR_COLUMNS:
                    pval = row_data[f"{prop}_pval_bonf"]
                    d_clr = row_data[f"{prop}_cohens_d"]

                    pval = max(pval, 1e-4)
                    log_pval = -np.log10(pval)

                    sig = get_significance_symbol(pval)
                    cat = get_effect_size_category(d_clr)

                    x_vals.append(log_pval)
                    colors.append(get_pvalue_color(log_pval))

                    hover_texts.append(
                        f"{VARIABLE_LABELS[prop]}: p={pval:.4f} {sig}<br>d={d_clr:.2f} ({cat})"
                    )
                    text_labels.append(f"d={d_clr:.2f} ({cat})")
                    y_labels.append(
                        f"{class1} vs {class2} - <b>{VARIABLE_LABELS[prop]}</b>"
                    )
                    y_positions.append(y_pos)
                    y_pos += 1.2
            else:
                # Omnibus not significant
                for prop in CLR_COLUMNS:
                    x_vals.append(0)
                    colors.append("#d3d3d3")
                    hover_texts.append("Omnibus NS")
                    text_labels.append("")
                    y_labels.append(
                        f"{class1} vs {class2} - <b>{VARIABLE_LABELS[prop]}</b>"
                    )
                    y_positions.append(y_pos)
                    y_pos += 1.2

        y_pos += 0.5

    # Add bars
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_positions,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="inside",
            textfont=dict(color="white", size=text_size, family="Arial"),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
            width=0.7,
        ),
        row=row,
        col=col,
    )

    # Add significance lines
    for x_val, label in [(1.301, "0.05"), (2.0, "0.01"), (3.0, "0.001")]:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x_val,
            x1=x_val,
            y0=-1,
            y1=max(y_positions) + 0.46 if y_positions else 1,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=x_val,
            y=max(y_positions) + 0.5 if y_positions else 1,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=text_size - 2, color="gray", family="Arial"),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log<sub>10</sub>(p-value)",
        title_font=dict(size=text_size, family="Arial"),
        tickfont=dict(size=text_size - 2, family="Arial"),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        tickfont=dict(size=text_size - 2, family="Arial"),
        row=row,
        col=col,
    )

    return fig


def make_ilr_scatter_plot(
    subject_df: pd.DataFrame,
    prosthesis: str | list,
    cycle: str,
    classes: list,
    class_colors: dict,
    fig: go.Figure,
    row: int,
    col: int,
    text_size: int = 14,
):
    """
    Create scatter plot of ILR1 vs ILR2 coordinates for specified classes.

    Args:
        subject_df: DataFrame with subject-level data (sim02_complete_data.csv)
        prosthesis: Prosthesis type(s) to filter
        cycle: Gait cycle phase
        classes: List of class names to plot
        class_colors: Dictionary mapping class names to colors
        fig: Plotly figure object
        row, col: Subplot position
        text_size: Font size for labels
    """
    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]

    # Filter data
    dfc = subject_df[
        (subject_df["prosthesis"].isin(prosthesis)) & (subject_df["cycle"] == cycle)
    ]

    # Plot each class
    for class_name in classes:
        # Determine which column to use for filtering
        if class_name in ["L-L", "R-R", "L-R", "R-L"]:
            class_data = dfc[dfc["toe_angle"] == class_name]
        else:
            class_data = dfc[dfc["amp_sound"] == class_name]

        if len(class_data) == 0:
            continue

        # Extract ILR coordinates
        ilr1 = class_data["ilr1"].values
        ilr2 = class_data["ilr2"].values

        # Add scatter trace
        fig.add_trace(
            go.Scatter(
                x=ilr1,
                y=ilr2,
                mode="markers",
                name=class_name,
                marker=dict(
                    size=10,
                    color=class_colors.get(class_name, "#333333"),
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=dfc["subject"],
                hovertemplate=(
                    f"<b>{class_name}</b><br>"
                    "Subject: %{text}<br>"
                    "ILR1: %{x:.3f}<br>"
                    "ILR2: %{y:.3f}<br>"
                    "<extra></extra>"
                ),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    # Add reference lines at origin
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", line_width=1, row=row, col=col
    )
    fig.add_vline(
        x=0, line_dash="dash", line_color="gray", line_width=1, row=row, col=col
    )

    # Update axes
    fig.update_xaxes(
        title_text="ILR1",
        title_font=dict(size=text_size, family="Arial"),
        tickfont=dict(size=text_size - 2, family="Arial"),
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="lightgray",
        row=row,
        col=col,
    )

    fig.update_yaxes(
        title_text="ILR2",
        title_font=dict(size=text_size, family="Arial"),
        tickfont=dict(size=text_size - 2, family="Arial"),
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="lightgray",
        row=row,
        col=col,
    )

    return fig


if __name__ == "__main__":
    # Load data
    metrics_df = pd.read_csv(METRICS_FILE)
    pvals_df = pd.read_csv(PVALS_ILR_FILE)

    # Load subject-level data for ILR scatter plots
    subject_df = pd.read_csv(SUBJECT_DATA_FILE)

    # Define colors for ILR scatter plots
    ILR_COLORS = {
        "L-L": "#0072B2",  # Blue
        "R-R": "#D55E00",  # Orange
        "L-R": "#009E73",  # Green
        "R-L": "#CC79A7",  # Pink
        "A-A": "#0072B2",  # Blue
        "S-S": "#D55E00",  # Orange
        "A-S": "#009E73",  # Green
        "S-A": "#CC79A7",  # Pink
    }

    # Load complete subject-level data for scatter plots
    complete_data_df = subject_df.copy()

    # Define plots
    PLOTS = [
        # Healthy subjects
        {
            "name": "Healthy",
            "condition": "healthy",
            "prosthesis": "none",
            "classes": ["L-L", "R-R", "L-R", "R-L"],
            "pairs": [
                ("L-L", "R-R"),
                ("L-R", "R-L"),
            ],
        },
        # Amputee subjects with Mechanical prosthesis
        {
            "name": "Amputee_Mech",
            "condition": "amputee",
            "prosthesis": "Mech",
            "classes": ["A-A", "S-S", "A-S", "S-A"],
            "pairs": [
                ("A-A", "S-S"),
                ("A-S", "S-A"),
            ],
        },
        # Amputee subjects with Echo prosthesis
        {
            "name": "Amputee_Ech",
            "condition": "amputee",
            "prosthesis": "Ech",
            "classes": ["A-A", "S-S", "A-S", "S-A"],
            "pairs": [
                ("A-A", "S-S"),
                ("A-S", "S-A"),
            ],
        },
    ]

    for plot_config in PLOTS:
        name = plot_config["name"]
        condition = plot_config["condition"]
        prosthesis = plot_config["prosthesis"]
        classes = plot_config["classes"]
        pairs = plot_config["pairs"]

        for cycle in CYCLES:
            # Determine layout based on condition
            is_healthy = condition == "healthy"

            if is_healthy:
                # Extended layout for Healthy subjects (3×3 with ILR scatter plots)
                subplot_titles = (
                    "<b>A.</b> Compositional Analysis by Class",
                    "<b>B.</b> Omnibus Test (Hotelling's T²)",
                    "<b>E.</b> ILR Space: Ipsilateral (L-L vs R-R)",
                    "<b>C.</b> ILR Coordinates Post-hoc",
                    "<b>F.</b> ILR Space: Contralateral (L-R vs R-L)",
                    "<b>D.</b> Individual Proportions Post-hoc (CLR)",
                )
                ips_clases = ["L-L", "R-R"]
                contra_clases = ["L-R", "R-L"]
            else:
                subplot_titles = (
                    "<b>A.</b> Compositional Analysis by Class",
                    "<b>B.</b> Omnibus Test (Hotelling's T²)",
                    "<b>E.</b> ILR Space: Same Side (A-A vs S-S)",
                    "<b>C.</b> ILR Coordinates Post-hoc",
                    "<b>F.</b> ILR Space: Contra Side (A-S vs S-A)",
                    "<b>D.</b> Individual Proportions Post-hoc (CLR)",
                )
                ips_clases = ["A-A", "S-S"]
                contra_clases = ["A-S", "S-A"]

            fig = make_subplots(
                rows=3,
                cols=3,
                subplot_titles=subplot_titles,
                specs=[
                    [{"rowspan": 3}, {"type": "bar"}, {"type": "scatter"}],
                    [None, {"type": "bar"}, {"type": "scatter"}],
                    [None, {"type": "bar"}, None],
                ],
                horizontal_spacing=0.1,
                vertical_spacing=0.12,
                row_heights=[0.33, 0.33, 0.34],
                column_widths=[0.35, 0.35, 0.30],
            )

            # Add subplots with larger text
            fig = make_proportion_barplot(
                metrics_df,
                condition,
                prosthesis,
                cycle,
                classes,
                fig,
                row=1,
                col=1,
                text_size=18,  # Larger text for composition
                bar_width=0.9,
            )

            fig = make_omnibus_plot(
                pvals_df, prosthesis, cycle, pairs, fig, row=1, col=2, text_size=18
            )

            fig = make_ilr_coordinates_plot(
                pvals_df, prosthesis, cycle, pairs, fig, row=2, col=2, text_size=18
            )

            fig = make_proportions_posthoc_plot(
                pvals_df, prosthesis, cycle, pairs, fig, row=3, col=2, text_size=18
            )

            # Add ILR scatter plots for Healthy subjects only (column 3)
            # Plot 1: Ipsi
            fig = make_ilr_scatter_plot(
                subject_df,
                prosthesis,
                cycle,
                classes=ips_clases,
                class_colors=ILR_COLORS,
                fig=fig,
                row=1,
                col=3,
                text_size=18,
            )

            # Plot 2: contra side
            fig = make_ilr_scatter_plot(
                subject_df,
                prosthesis,
                cycle,
                classes=contra_clases,
                class_colors=ILR_COLORS,
                fig=fig,
                row=2,
                col=3,
                text_size=18,
            )

            # Update layout for publication
            subject_type = "Amputee" if condition == "amputee" else "Healthy"
            prosthesis_label = f" - {prosthesis}" if condition == "amputee" else ""
            if cycle == "cycle":
                cycle_text = "Complete Cycle"
            elif cycle == "swing":
                cycle_text = "Swing Phase"
            else:
                cycle_text = "Stance Phase"

            title = f"<b>Compositional Analysis - {cycle_text}</b>"

            figure_width = 2100  # A4 landscape width
            figure_height = 1200

            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 24, "family": "Arial"},
                },
                barmode="stack",
                height=figure_height,
                width=figure_width,
                font=dict(size=18, family="Arial"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=16, family="Arial"),
                ),
            )
            for i in fig["layout"]["annotations"]:
                for t in subplot_titles:
                    if t in i["text"]:
                        i["font"] = dict(size=20)

            # Save figure
            FIGS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            output_file = FIGS_OUTPUT_PATH / f"pub_ilr_analysis_{name}_{cycle}.html"
            fig.write_html(str(output_file))
            print(f"Saved: {output_file}")

            # Also save as high-res static image for paper
            output_pdf = FIGS_OUTPUT_PATH / f"pub_ilr_analysis_{name}_{cycle}.pdf"
            fig.write_image(
                str(output_pdf), width=figure_width, height=figure_height, scale=1
            )
            print(f"Saved: {output_pdf}")
