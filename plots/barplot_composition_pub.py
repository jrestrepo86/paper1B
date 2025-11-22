from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

# Configuration files
CONFIG_FILE = Path("../config.yml")
METRICS_FILE = Path("../data/csv/sim02_metrics.csv")
PVALS_ILR_FILE = Path("../data/csv/sim02_composition_pvals_nonparam.csv")
SUBJECT_DATA_FILE = Path("../data/csv/sim02_complete_data.csv")

ILR_COLUMNS = ["ilr2", "ilr1"]
PROPORTIONS = ["toe_var_prop", "angle_var_prop", "inter_dependence_prop"]
CLR_COLUMNS = ["toe_clr", "angle_clr", "inter_dep_clr"]


def load_config():
    """Load configuration from YAML file"""
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config


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


def format_bar_text(proportion, value):
    """Format text for stacked bars with better logic for small values"""
    if proportion < 5:
        return f"<b>{proportion:.2f}%</b> ({value:.2f})"
    else:
        return f"<b>{proportion:.2f}%</b><br>({value:.2f})"


def make_proportion_barplot(
    df: pd.DataFrame,
    condition: str | list,
    prosthesis: str | list,
    cycle: str,
    classes: list,
    fig: go.Figure,
    row: int,
    col: int,
    config: dict,
):
    """Create stacked bar plot showing proportions of H(T|A), H(A|T), and I(T;A)"""

    plot_config = config["coposition"]["proportions-bar-plot"]
    colors = plot_config["colors"]
    text_size = plot_config["text-size"]
    bar_width = plot_config["bar-width"]
    annotation_text_size = plot_config["annotation-text-size"]
    annotation_font_family = plot_config["annotation-font-family"]
    annotation_font_color = plot_config["annotation-font-color"]
    xaxis_label_text_size = plot_config["xaxis-label-text-size"]
    yaxis_label_text_size = plot_config["yaxis-label-text-size"]

    variable_labels = config["coposition"]["variable-labels"]

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
            class_data = dfc[dfc["amp_sound"] == class_name]

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
    valid_classes = [c for c in classes if c in data_by_class]
    toe_var_prop = [data_by_class[c]["toe_var_prop"] for c in valid_classes]
    angle_var_prop = [data_by_class[c]["angle_var_prop"] for c in valid_classes]
    inter_dependence_prop = [
        data_by_class[c]["inter_dependence_prop"] for c in valid_classes
    ]
    toe_var = [data_by_class[c]["toe_var"] for c in valid_classes]
    angle_var = [data_by_class[c]["angle_var"] for c in valid_classes]
    inter_dependence = [data_by_class[c]["inter_dependence"] for c in valid_classes]

    # Add stacked bars with config colors
    fig.add_trace(
        go.Bar(
            name=variable_labels["toe_var_prop"],
            x=valid_classes,
            y=toe_var_prop,
            marker_color=colors["toe_var_prop"],
            text=[format_bar_text(p, v) for p, v in zip(toe_var_prop, toe_var)],
            textposition="inside",
            textfont=dict(
                color=annotation_font_color,
                size=annotation_text_size,
                family=annotation_font_family,
            ),
            insidetextanchor="middle",
            showlegend=True,
            legendgroup="legend",
            legend="legend",
            legendgrouptitle_text="Composition" if row == 1 else None,
            hovertemplate="<b>H(T|A)</b><br>%{y:.2f}%<br>%{customdata:.3f} nats<extra></extra>",
            customdata=toe_var,
            width=bar_width,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Bar(
            name=variable_labels["angle_var_prop"],
            x=valid_classes,
            y=angle_var_prop,
            marker_color=colors["angle_var_prop"],
            text=[format_bar_text(p, v) for p, v in zip(angle_var_prop, angle_var)],
            textposition="inside",
            textfont=dict(
                color=annotation_font_color,
                size=annotation_text_size,
                family=annotation_font_family,
            ),
            insidetextanchor="middle",
            showlegend=True,
            legend="legend",
            legendgroup="legend",
            legendgrouptitle_text="Composition" if row == 1 else None,
            hovertemplate="<b>H(A|T)</b><br>%{y:.2f}%<br>%{customdata:.3f} nats<extra></extra>",
            customdata=angle_var,
            width=bar_width,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Bar(
            name=variable_labels["inter_dependence_prop"],
            x=valid_classes,
            y=inter_dependence_prop,
            marker_color=colors["inter_dependence_prop"],
            text=[
                format_bar_text(p, v)
                for p, v in zip(inter_dependence_prop, inter_dependence)
            ],
            textposition="inside",
            textfont=dict(
                color=annotation_font_color,
                size=annotation_text_size,
                family=annotation_font_family,
            ),
            insidetextanchor="middle",
            showlegend=True,
            legend="legend",
            legendgroup="legend",
            legendgrouptitle_text="Composition" if row == 1 else None,
            hovertemplate="<b>I(T;A)</b><br>%{y:.2f}%<br>%{customdata:.3f} nats<extra></extra>",
            customdata=inter_dependence,
            width=bar_width,
        ),
        row=row,
        col=col,
    )

    # Add H(T,A) annotations above bars
    for class_name, joint_ent in zip(valid_classes, total_var):
        fig.add_annotation(
            x=class_name,
            y=105,
            text=f"<b>{joint_ent:.2f}</b>",
            showarrow=False,
            font=dict(
                size=annotation_text_size, color="black", family=annotation_font_family
            ),
            xanchor="center",
            yanchor="bottom",
            row=row,
            col=col,
        )

    # Update axis labels
    fig.update_yaxes(
        title_text="Composition (%)",
        title_font=dict(size=yaxis_label_text_size, family=annotation_font_family),
        tickfont=dict(size=yaxis_label_text_size - 2, family=annotation_font_family),
        range=[0, 115],
        row=row,
        col=col,
    )

    fig.update_xaxes(
        tickfont=dict(size=xaxis_label_text_size - 2, family=annotation_font_family),
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
    config: dict,
):
    """Create plot showing omnibus test results"""

    plot_config = config["coposition"]["omnibus-plot"]
    mahalanobis_text_size = plot_config["mahalanobis-text-size"]
    mahalanobis_font_family = plot_config["mahalanobis-font-family"]
    mahalanobis_font_color = plot_config["mahalanobis-font-color"]
    pval_annotation_text_size = plot_config["p-value-annotation-text-size"]
    pval_annotation_font_family = plot_config["p-value-annotation-font-family"]
    pval_annotation_font_color = plot_config["p-value-annotation-font-color"]
    xaxis_label_text_size = plot_config["xaxis-label-text-size"]
    yaxis_label_text_size = plot_config["yaxis-label-text-size"]

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
            row_data = comp_data.iloc[0]
            pval = row_data["omnibus_pval"]
            mahal_d = row_data["mahalanobis_D"]

            pval = max(pval, 1e-4)
            log_pval = -np.log10(pval)
            sig = get_significance_symbol(pval)

            x_vals.append(log_pval)
            colors.append(get_pvalue_color(log_pval))
            hover_texts.append(f"p = {pval:.4f} {sig}<br>Mahalanobis D = {mahal_d:.2f}")
            text_labels.append(f"<b>D = {mahal_d:.2f}</b>")
            y_labels.append(f"{class1} vs {class2}")

    # Create horizontal bar plot
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_labels,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="inside",
            textfont=dict(
                color=mahalanobis_font_color,
                size=mahalanobis_text_size,
                family=mahalanobis_font_family,
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
            width=0.5,
        ),
        row=row,
        col=col,
    )

    y_positions = list(range(len(y_labels)))

    # Add significance threshold lines
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
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(
                size=pval_annotation_text_size - 2,
                color=pval_annotation_font_color,
                family=pval_annotation_font_family,
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log<sub>10</sub>(p-value)",
        title_font=dict(size=xaxis_label_text_size, family=pval_annotation_font_family),
        tickfont=dict(
            size=xaxis_label_text_size - 2, family=pval_annotation_font_family
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        tickfont=dict(
            size=yaxis_label_text_size - 2, family=pval_annotation_font_family
        ),
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
    config: dict,
):
    """Create plot showing ILR coordinate post-hoc tests"""

    plot_config = config["coposition"]["ilr-coordinates-plot"]
    cohens_text_size = plot_config["cohens-text-size"]
    cohens_font_family = plot_config["cohens-font-family"]
    pval_annotation_text_size = plot_config["p-value-annotation-text-size"]
    pval_annotation_font_family = plot_config["p-value-annotation-font-family"]
    pval_annotation_font_color = plot_config["p-value-annotation-font-color"]
    xaxis_label_text_size = plot_config["xaxis-label-text-size"]
    yaxis_label_text_size = plot_config["yaxis-label-text-size"]

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
                        f"{ilr}: p={pval:.4f} {sig}<br>d = {d:.2f} ({cat})"
                    )
                    text_labels.append(f"<b>d = {d:.2f} ({cat})</b>")
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
            textfont=dict(
                color="white", size=cohens_text_size, family=cohens_font_family
            ),
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
            font=dict(
                size=pval_annotation_text_size - 2,
                color=pval_annotation_font_color,
                family=pval_annotation_font_family,
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log<sub>10</sub>(p-value)",
        title_font=dict(size=xaxis_label_text_size, family=pval_annotation_font_family),
        tickfont=dict(
            size=xaxis_label_text_size - 2, family=pval_annotation_font_family
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        tickfont=dict(
            size=yaxis_label_text_size - 2, family=pval_annotation_font_family
        ),
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
    config: dict,
):
    """Create plot showing individual proportion tests"""

    plot_config = config["coposition"]["clr-posthoc-plot"]
    variable_labels = config["coposition"]["variable-labels"]
    cohens_text_size = plot_config["cohens-text-size"]
    cohens_font_family = plot_config["cohens-font-family"]
    pval_annotation_text_size = plot_config["p-value-annotation-text-size"]
    pval_annotation_font_family = plot_config["p-value-annotation-font-family"]
    pval_annotation_font_color = plot_config["p-value-annotation-font-color"]
    xaxis_label_text_size = plot_config["xaxis-label-text-size"]
    yaxis_label_text_size = plot_config["yaxis-label-text-size"]

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
                        f"{variable_labels[prop]}: p={pval:.4f} {sig}<br>d={d_clr:.2f} ({cat})"
                    )
                    text_labels.append(f"<b>d = {d_clr:.2f} ({cat})</b>")
                    y_labels.append(
                        f"{class1} vs {class2} - <b>{variable_labels[prop]}</b>"
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
                        f"{class1} vs {class2} - <b>{variable_labels[prop]}</b>"
                    )
                    y_positions.append(y_pos)
                    y_pos += 1.2

        y_pos += 0.3

    # Create plot
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_positions,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="inside",
            textfont=dict(
                color="white", size=cohens_text_size, family=cohens_font_family
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
            width=1.0,
        ),
        row=row,
        col=col,
    )

    # Add significance threshold lines
    for x_val, label in [(1.301, "0.05"), (2.0, "0.01"), (3.0, "0.001")]:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x_val,
            x1=x_val,
            y0=-1,
            y1=max(y_positions) + 1.15 if y_positions else 1,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=x_val,
            y=max(y_positions) + 1.2 if y_positions else 1,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(
                size=pval_annotation_text_size - 2,
                color=pval_annotation_font_color,
                family=pval_annotation_font_family,
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log<sub>10</sub>(p-value)",
        title_font=dict(size=xaxis_label_text_size, family=pval_annotation_font_family),
        tickfont=dict(
            size=xaxis_label_text_size - 2, family=pval_annotation_font_family
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        tickfont=dict(
            size=yaxis_label_text_size - 2, family=pval_annotation_font_family
        ),
        row=row,
        col=col,
    )

    return fig


def make_ilr_scatter_plot(
    subject_df: pd.DataFrame,
    prosthesis: str | list,
    cycle: str,
    classes: list,
    fig: go.Figure,
    row: int,
    col: int,
    config: dict,
):
    """Create ILR scatter plot for specified classes"""

    plot_config = config["coposition"]["ilr-scatter-plot"]
    class_colors = plot_config["colors"]
    marker_size = plot_config["marker-size"]
    text_size = plot_config["text-size"]
    xaxis_label = plot_config["xaxis-label"]
    xaxis_label_text_size = plot_config["xaxis-label-text-size"]
    yaxis_label = plot_config["yaxis-label"]
    yaxis_label_text_size = plot_config["yaxis-label-text-size"]
    x_axis_zero = plot_config["xaxis-zeroline"]
    y_axis_zero = plot_config["yaxis-zeroline"]

    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]

    # Filter data
    dfc = subject_df[
        (subject_df["prosthesis"].isin(prosthesis)) & (subject_df["cycle"] == cycle)
    ]

    xmin = dfc["ilr1"].min()
    xmax = dfc["ilr1"].max()
    ymin = dfc["ilr2"].min()
    ymax = dfc["ilr2"].max()

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
                    size=marker_size,
                    color=class_colors.get(class_name, "#333333"),
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=class_data["subject"],
                hovertemplate=(
                    f"<b>{class_name}</b><br>"
                    "Subject: %{text}<br>"
                    "ILR1: %{x:.3f}<br>"
                    "ILR2: %{y:.3f}<br>"
                    "<extra></extra>"
                ),
                showlegend=True,
                legend="legend2",
                legendgroup="legend2",
                legendgrouptitle_text="ILR Classes" if row == 1 else None,
            ),
            row=row,
            col=col,
        )

    xmax = max(xmax, 0)
    ymax = max(ymax, 0)
    xmin = min(xmin, 0)
    ymin = min(ymin, 0)

    # Update axes
    fig.update_xaxes(
        range=[xmin - 0.1, xmax + 0.1],
        title_text=xaxis_label,
        title_font=dict(
            size=xaxis_label_text_size,
            family=config["coposition"]["subtitle-font-family"],
        ),
        tickfont=dict(
            size=xaxis_label_text_size - 2,
            family=config["coposition"]["subtitle-font-family"],
        ),
        zeroline=True,
        zerolinewidth=x_axis_zero["width"],
        zerolinecolor=x_axis_zero["color"],
        row=row,
        col=col,
    )

    fig.update_yaxes(
        range=[ymin - 0.1, ymax + 0.1],
        title_text=yaxis_label,
        title_font=dict(
            size=yaxis_label_text_size,
            family=config["coposition"]["subtitle-font-family"],
        ),
        tickfont=dict(
            size=yaxis_label_text_size - 2,
            family=config["coposition"]["subtitle-font-family"],
        ),
        zeroline=True,
        zerolinewidth=y_axis_zero["width"],
        zerolinecolor=y_axis_zero["color"],
        row=row,
        col=col,
    )

    return fig


def get_dynamic_subtitle(base_subtitle: str, is_healthy: bool, position: int) -> str:
    """
    Generate dynamic subtitle based on condition.

    Args:
        base_subtitle: Base subtitle from config
        is_healthy: Whether subject is healthy
        position: Position index (for ILR plots: 2 for ipsi/same, 4 for contra)

    Returns:
        Formatted subtitle with appropriate class labels
    """
    # For ILR Space plots, add class-specific labels
    if "ILR Space" in base_subtitle:
        if position == 2:  # First ILR plot
            if is_healthy:
                return base_subtitle.replace("Ipsilateral", "Ipsilateral (L-L vs R-R)")
            else:
                return base_subtitle.replace("Ipsilateral", "Same Side (A-A vs S-S)")
        elif position == 4:  # Second ILR plot
            if is_healthy:
                return base_subtitle.replace(
                    "Contralateral", "Contralateral (L-R vs R-L)"
                )
            else:
                return base_subtitle.replace(
                    "Contralateral", "Contra Side (A-S vs S-A)"
                )

    return base_subtitle


def main():
    """Main function to generate all plots"""

    # Load configuration
    config = load_config()
    cycles = config["cycles"]
    comp_config = config["coposition"]

    # Extract output folder path
    figs_output_path = Path("../figs/composition_publication")
    figs_output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    metrics_df = pd.read_csv(METRICS_FILE)
    pvals_df = pd.read_csv(PVALS_ILR_FILE)
    subject_df = pd.read_csv(SUBJECT_DATA_FILE)

    # Define plots
    PLOTS = [
        # Healthy subjects
        {
            "name": "Healthy",
            "condition": "healthy",
            "prosthesis": "none",
            "classes": ["R-R", "L-L", "R-L", "L-R"],
            "pairs": [
                ("R-R", "L-L"),
                ("R-L", "L-R"),
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

        for cycle in cycles:
            # Determine layout based on condition
            is_healthy = condition == "healthy"

            # Get base subtitles from config and make them dynamic
            base_subtitles = comp_config["subtitles"]
            subplot_titles = [
                get_dynamic_subtitle(base_subtitles[i], is_healthy, i)
                for i in range(len(base_subtitles))
            ]

            # Define which classes to use for ILR scatter plots
            if is_healthy:
                ips_clases = ["R-R", "L-L"]
                contra_clases = ["R-L", "L-R"]
            else:
                ips_clases = ["A-A", "S-S"]
                contra_clases = ["A-S", "S-A"]

            # Create figure with subplots
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

            # Add subplots using config parameters
            fig = make_proportion_barplot(
                metrics_df,
                condition,
                prosthesis,
                cycle,
                classes,
                fig,
                row=1,
                col=1,
                config=config,
            )

            fig = make_omnibus_plot(
                pvals_df, prosthesis, cycle, pairs, fig, row=1, col=2, config=config
            )

            fig = make_ilr_coordinates_plot(
                pvals_df, prosthesis, cycle, pairs, fig, row=2, col=2, config=config
            )

            fig = make_proportions_posthoc_plot(
                pvals_df, prosthesis, cycle, pairs, fig, row=3, col=2, config=config
            )

            # Add ILR scatter plots (column 3)
            fig = make_ilr_scatter_plot(
                subject_df,
                prosthesis,
                cycle,
                classes=contra_clases,
                fig=fig,
                row=2,
                col=3,
                config=config,
            )

            fig = make_ilr_scatter_plot(
                subject_df,
                prosthesis,
                cycle,
                classes=ips_clases,
                fig=fig,
                row=1,
                col=3,
                config=config,
            )

            # Generate dynamic title with cycle information
            if cycle == "cycle":
                cycle_text = "Complete Cycle"
            elif cycle == "swing":
                cycle_text = "Swing Phase"
            else:
                cycle_text = "Stance Phase"

            # Build title from config base + dynamic cycle info
            base_title = comp_config["title"]
            title = f"{base_title} - <b>{cycle_text}</b>"

            # Get export dimensions from config
            figure_width = comp_config["export-html"]["width"]
            figure_height = comp_config["export-html"]["height"]

            # Get legend configuration
            composition_legend = comp_config.get("composition-legend", {})
            scatter_legend = comp_config.get("scatter-legend", {})

            showlegend = True

            # Update layout with config parameters
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {
                        "size": comp_config["title-font-size"],
                        "family": comp_config["title-font-family"],
                        "color": comp_config["title-text-color"],
                    },
                },
                barmode="stack",
                height=figure_height,
                width=figure_width,
                font=dict(
                    size=comp_config["subtitle-font-size"],
                    family=comp_config["subtitle-font-family"],
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=showlegend,
                legend=composition_legend,
                legend2=scatter_legend,
            )

            # Update subplot title fonts
            for i in fig["layout"]["annotations"]:
                for t in subplot_titles:
                    if t in i["text"]:
                        i["font"] = dict(
                            size=comp_config["subtitle-font-size"],
                            family=comp_config["subtitle-font-family"],
                            color=comp_config["subtitle-text-color"],
                        )

            # Save figure
            output_file = figs_output_path / f"pub_ilr_analysis_{name}_{cycle}.html"
            fig.write_html(str(output_file))
            print(f"Saved: {output_file}")

            # Save as PDF with config dimensions
            output_pdf = figs_output_path / f"pub_ilr_analysis_{name}_{cycle}.pdf"
            fig.write_image(
                str(output_pdf),
                width=comp_config["export-pdf"]["width"],
                height=comp_config["export-pdf"]["height"],
                scale=comp_config["export-pdf"]["scale"],
            )
            print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
