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


def replace_class_labels(class_name):
    """Replace Same with Ipsilateral and Contra with Contralateral"""
    if class_name == "Same":
        return "Ipsi"
    elif class_name == "Contra":
        return "Contra"
    else:
        return class_name


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
    """Create stacked bar plot showing proportions of H(T|A), H(A|T), and I(T;A)
    Adapted from barplot_composition_pub.py for crossgroup use"""

    plot_config = config["composition-crossgroup"]["proportions-bar-plot"]
    colors = plot_config["colors"]
    text_size = plot_config["text-size"]
    bar_width = plot_config["bar-width"]
    annotation_text_size = plot_config["annotation-text-size"]
    annotation_font_family = plot_config["annotation-font-family"]
    annotation_font_color = plot_config["annotation-font-color"]
    xaxis_label_text_size = plot_config["xaxis-label-text-size"]
    yaxis_label_text_size = plot_config["yaxis-label-text-size"]
    variable_labels = config["composition-crossgroup"]["variable-labels"]

    # Handle condition and prosthesis as lists
    if isinstance(condition, str):
        condition = [condition]
    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]

    # Filter data - Same and Contra are in amp_sound column
    dfc = df[
        (df["condition"].isin(condition))
        & (df["prosthesis"].isin(prosthesis))
        & (df["cycle"] == cycle)
    ]

    # Extract data for each class
    toe_var_prop = []
    angle_var_prop = []
    inter_dependence_prop = []
    toe_var = []
    angle_var = []
    inter_dependence = []
    total_var = []
    valid_classes = []

    for class_name in classes:
        # Filter by amp_sound column for Same/Contra classes
        class_data = dfc[dfc["amp_sound"] == class_name]

        if len(class_data) == 0:
            continue

        valid_classes.append(class_name)

        # Extract mean values
        toe_var_prop.append(class_data["toe_var_prop"].values[0] * 100)
        angle_var_prop.append(class_data["angle_var_prop"].values[0] * 100)
        inter_dependence_prop.append(
            class_data["inter_dependence_prop"].values[0] * 100
        )

        toe_var.append(class_data["toe_var"].values[0])
        angle_var.append(class_data["angle_var"].values[0])
        inter_dependence.append(class_data["inter_dependence"].values[0])
        total_var.append(class_data["total_var"].values[0])

    # Replace class labels for display
    display_classes = [replace_class_labels(c) for c in valid_classes]

    # Add stacked bars - Toe variation
    fig.add_trace(
        go.Bar(
            name=variable_labels["toe_var_prop"],
            x=display_classes,
            y=toe_var_prop,
            marker_color=colors["toe_var_prop"],
            text=[format_bar_text(p, v) for p, v in zip(toe_var_prop, toe_var)],
            textposition="inside",
            textfont=dict(
                color=annotation_font_color,
                size=text_size,
                family=annotation_font_family,
            ),
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

    # Add stacked bars - Angle variation
    fig.add_trace(
        go.Bar(
            name=variable_labels["angle_var_prop"],
            x=display_classes,
            y=angle_var_prop,
            marker_color=colors["angle_var_prop"],
            text=[format_bar_text(p, v) for p, v in zip(angle_var_prop, angle_var)],
            textposition="inside",
            textfont=dict(
                color=annotation_font_color,
                size=text_size,
                family=annotation_font_family,
            ),
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

    # Add stacked bars - Inter-dependence
    fig.add_trace(
        go.Bar(
            name=variable_labels["inter_dependence_prop"],
            x=display_classes,
            y=inter_dependence_prop,
            marker_color=colors["inter_dependence_prop"],
            text=[
                format_bar_text(p, v)
                for p, v in zip(inter_dependence_prop, inter_dependence)
            ],
            textposition="inside",
            textfont=dict(
                color=annotation_font_color,
                size=text_size,
                family=annotation_font_family,
            ),
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
    for display_class, joint_ent in zip(display_classes, total_var):
        fig.add_annotation(
            x=display_class,
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
    """Create plot showing omnibus test results
    Adapted from barplot_composition_pub.py"""

    plot_config = config["composition-crossgroup"]["omnibus-plot"]
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
        # Replace labels
        display_class1 = replace_class_labels(class1)
        display_class2 = replace_class_labels(class2)

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
            y_labels.append(f"{display_class1} vs {display_class2}")
        else:
            x_vals.append(0)
            colors.append("#d3d3d3")
            hover_texts.append("No data")
            text_labels.append("")
            y_labels.append(f"{display_class1} vs {display_class2}")

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
    """Create plot showing ILR coordinate post-hoc tests
    Adapted from barplot_composition_pub.py"""

    plot_config = config["composition-crossgroup"]["ilr-coordinates-plot"]
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
        # Replace labels
        display_class1 = replace_class_labels(class1)
        display_class2 = replace_class_labels(class2)

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
                    y_labels.append(
                        f"{display_class1} vs {display_class2}<br><b>{ilr}</b>"
                    )
                    y_positions.append(y_pos)
                    y_pos += 1
            else:
                # Omnibus not significant
                for coord in ILR_COLUMNS:
                    x_vals.append(0)
                    colors.append("#d3d3d3")
                    hover_texts.append("Omnibus NS")
                    text_labels.append("")
                    y_labels.append(
                        f"{display_class1} vs {display_class2}<br><b>{coord}</b>"
                    )
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
    """Create plot showing individual proportion tests
    Adapted from barplot_composition_pub.py"""

    plot_config = config["composition-crossgroup"]["clr-posthoc-plot"]
    variable_labels = config["composition-crossgroup"]["variable-labels"]
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
        # Replace labels
        display_class1 = replace_class_labels(class1)
        display_class2 = replace_class_labels(class2)

        pair_str = f"({class1})-({class2})"
        comp_data = dfc[dfc["pair"] == pair_str]

        if len(comp_data) > 0:
            row_data = comp_data.iloc[0]

            if row_data["omnibus_significant"]:
                for prop in CLR_COLUMNS:
                    pval = row_data[f"{prop}_pval_bonf"]
                    d_orig = row_data[f"{prop}_cohens_d"]
                    pval = max(pval, 1e-4)
                    log_pval = -np.log10(pval)
                    sig = get_significance_symbol(pval)
                    cat = get_effect_size_category(d_orig)

                    x_vals.append(log_pval)
                    colors.append(get_pvalue_color(log_pval))
                    hover_texts.append(
                        f"{variable_labels[prop]}: p={pval:.4f} {sig}<br>d = {d_orig:.2f} ({cat})"
                    )
                    text_labels.append(f"<b>d = {d_orig:.2f} ({cat})</b>")
                    y_labels.append(
                        f"{display_class1} vs {display_class2}<br><b>{variable_labels[prop]}</b>"
                    )
                    y_positions.append(y_pos)
                    y_pos += 1
            else:
                # Omnibus not significant
                for prop in CLR_COLUMNS:
                    x_vals.append(0)
                    colors.append("#d3d3d3")
                    hover_texts.append("Omnibus NS")
                    text_labels.append("")
                    y_labels.append(
                        f"{display_class1} vs {display_class2}<br><b>{variable_labels[prop]}</b>"
                    )
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
    """Create ILR scatter plot for specified classes
    Adapted from barplot_composition_pub.py"""

    plot_config = config["composition-crossgroup"]["ilr-scatter-plot"]
    class_colors = plot_config["colors"]
    marker_size = plot_config["marker-size"]
    xaxis_label = plot_config["xaxis-label"]
    xaxis_label_text_size = plot_config["xaxis-label-text-size"]
    yaxis_label = plot_config["yaxis-label"]
    yaxis_label_text_size = plot_config["yaxis-label-text-size"]
    x_axis_zero = plot_config["xaxis-zeroline"]
    y_axis_zero = plot_config["yaxis-zeroline"]

    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]

    # Filter data - Same and Contra are in amp_sound column for healthy subjects
    dfc = subject_df[
        (subject_df["prosthesis"].isin(prosthesis)) & (subject_df["cycle"] == cycle)
    ]

    if len(dfc) == 0:
        return fig

    xmin = dfc["ilr1"].min()
    xmax = dfc["ilr1"].max()
    ymin = dfc["ilr2"].min()
    ymax = dfc["ilr2"].max()

    # Plot each class
    for class_name in classes:
        # Replace labels for display
        display_class = replace_class_labels(class_name)

        # Filter by amp_sound column (Same/Contra are here for healthy subjects)
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
                name=display_class,
                marker=dict(
                    size=marker_size,
                    color=class_colors.get(class_name, "#333333"),
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=class_data["subject"],
                hovertemplate=(
                    f"<b>{display_class}</b><br>"
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
            family=config["composition-crossgroup"]["subtitle-font-family"],
        ),
        tickfont=dict(
            size=xaxis_label_text_size - 2,
            family=config["composition-crossgroup"]["subtitle-font-family"],
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
            family=config["composition-crossgroup"]["subtitle-font-family"],
        ),
        tickfont=dict(
            size=yaxis_label_text_size - 2,
            family=config["composition-crossgroup"]["subtitle-font-family"],
        ),
        zeroline=True,
        zerolinewidth=y_axis_zero["width"],
        zerolinecolor=y_axis_zero["color"],
        row=row,
        col=col,
    )

    return fig


def main():
    """Main function to generate all plots"""

    # Load configuration
    config = load_config()
    cycles = config["cycles"]
    comp_config = config["composition-crossgroup"]

    # Extract output folder path
    figs_output_path = Path("../figs/figs_crossgroup/")
    figs_output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    metrics_df = pd.read_csv(METRICS_FILE)
    pvals_df = pd.read_csv(PVALS_ILR_FILE)
    subject_df = pd.read_csv(SUBJECT_DATA_FILE)

    # Define cross-group plots
    PLOTS = [
        # Healthy: Same vs Contra (Ipsilateral vs Contralateral)
        {
            "name": "Healthy_Ipsi_vs_Contra",
            "conditions": ["healthy"],
            "prostheses": ["none"],
            "classes": ["Same", "Contra"],
            "pairs": [
                ("Same", "Contra"),
            ],
            "title_suffix": "Ipsilateral vs Contralateral",
        },
    ]

    for plot_config_item in PLOTS:
        name = plot_config_item["name"]
        conditions = plot_config_item["conditions"]
        prostheses = plot_config_item["prostheses"]
        classes = plot_config_item["classes"]
        pairs = plot_config_item["pairs"]
        title_suffix = plot_config_item["title_suffix"]

        # Determine prosthesis string for filtering p-values
        if "none" in prostheses and len(prostheses) == 1:
            pval_prosthesis = ["none"]
        else:
            pval_prosthesis = [f"{prostheses[0]}-none"]

        for cycle in cycles:
            # Get base subtitles from config
            subplot_titles = comp_config["subtitles"]

            # Create 3x3 subplot layout
            # Col1: Barplots (3 rows tall)
            # Col2 Row1: Omnibus, Row2: ILR post-hoc, Row3: CLR post-hoc
            # Col3 Row1-2: Scatter (2 rows tall), Row3: Legends placeholder
            fig = make_subplots(
                rows=3,
                cols=3,
                subplot_titles=subplot_titles,
                specs=[
                    [
                        {"rowspan": 3, "type": "bar"},
                        {"type": "bar"},
                        {"rowspan": 2, "type": "scatter"},
                    ],
                    [None, {"type": "bar"}, None],
                    [
                        None,
                        {"type": "bar"},
                        None,
                    ],  # Row 3 Col 3 is for legends (no subplot)
                ],
                horizontal_spacing=0.10,
                vertical_spacing=0.10,
                row_heights=[0.33, 0.33, 0.34],
                column_widths=[0.35, 0.35, 0.30],
            )

            # Add subplots
            # A. Barplots (col 1, rows 1-3)
            fig = make_proportion_barplot(
                metrics_df,
                conditions,
                prostheses,
                cycle,
                classes,
                fig,
                row=1,
                col=1,
                config=config,
            )

            # B. Omnibus (col 2, row 1)
            fig = make_omnibus_plot(
                pvals_df,
                pval_prosthesis,
                cycle,
                pairs,
                fig,
                row=1,
                col=2,
                config=config,
            )

            # ILR post-hoc (col 2, row 2)
            fig = make_ilr_coordinates_plot(
                pvals_df,
                pval_prosthesis,
                cycle,
                pairs,
                fig,
                row=2,
                col=2,
                config=config,
            )

            # C. CLR post-hoc (col 2, row 3)
            fig = make_proportions_posthoc_plot(
                pvals_df,
                pval_prosthesis,
                cycle,
                pairs,
                fig,
                row=3,
                col=2,
                config=config,
            )

            # E. Scatter plot (col 3, rows 1-2)
            fig = make_ilr_scatter_plot(
                subject_df,
                prostheses,
                cycle,
                classes,
                fig,
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
            title = f"{base_title} - {title_suffix} - <b>{cycle_text}</b>"

            # Get export dimensions from config
            figure_width = comp_config["export-html"]["width"]
            figure_height = comp_config["export-html"]["height"]

            # Get legend configuration
            composition_legend = comp_config.get("composition-legend", {})
            scatter_legend = comp_config.get("scatter-legend", {})

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
                showlegend=True,
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
            output_file = (
                figs_output_path / f"crossgroup_ilr_analysis_{name}_{cycle}.html"
            )
            fig.write_html(str(output_file))
            print(f"Saved: {output_file}")

            # Save as PDF with config dimensions
            output_pdf = (
                figs_output_path / f"crossgroup_ilr_analysis_{name}_{cycle}.pdf"
            )
            fig.write_image(
                str(output_pdf),
                width=comp_config["export-pdf"]["width"],
                height=comp_config["export-pdf"]["height"],
                scale=comp_config["export-pdf"]["scale"],
            )
            print(f"Saved: {output_pdf}")


if __name__ == "__main__":
    main()
