from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
METRICS_FILE = Path("../datos/csv/sim02_metrics.csv")
PVALS_ILR_FILE = Path("../datos/csv/sim02_composition_pvals.csv")

FIGS_OUTPUT_PATH = Path("./figs/figs_crossgroup/")

CYCLES = ["swing", "stance", "cycle"]

# Define colors
COLORS = {
    "toe_var_prop": "#3B82F6",  # Blue for H(T|A)
    "angle_var_prop": "#10B981",  # Green for H(A|T)
    "inter_dependence_prop": "#F59E0B",  # Orange for I(T;A)
}

VARIABLE_LABELS = {
    "toe_var_prop": "H(T|A)",
    "angle_var_prop": "H(A|T)",
    "inter_dependence_prop": "I(T;A)",
    "total_var": "H(T;A)",
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


def make_proportion_barplot_crossgroup(
    df: pd.DataFrame,
    conditions: list,
    prostheses: list,
    cycle: str,
    classes: list,
    fig: go.Figure,
    row: int,
    col: int,
):
    """Create stacked bar plot for cross-group comparisons"""

    # Extract data for each class
    data_by_class = {}
    total_var = []

    for class_name in classes:
        class_col = "amp_sound"
        # Determine which condition and prosthesis this class belongs to
        if class_name in ["A-A", "S-S", "A-S", "S-A"]:
            # Amputee class
            condition_filter = "amputee"
            prosthesis_filter = (
                prostheses[0] if prostheses[0] != "none" else prostheses[1]
            )
        else:
            # Healthy class (Same, Contra)
            condition_filter = "healthy"
            prosthesis_filter = "none"

        # Filter data
        class_data = df[
            (df["condition"] == condition_filter)
            & (df["prosthesis"] == prosthesis_filter)
            & (df["cycle"] == cycle)
            & (df[class_col] == class_name)
        ]

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

    w = 0.9

    # Add stacked bars
    fig.add_trace(
        go.Bar(
            name="H(T|A)",
            x=valid_classes,
            y=toe_var_prop,
            marker_color=COLORS["toe_var_prop"],
            text=[f"{p:.1f}% ({v:.2f})" for p, v in zip(toe_var_prop, toe_var)],
            textposition="inside",
            textfont=dict(color="white", size=12),
            showlegend=True,
            legendgroup="htoe",
            hovertemplate="H(T|A): %{y:.1f}% (%{customdata:.2f} nats)<extra></extra>",
            customdata=toe_var,
            width=w,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Bar(
            name="H(A|T)",
            x=valid_classes,
            y=angle_var_prop,
            marker_color=COLORS["angle_var_prop"],
            text=[f"{p:.1f}% ({v:.2f})" for p, v in zip(angle_var_prop, angle_var)],
            textposition="inside",
            textfont=dict(color="white", size=12),
            showlegend=True,
            legendgroup="hangle",
            hovertemplate="H(A|T): %{y:.1f}% (%{customdata:.2f} nats)<extra></extra>",
            customdata=angle_var,
            width=w,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Bar(
            name="I(T;A)",
            x=valid_classes,
            y=inter_dependence_prop,
            marker_color=COLORS["inter_dependence_prop"],
            text=[
                f"{p:.1f}% ({v:.2f})"
                for p, v in zip(inter_dependence_prop, inter_dependence)
            ],
            textposition="inside",
            textfont=dict(color="white", size=12),
            showlegend=True,
            legendgroup="coord",
            hovertemplate="I(T;A): %{y:.1f}% (%{customdata:.2f} nats)<extra></extra>",
            customdata=inter_dependence,
            width=w,
        ),
        row=row,
        col=col,
    )

    # Add H(T,A) annotations above bars
    for class_name, joint_ent in zip(valid_classes, total_var):
        fig.add_annotation(
            x=class_name,
            y=102,
            text=f"<b>{joint_ent:.2f}</b>",
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center",
            yanchor="bottom",
            row=row,
            col=col,
        )

    fig.update_yaxes(
        title_text="Composition (%)",
        title_font=dict(size=12),
        tickfont=dict(size=11),
        range=[0, 110],
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
):
    """Create plot showing omnibus test results"""

    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]

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

            # Fix p-value at 1e-4 if smaller
            pval = max(pval, 1e-4)
            log_pval = -np.log10(pval)

            sig = get_significance_symbol(pval)
            x_vals.append(log_pval)
            colors.append(get_pvalue_color(log_pval))
            hover_texts.append(f"{class1} vs {class2}<br>Omnibus p={pval:.4f}<br>{sig}")
            text_labels.append(sig)
            y_labels.append(f"{class1} vs {class2}")
        else:
            x_vals.append(0)
            colors.append("#d3d3d3")
            hover_texts.append("No data")
            text_labels.append("")
            y_labels.append(f"{class1} vs {class2}")

    # Add bars
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_labels,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="inside",
            textfont=dict(color="white", size=11),
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
        fig.add_vline(
            x=x_val, line_dash="dash", line_color="gray", line_width=1, row=row, col=col
        )
        fig.add_annotation(
            x=x_val,
            y=len(y_labels) - 0.5,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=10, color="gray"),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log10(p-value)", title_font=dict(size=12), row=row, col=col
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(y_labels))),
        ticktext=y_labels,
        tickfont=dict(size=11),
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
):
    """Create plot showing ILR coordinate tests with effect sizes inside bars"""

    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]

    dfc = df[(df["prosthesis"].isin(prosthesis)) & (df["cycle"] == cycle)]
    coord_names = ["ILR2", "ILR1"]
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
                for coord_name in coord_names:
                    pval = row_data[f"{coord_name}_pval_bonf"]
                    d = row_data[f"{coord_name}_cohens_d"]

                    # Fix p-value at 1e-4 if smaller
                    pval = max(pval, 1e-4)
                    log_pval = -np.log10(pval)

                    sig = get_significance_symbol(pval)
                    cat = get_effect_size_category(d)

                    x_vals.append(log_pval)
                    colors.append(get_pvalue_color(log_pval))
                    hover_texts.append(
                        f"{coord_name} p={pval:.4f} {sig}<br>d={d:.2f} ({cat})"
                    )
                    text_labels.append(f"{sig} | d={abs(d):.2f} ({cat})")
                    y_labels.append(f"{class1} vs {class2}<br>{coord_name}")
                    y_positions.append(y_pos)
                    y_pos += 1
            else:
                # Omnibus not significant
                for coord_name in coord_names:
                    x_vals.append(0)
                    colors.append("#d3d3d3")
                    hover_texts.append("Omnibus NS")
                    text_labels.append("")
                    y_labels.append(f"{class1} vs {class2}<br>{coord_name}")
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
            textfont=dict(color="white", size=11),
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
        fig.add_vline(
            x=x_val, line_dash="dash", line_color="gray", line_width=1, row=row, col=col
        )
        fig.add_annotation(
            x=x_val,
            y=max(y_positions) + 0.5 if y_positions else 1,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=10, color="gray"),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log10(p-value)", title_font=dict(size=12), row=row, col=col
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        tickfont=dict(size=11),
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
):
    """Create plot showing individual proportion tests with effect sizes inside bars"""

    if isinstance(prosthesis, str):
        prosthesis = [prosthesis]

    dfc = df[(df["prosthesis"].isin(prosthesis)) & (df["cycle"] == cycle)]
    prop_names = ["toe_var_prop", "angle_var_prop", "inter_dependence_prop"]
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
                for prop in prop_names:
                    pval = row_data[f"{prop}_pval_bonf"]
                    d_orig = row_data[f"{prop}_cohens_d_orig"]

                    # Fix p-value at 1e-4 if smaller
                    pval = max(pval, 1e-4)
                    log_pval = -np.log10(pval)

                    sig = get_significance_symbol(pval)
                    cat = get_effect_size_category(d_orig)

                    x_vals.append(log_pval)
                    colors.append(get_pvalue_color(log_pval))

                    hover_texts.append(
                        f"{VARIABLE_LABELS[prop]} p={pval:.4f} {sig}<br>d={d_orig:.2f} ({cat})"
                    )
                    text_labels.append(f"{sig} | d={abs(d_orig):.2f} ({cat})")
                    y_labels.append(f"{class1} vs {class2}<br>{VARIABLE_LABELS[prop]}")
                    y_positions.append(y_pos)
                    y_pos += 1
            else:
                # Omnibus not significant
                for prop in prop_names:
                    x_vals.append(0)
                    colors.append("#d3d3d3")
                    hover_texts.append("Omnibus NS")
                    text_labels.append("")
                    y_labels.append(f"{class1} vs {class2}<br>{VARIABLE_LABELS[prop]}")
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
            textfont=dict(color="white", size=10),
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
        fig.add_vline(
            x=x_val, line_dash="dash", line_color="gray", line_width=1, row=row, col=col
        )
        fig.add_annotation(
            x=x_val,
            y=max(y_positions) + 0.5 if y_positions else 1,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=10, color="gray"),
            row=row,
            col=col,
        )

    fig.update_xaxes(
        title_text="-log10(p-value)", title_font=dict(size=12), row=row, col=col
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        tickfont=dict(size=10),
        row=row,
        col=col,
    )

    return fig


if __name__ == "__main__":
    # Load data
    metrics_df = pd.read_csv(METRICS_FILE)
    pvals_df = pd.read_csv(PVALS_ILR_FILE)

    # Define cross-group plots
    PLOTS = [
        # Healthy: Same vs Contra
        {
            "name": "Healthy_Same_vs_Contra",
            "conditions": ["healthy"],
            "prostheses": ["none"],
            "classes": ["Same", "Contra"],
            "pairs": [
                ("Same", "Contra"),
            ],
            "title_suffix": "Healthy Subjects: Same vs Contra",
        },
        # Amputee (Mech) vs Healthy (Ipsilateral)
        {
            "name": "Amputee_Mech_vs_Healthy_Ipsilateral",
            "conditions": ["amputee", "healthy"],
            "prostheses": ["Mech", "none"],
            "classes": [
                "A-A",
                "S-S",
                "Same",
            ],
            "pairs": [
                ("A-A", "Same"),
                ("S-S", "Same"),
            ],
            "title_suffix": "Amputee (Mech) vs Healthy",
        },
        # Amputee (Mech) vs Healthy (Contralateral)
        {
            "name": "Amputee_Mech_vs_Healthy_contralateral",
            "conditions": ["amputee", "healthy"],
            "prostheses": ["Mech", "none"],
            "classes": ["A-S", "S-A", "Contra"],
            "pairs": [
                ("A-S", "Contra"),
                ("S-A", "Contra"),
            ],
            "title_suffix": "Amputee (Mech) vs Healthy",
        },
        # Amputee (Ech) vs Healthy
        {
            "name": "Amputee_Ech_vs_Healthy",
            "conditions": ["amputee", "healthy"],
            "prostheses": ["Ech", "none"],
            "classes": ["A-A", "S-S", "Same", "A-S", "S-A", "Contra"],
            "pairs": [
                ("A-A", "Same"),
                ("S-S", "Same"),
                # ("A-S", "Contra"),
                # ("S-A", "Contra"),
            ],
            "title_suffix": "Amputee (Ech) vs Healthy",
        },
    ]

    for plot_config in PLOTS:
        name = plot_config["name"]
        conditions = plot_config["conditions"]
        prostheses = plot_config["prostheses"]
        classes = plot_config["classes"]
        pairs = plot_config["pairs"]
        title_suffix = plot_config["title_suffix"]

        # Determine prosthesis string for filtering p-values
        if "none" in prostheses and len(prostheses) == 1:
            # Healthy only
            pval_prosthesis = ["none"]
        else:
            # Cross-group comparison
            pval_prosthesis = [f"{prostheses[0]}-none"]

        for cycle in CYCLES:
            # Create 2×2 subplot layout
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "A. Compositional Analysis by Class",
                    "B. Omnibus Test (Hotelling's T²)",
                    "C. ILR Coordinates Post-hoc",
                    "D. Individual Proportions Post-hoc (CLR)",
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}],
                ],
                horizontal_spacing=0.10,
                vertical_spacing=0.10,
                row_heights=[0.6, 0.4],
                column_widths=[0.5, 0.5],
            )

            # Add subplots
            fig = make_proportion_barplot_crossgroup(
                metrics_df, conditions, prostheses, cycle, classes, fig, row=1, col=1
            )

            fig = make_omnibus_plot(
                pvals_df, pval_prosthesis, cycle, pairs, fig, row=1, col=2
            )

            fig = make_ilr_coordinates_plot(
                pvals_df, pval_prosthesis, cycle, pairs, fig, row=2, col=1
            )

            fig = make_proportions_posthoc_plot(
                pvals_df, pval_prosthesis, cycle, pairs, fig, row=2, col=2
            )

            # Update layout
            title = f"Compositional Analysis (Balance ILR) - {title_suffix} - {cycle.capitalize()} Phase"

            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 20},
                },
                barmode="stack",
                height=900,
                width=1400,
                font=dict(size=12),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            # Save figure
            FIGS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            output_file = FIGS_OUTPUT_PATH / f"ilr_analysis_{name}_{cycle}.html"
            fig.write_html(str(output_file))
            print(f"Saved: {output_file}")

            # Uncomment to show figures interactively
            # fig.show()
