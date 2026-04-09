"""
Publication-quality figures for Deloitte Quantum Sustainability Challenge 2026.
Run from repo root: python figures/generate_figures.py
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Palette ─────────────────────────────────────────────────────────────────
GREEN   = "#86BC25"
BLUE    = "#0076A8"
GRAY    = "#333333"
ORANGE  = "#E87722"
LGREEN  = "#C4D600"
DGRAY   = "#666666"
RED     = "#C0392B"

DOMAIN_COLORS = {
    "Fire History":   ORANGE,
    "Weather":        BLUE,
    "Insurance":      GREEN,
    "Socioeconomic":  DGRAY,
}

OUT = os.path.join(os.path.dirname(__file__))
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

SAVEKW = dict(dpi=300, bbox_inches="tight")

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, **SAVEKW)
    print(f"  saved → {name}")
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Pipeline Overview
# ════════════════════════════════════════════════════════════════════════════
def fig1_pipeline():
    print("Figure 1: Pipeline Overview")

    # Real counts
    fm = pd.read_csv(os.path.join(DATA, "feature_matrix_clean.csv"))
    n_features = len([c for c in fm.columns if c not in ("zip", "Year", "had_fire")])
    t1 = pd.read_csv(os.path.join(DATA, "task1_predictions_2023.csv"))
    t2 = pd.read_csv(os.path.join(DATA, "task2_predictions_2021_final.csv"))
    n_zips = len(t1)
    r2_val = r2_score(t2["Actual_Premium"], t2["Predicted_Premium"])

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(ax, x, y, w, h, label, subtitle, color, text_color="white", fontsize=10):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor="white",
                               linewidth=1.5, zorder=3)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*0.62, label,
                ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x + w/2, y + h*0.28, subtitle,
                ha="center", va="center", color=text_color,
                fontsize=7.5, zorder=4, style="italic")

    def arrow(ax, x1, y1, x2, y2, color="#999999"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.8, mutation_scale=16),
                    zorder=2)

    # Box positions (x, y, w, h)
    bh = 1.1   # box height
    bw = 1.85  # box width
    mid_y = (4.2 - bh) / 2  # ~1.55

    # Box 1 — Raw Data
    box(ax, 0.2, mid_y, bw, bh,
        "Raw Data",
        "Wildfire 125,476 rows\nInsurance 47,033 rows",
        GRAY)

    # Arrow 1→2
    arrow(ax, 0.2 + bw, mid_y + bh/2, 2.3, mid_y + bh/2)

    # Box 2 — Feature Engineering
    box(ax, 2.3, mid_y, bw, bh,
        "Feature Engineering",
        f"{n_features} features · 4 domains",
        DGRAY)

    # Split arrow top (→ Classical)
    ax.annotate("", xy=(4.4, mid_y + bh + 0.32), xytext=(2.3 + bw, mid_y + bh/2),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.8, mutation_scale=16), zorder=2)
    # Split arrow bottom (→ Quantum)
    ax.annotate("", xy=(4.4, mid_y - 0.32), xytext=(2.3 + bw, mid_y + bh/2),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.8, mutation_scale=16), zorder=2)

    # Box 3a — Classical (upper)
    box(ax, 4.4, mid_y + bh + 0.32 - bh/2, bw, bh,
        "Classical Models",
        "XGBoost · Random Forest",
        BLUE)

    # Box 3b — Quantum (lower)
    box(ax, 4.4, mid_y - 0.32 - bh/2, bw, bh,
        "Quantum Models",
        "5 VQC variants · 10 qubits",
        GREEN)

    # Merge arrows → Box 4
    upper_cx = 4.4 + bw
    lower_cx = 4.4 + bw
    merge_x  = 7.0
    ax.annotate("", xy=(merge_x, mid_y + bh/2), xytext=(upper_cx, mid_y + bh + 0.32),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.8, mutation_scale=16), zorder=2)
    ax.annotate("", xy=(merge_x, mid_y + bh/2), xytext=(lower_cx, mid_y - 0.32),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.8, mutation_scale=16), zorder=2)

    # Box 4 — 2023 Fire Risk
    box(ax, merge_x, mid_y, bw, bh,
        "Task 1: Fire Risk",
        f"{n_zips:,} ZIP codes\n2023 predictions",
        ORANGE)

    # Arrow 4→5
    arrow(ax, merge_x + bw, mid_y + bh/2, 9.15, mid_y + bh/2)

    # Box 5 — Task 2
    box(ax, 9.15, mid_y, bw + 0.2, bh,
        "Task 2: Premiums",
        f"Insurance prediction\nR² = {r2_val:.3f}",
        BLUE)

    # Title
    ax.text(6.5, 4.0, "Wildfire Risk & Insurance Premium Prediction Pipeline",
            ha="center", va="center", fontsize=12, fontweight="bold", color=GRAY)

    fig.tight_layout()
    savefig(fig, "fig1_pipeline.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — WildfireQCircuit Entanglement Topology
# ════════════════════════════════════════════════════════════════════════════
def fig2_circuit_topology():
    print("Figure 2: Circuit Topology")
    import networkx as nx

    QUBITS = {
        0: ("prior_total_acres",          "Fire History"),
        1: ("prior_max_acres",            "Fire History"),
        2: ("High Fire Risk\nExposure",   "Fire History"),   # HUB
        3: ("min_precip",                 "Weather"),
        4: ("Earned Premium",             "Insurance"),
        5: ("CAT Cov A\nFire Claims",     "Insurance"),
        6: ("Avg PPC",                    "Insurance"),
        7: ("Renter\nOccupied",           "Socioeconomic"),
        8: ("Bachelor+\nEdu",             "Socioeconomic"),
        9: ("Housing\nVacancy",           "Socioeconomic"),
    }

    WITHIN = [(0,1),(1,2),(0,2), (4,5),(5,6),(4,6), (7,8),(8,9),(7,9)]
    CROSS  = [(3,0),(3,2),(4,2),(8,2),(7,2)]

    # Manual layout — clusters spatially grouped
    pos = {
        0: (-2.2,  1.0),
        1: (-2.2, -0.5),
        2: (-0.5,  0.25),   # hub
        3: (-0.5,  2.5),    # Weather (top)
        4: ( 2.2,  1.5),    # Insurance (right)
        5: ( 2.2,  0.0),
        6: ( 2.2, -1.5),
        7: ( 0.5, -2.8),    # Socioeconomic (bottom)
        8: (-0.5, -2.8),
        9: ( 1.5, -2.8),
    }

    # External label offsets: (dx, dy, ha, va)
    LABEL_OFFSET = {
        0: (-0.42,  0.0,  "right",  "center"),
        1: (-0.42,  0.0,  "right",  "center"),
        2: ( 0.0,  -0.52, "center", "top"),
        3: ( 0.0,   0.48, "center", "bottom"),
        4: ( 0.42,  0.0,  "left",   "center"),
        5: ( 0.42,  0.0,  "left",   "center"),
        6: ( 0.42,  0.0,  "left",   "center"),
        7: ( 0.0,  -0.45, "center", "top"),
        8: ( 0.0,  -0.45, "center", "top"),
        9: ( 0.0,  -0.45, "center", "top"),
    }

    G = nx.Graph()
    G.add_nodes_from(range(10))
    G.add_edges_from(WITHIN)
    G.add_edges_from(CROSS)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Edges
    nx.draw_networkx_edges(G, pos, edgelist=CROSS,
                           style="dashed", width=1.5, edge_color="#AAAAAA",
                           ax=ax, alpha=0.85)
    nx.draw_networkx_edges(G, pos, edgelist=WITHIN,
                           style="solid", width=2.5, edge_color="#555555",
                           ax=ax)

    # Nodes — smaller so they don't crowd labels
    node_colors = [DOMAIN_COLORS[QUBITS[n][1]] for n in range(10)]
    node_sizes  = [900 if n == 2 else 550 for n in range(10)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax,
                           linewidths=1.8, edgecolors="white")

    # Qubit number inside node (white, small)
    for n, (x, y) in pos.items():
        ax.text(x, y, f"q{n}",
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=5)

    # Feature name label outside node (black, on white background)
    for n, (x, y) in pos.items():
        dx, dy, ha, va = LABEL_OFFSET[n]
        ax.text(x + dx, y + dy, QUBITS[n][0],
                ha=ha, va=va,
                fontsize=8.2, fontweight="bold", color="#000000",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.75))

    # Legend
    legend_handles = [
        mpatches.Patch(color=DOMAIN_COLORS[d], label=d)
        for d in DOMAIN_COLORS
    ]
    legend_handles += [
        plt.Line2D([0],[0], color="#555555", linewidth=2.5,
                   label="Within-cluster CZ gate"),
        plt.Line2D([0],[0], color="#AAAAAA", linewidth=1.5,
                   linestyle="dashed", label="Cross-cluster CZ gate"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, framealpha=0.95, edgecolor="#CCCCCC")

    ax.set_title("WildfireQCircuit — 10-Qubit CZ Entanglement Topology",
                 fontsize=13, fontweight="bold", color=GRAY, pad=12)
    ax.text(0, -3.65,
            "q2 (High Fire Risk Exposure) is the hub node — connected to all 4 domain clusters",
            ha="center", fontsize=9, color=DGRAY, style="italic")
    ax.axis("off")
    ax.set_xlim(-3.4, 3.8)
    ax.set_ylim(-4.1, 3.3)

    fig.tight_layout()
    savefig(fig, "fig2_circuit_topology.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Model Comparison Bar Chart
# ════════════════════════════════════════════════════════════════════════════
def fig3_model_comparison():
    print("Figure 3: Model Comparison")

    models = [
        # (label,              f1,    type)
        ("XGBoost\n43-feat",   0.509, "classical"),
        ("XGBoost\ntop-10",    0.488, "classical"),
        ("Random Forest\n43-feat", 0.473, "classical"),
        ("Q2 Reordered\n(best quantum)", 0.337, "quantum_best"),
        ("Q1 Baseline",        0.285, "quantum"),
        ("Q4 WQC Tuned",       0.284, "quantum"),
        ("Q5 Random",          0.262, "quantum"),
        ("Q3 WildfireQCircuit",0.249, "quantum"),
    ]

    labels = [m[0] for m in models]
    f1s    = [m[1] for m in models]
    types  = [m[2] for m in models]

    colors = []
    edges  = []
    lws    = []
    for t in types:
        if t == "classical":
            colors.append(BLUE)
            edges.append(BLUE)
            lws.append(0)
        elif t == "quantum_best":
            colors.append(GREEN)
            edges.append(GRAY)
            lws.append(2.2)
        else:
            colors.append(GREEN)
            edges.append(GREEN)
            lws.append(0)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(labels))
    bars = ax.bar(x, f1s, color=colors, edgecolor=edges, linewidth=lws,
                  width=0.62, zorder=3)

    # Q1 baseline reference line
    ax.axhline(0.285, color=ORANGE, linewidth=1.5, linestyle="--",
               label="Q1 Baseline (0.285)", zorder=4)

    # Value labels on top of bars
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=GRAY)

    # Divider between classical and quantum
    ax.axvline(2.5, color="#DDDDDD", linewidth=1.5, linestyle=":", zorder=2)
    ax.text(1.0, 0.54, "Classical", ha="center", fontsize=10,
            color=BLUE, fontweight="bold", alpha=0.7)
    ax.text(5.5, 0.54, "Quantum", ha="center", fontsize=10,
            color=GREEN, fontweight="bold", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=11, color=GRAY)
    ax.set_ylim(0, 0.60)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")

    # Legend patches
    leg_handles = [
        mpatches.Patch(color=BLUE,   label="Classical model"),
        mpatches.Patch(color=GREEN,  label="Quantum model (VQC)"),
        mpatches.Patch(facecolor=GREEN, edgecolor=GRAY, linewidth=2,
                       label="Best quantum (Q2)"),
        plt.Line2D([0],[0], color=ORANGE, linewidth=1.5, linestyle="--",
                   label="Q1 Baseline"),
    ]
    ax.legend(handles=leg_handles, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_title("F1 Score Comparison — Classical vs. Quantum Models\n"
                 "(Train: 2018–2020 · Test: 2021)",
                 fontsize=12, fontweight="bold", color=GRAY, pad=10)

    fig.tight_layout()
    savefig(fig, "fig3_model_comparison.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — 2023 Fire Capture Validation
# ════════════════════════════════════════════════════════════════════════════
def fig4_capture_rate():
    print("Figure 4: Capture Rate")

    df = pd.read_csv(os.path.join(DATA, "task1_predictions_2023.csv"))
    total_fires = int(df["Known_2023_Fire"].sum())

    rows = []

    # Classical: top-500 by probability
    for name, prob_col, color in [("XGBoost Top-500", "XGBoost_Probability", BLUE),
                                   ("Random Forest Top-500", "RF_Probability", BLUE)]:
        top500 = df.nlargest(500, prob_col)
        captured = int(top500["Known_2023_Fire"].sum())
        rows.append((name, captured, color))

    # Quantum: direct predictions
    q_map = {
        "Q1 Baseline":         "Q1_Prediction",
        "Q2 Reordered (best)": "Q2_Prediction",
        "Q3 WildfireQCircuit": "Q3_Prediction",
        "Q4 WQC Tuned":        "Q4_Prediction",
        "Q5 Random":           "Q5_Prediction",
    }
    for name, col in q_map.items():
        captured = int(df.loc[(df[col] == 1) & (df["Known_2023_Fire"] == 1)].shape[0])
        rows.append((name, captured, GREEN))

    # Sort by capture count
    rows.sort(key=lambda r: r[1])

    labels   = [r[0] for r in rows]
    captures = [r[1] for r in rows]
    colors   = [r[2] for r in rows]
    pcts     = [c / total_fires * 100 for c in captures]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(labels))
    bars = ax.barh(y, captures, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.6, zorder=3)

    # Total fires reference line
    ax.axvline(total_fires, color=RED, linewidth=1.8, linestyle="--",
               label=f"Total known 2023 fires ({total_fires})", zorder=4)

    # Percentage labels
    for bar, cap, pct in zip(bars, captures, pcts):
        ax.text(cap + 1.0, bar.get_y() + bar.get_height()/2,
                f"{cap} ({pct:.0f}%)", va="center", ha="left",
                fontsize=9, fontweight="bold", color=GRAY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Known 2023 Fires Captured", fontsize=11, color=GRAY)
    ax.set_xlim(0, total_fires * 1.35)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")

    leg_handles = [
        mpatches.Patch(color=BLUE,  label="Classical (Top-500 by probability)"),
        mpatches.Patch(color=GREEN, label="Quantum (direct prediction)"),
        plt.Line2D([0],[0], color=RED, linewidth=1.8, linestyle="--",
                   label=f"Total known 2023 fires ({total_fires})"),
    ]
    ax.legend(handles=leg_handles, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_title("2023 Wildfire Capture Validation\n"
                 "(Models trained on 2018–2022 data, validated on known 2023 fires)",
                 fontsize=12, fontweight="bold", color=GRAY, pad=10)

    fig.tight_layout()
    savefig(fig, "fig4_capture_rate.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Task 2 Predicted vs Actual Scatter
# ════════════════════════════════════════════════════════════════════════════
def fig5_task2_scatter():
    print("Figure 5: Task 2 Scatter")

    df = pd.read_csv(os.path.join(DATA, "task2_predictions_2021_final.csv"))

    # Filter to premium >= $10,000
    mask = (df["Actual_Premium"] >= 10_000) & (df["Predicted_Premium"] >= 10_000)
    dfm = df[mask].copy()

    actual    = dfm["Actual_Premium"].values
    predicted = dfm["Predicted_Premium"].values

    r2   = r2_score(actual, predicted)
    ape  = np.abs(actual - predicted) / actual
    mape = ape.mean() * 100
    med_ape = np.median(ape) * 100

    fig, ax = plt.subplots(figsize=(7, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(actual, predicted, s=10, alpha=0.35, color=BLUE,
               linewidths=0, zorder=3)

    # y = x line
    lims = [min(actual.min(), predicted.min()) * 0.85,
            max(actual.max(), predicted.max()) * 1.15]
    ax.plot(lims, lims, color=RED, linewidth=1.8, linestyle="--",
            label="Perfect prediction (y = x)", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Actual Earned Premium ($)", fontsize=11, color=GRAY)
    ax.set_ylabel("Predicted Earned Premium ($)", fontsize=11, color=GRAY)

    # Annotation box
    stats_txt = f"R² = {r2:.3f}\nMAPE = {mape:.1f}%\nMedian APE = {med_ape:.1f}%\nn = {len(dfm):,} ZIP codes"
    ax.text(0.04, 0.96, stats_txt, transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.95),
            color=GRAY, fontweight="bold")

    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#CCCCCC",
              loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(axis="both", which="both", labelsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, which="both", zorder=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, which="both", zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Task 2: Predicted vs. Actual Insurance Premium (2021)\n"
                 "XGBoost Regression — ZIP-code level",
                 fontsize=12, fontweight="bold", color=GRAY, pad=10)

    fig.tight_layout()
    savefig(fig, "fig5_task2_scatter.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Feature Importance (Top 15)
# ════════════════════════════════════════════════════════════════════════════
def fig6_feature_importance():
    print("Figure 6: Feature Importance")

    fm = pd.read_csv(os.path.join(DATA, "feature_matrix_clean.csv"))
    feature_cols = [c for c in fm.columns if c not in ("zip", "Year", "had_fire")]

    # Domain mapping
    fire_cols = {"prior_fire_count", "prior_total_acres", "prior_max_acres",
                 "had_prior_fire", "prior_total_acres_log", "prior_max_acres_log"}
    weather_cols = {"mean_tmax", "max_tmax", "mean_tmin", "min_tmin",
                    "total_precip", "mean_precip", "min_precip", "temp_range",
                    "fire_season_mean_tmax", "fire_season_max_tmax",
                    "fire_season_total_precip"}
    insurance_cols = {"Avg Fire Risk Score", "Avg PPC", "Earned Exposure", "Earned Premium",
                      "Cov A Amount Weighted Avg", "Cov C Amount Weighted Avg",
                      "CAT Cov A Fire -  Incurred Losses", "CAT Cov A Fire -  Number of Claims",
                      "Non-CAT Cov A Fire -  Incurred Losses", "Non-CAT Cov A Fire -  Number of Claims",
                      "Number of High Fire Risk Exposure", "Number of Very High Fire Risk Exposure",
                      "Number of Low Fire Risk Exposure", "Number of Moderate Fire Risk Exposure",
                      "Number of Negligible Fire Risk Exposure"}

    def domain(col):
        if col in fire_cols:         return "Fire History"
        if col in weather_cols:      return "Weather"
        if col in insurance_cols:    return "Insurance"
        return "Socioeconomic"

    # Train on 2018–2020, test on 2021 — same split as notebooks
    train = fm[fm["Year"].isin([2018, 2019, 2020])].copy()
    X_tr  = train[feature_cols].fillna(0)
    y_tr  = train["had_fire"]

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6,
        scale_pos_weight=pos_weight,
        learning_rate=0.1, random_state=42,
        eval_metric="logloss", verbosity=0,
        use_label_encoder=False,
    )
    model.fit(X_tr, y_tr)

    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top15 = imp.nlargest(15).sort_values()

    colors = [DOMAIN_COLORS[domain(c)] for c in top15.index]

    # Shorten display labels
    def shorten(c):
        return (c.replace("Number of ", "# ")
                 .replace(" Fire Risk Exposure", " Risk Exp.")
                 .replace("CAT Cov A Fire -  ", "CAT ")
                 .replace("Non-CAT Cov A Fire -  ", "Non-CAT ")
                 .replace("prior_", "")
                 .replace("fire_season_", "fs_")
                 .replace("educational_attainment_", "edu_")
                 .replace("_", " "))

    labels_short = [shorten(c) for c in top15.index]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(top15))
    bars = ax.barh(y, top15.values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.65, zorder=3)

    # Value labels
    for bar, val in zip(bars, top15.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8.5, color=GRAY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_short, fontsize=10)
    ax.set_xlabel("XGBoost Feature Importance (gain)", fontsize=11, color=GRAY)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")

    legend_handles = [
        mpatches.Patch(color=DOMAIN_COLORS[d], label=d)
        for d in DOMAIN_COLORS
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_title("Top 15 Feature Importances — XGBoost Wildfire Classifier\n"
                 "(Train: 2018–2020 · 43 features · coloured by domain)",
                 fontsize=12, fontweight="bold", color=GRAY, pad=10)

    fig.tight_layout()
    savefig(fig, "fig6_feature_importance.pdf")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(f"\nGenerating figures → {OUT}/\n")

    fig1_pipeline()
    fig2_circuit_topology()
    fig3_model_comparison()
    fig4_capture_rate()
    fig5_task2_scatter()
    fig6_feature_importance()

    print("\nAll figures generated.")
