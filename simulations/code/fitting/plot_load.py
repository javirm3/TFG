#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import ast
import pathlib
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib import cm
from scipy.stats import t

# ==== PATHS DEL PROYECTO ====
base_path = pathlib.Path().resolve().parents[1]
PROJECT_ROOT = pathlib.Path("../").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import paths
from helpers.plots import truncate_colormap, get_onset_offset

# ======= CONFIG PLOTS =======
N_T4_BINS = 3
N_X_BINS = 7
SUBSAMPLE_PER_BIN = 10000

trunc_purples = truncate_colormap("Purples_r", 0, 0.7)
trunc_oranges = truncate_colormap("Oranges", 0.3, 1.0)



def get_plot_path(kind: str, filename: str, model_name: str) -> str:
    base_model_dir = os.path.join(paths.PLOTS, "fitting", model_name)
    kind_dir = os.path.join(base_model_dir, kind)
    os.makedirs(kind_dir, exist_ok=True)
    return os.path.join(kind_dir, filename)


# ========= UTIL: parsear columna 'model' =========
def parse_model_probs_column(df, col="model"):
    df = df.copy()

    def parse_one(x):
        if isinstance(x, np.ndarray):
            arr = x.astype(float)
        elif isinstance(x, (list, tuple)):
            arr = np.array(x, dtype=float)
        elif isinstance(x, str):
            try:
                s = x.replace("  ", " ")
                if "[" in s and "," not in s:
                    s = s.replace("[", "").replace("]", "")
                    parts = [float(t) for t in s.split()]
                    arr = np.array(parts, dtype=float)
                else:
                    arr = np.array(ast.literal_eval(s), dtype=float)
            except Exception:
                arr = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
        else:
            arr = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

        if arr.shape[0] != 4:
            a = np.full(4, np.nan, dtype=float)
            a[: min(4, arr.shape[0])] = arr[:4]
            arr = a
        return arr

    arrs = df[col].apply(parse_one)
    pL = np.array([a[0] for a in arrs], dtype=float)
    pC = np.array([a[1] for a in arrs], dtype=float)
    pR = np.array([a[2] for a in arrs], dtype=float)
    pMiss = np.array([a[3] for a in arrs], dtype=float)

    df["pL"] = pL
    df["pC"] = pC
    df["pR"] = pR
    df["pMiss"] = pMiss

    # === NUEVO: prob de acierto del modelo según el LADO CORRECTO x_c ===
    correct_map = {"L": 0, "C": 1, "R": 2}

    idx_corr = (
        df["x_c"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(correct_map)
    )

    mat = np.stack([pL, pC, pR], axis=1)  # (N,3)
    p_correct = np.full(len(df), np.nan, dtype=float)

    valid = idx_corr.notna()
    valid_rows = np.where(valid.to_numpy())[0]
    valid_idx = idx_corr[valid].to_numpy(dtype=int)

    p_correct[valid_rows] = mat[valid_rows, valid_idx]

    df["p_model_correct"] = p_correct

    return df


# ========= Cálculo nested curves =========
def compute_nested_curves(df_subj,outer_var,x_var,n_t4_bins,n_x_bins,subsample,data_col="correct_bool",model_col="p_model_correct"):
    df_subj = df_subj.copy()

    df_subj["outer_bin"], outer_edges = pd.qcut(df_subj[outer_var], q=n_t4_bins, retbins=True, duplicates="drop",)

    df_subj["x_bin"] = pd.Series(index=df_subj.index, dtype="object")

    outer_centros = (df_subj.groupby("outer_bin", observed=True)[outer_var].median().rename("outer_center").reset_index().sort_values("outer_center"))
    outer_order = list(outer_centros["outer_bin"])
    outer_centers = outer_centros["outer_center"].to_numpy()

    outer_info = {}

    for i, obin in enumerate(outer_order):
        group = df_subj[df_subj["outer_bin"] == obin].copy()
        if len(group) == 0:
            continue

        group["x_bin_cat"], x_edges = pd.qcut(group[x_var],n_x_bins,retbins=True,duplicates="drop",)
        group["x_bin"] = group["x_bin_cat"].astype(str)
        df_subj.loc[group.index, "x_bin"] = group["x_bin"]

        x_centros = (group.groupby("x_bin", observed=True)[x_var].median().rename("x_center").reset_index().sort_values("x_center"))
        x_centers = x_centros["x_center"].to_numpy()
        x_order = list(x_centros["x_bin"])
        
        data_mean = []
        data_sem = []
        model_mean = []
        model_sem = []

        for cat in x_order:
            bin_trials = group[group["x_bin"] == cat].copy()


            if len(bin_trials) > subsample:
                parts = []
                for s in ["L", "C", "R"]:
                    g = bin_trials[bin_trials["x_c"].astype(str).str.upper() == s]
                    k = min(len(g), subsample // 3)
                    if k > 0:
                        parts.append(g.sample(k, random_state=0))
                if len(parts):
                    bin_trials = pd.concat(parts, axis=0)

            if len(bin_trials) == 0:
                data_mean.append(np.nan)
                data_sem.append(0.0)
                model_mean.append(np.nan)
                model_sem.append(0.0)
                continue

            acc = bin_trials[data_col].to_numpy(float)
            p_hat = float(np.nanmean(acc))
            data_mean.append(p_hat)
            if len(acc) > 1:
                n = np.sum(~np.isnan(acc))
                data_sem.append(np.sqrt(p_hat * (1 - p_hat) / max(n, 1))if n > 1 else 0.0)
            else:
                data_sem.append(0.0)


            p_mod = bin_trials[model_col].to_numpy(float)
            m_hat = float(np.nanmean(p_mod))
            model_mean.append(m_hat)
            if len(p_mod) > 1:
                n_m = np.sum(~np.isnan(p_mod))
                model_sem.append(np.nanstd(p_mod, ddof=1) / np.sqrt(max(n_m, 1)) if n_m > 1 else 0.0)
            else:
                model_sem.append(0.0)

        data_mean = np.asarray(data_mean, float)
        data_sem = np.asarray(data_sem, float)
        model_mean = np.asarray(model_mean, float)
        model_sem = np.asarray(model_sem, float)

        cat_to_idx = {cat: j for j, cat in enumerate(x_order)}
        outer_info[obin] = dict(idx=i, outer_center=float(outer_centers[i]), outer_range=(float(outer_edges[i]), float(outer_edges[i + 1])), x_centers=x_centers, x_order=x_order, cat_to_idx=cat_to_idx, data_mean=data_mean, data_sem=data_sem, model_mean=model_mean, model_sem=model_sem, x_edges=x_edges)

    if not outer_info:
        return [], outer_edges

    curves = []
    for obin in outer_order:
        info = outer_info.get(obin)
        if info is None:
            continue
        curves.append(
            dict(outer_bin=obin, outer_center=info["outer_center"], outer_range=info["outer_range"], x_centers=info["x_centers"], data_mean=info["data_mean"], data_sem=info["data_sem"], model_mean=info["model_mean"], model_sem=info["model_sem"])
        )

    curves.sort(key=lambda d: d["outer_center"])
    return curves, outer_edges, {obin: outer_info[obin]["x_edges"] for obin in outer_order}


# ========= Plot =========
def plot_delay_stim_nested(subject, delay_curves, stim_curves, n_delay, n_stim, outer_var_delay, outer_var_stim, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axd, axs = axes

    for i, info in enumerate(delay_curves):
        col = trunc_purples(0.3 + 0.4 * i / max(1, len(delay_curves) - 1))
        xc = info["x_centers"]
        rng = info["outer_range"]
        
        axd.plot(xc, info["model_mean"], color=col, lw=2)
        axd.fill_between(xc, info["model_mean"]-info["model_sem"], info["model_mean"]+info["model_sem"], color=col, alpha=0.15, lw=0)
        axd.errorbar(xc, info["data_mean"], yerr=info["data_sem"], fmt="o", color=col, ms=5, capsize=3, ls="none")

    axd.axhspan(0, 1 / 3, color="gray", alpha=0.2, zorder=0)
    axd.set_xlabel("Delay duration (s)")
    axd.set_ylabel("Frac. correct responses")
    axd.set_title(f"{subject} - Delay (n={n_delay})")
    axd.set_ylim(0.2, 1.05)
    # axd.legend(frameon=False, fontsize=8)

    # ---- Stim ----
    for i, info in enumerate(stim_curves):
        col = trunc_oranges(0.3 + 0.5 * i / max(1, len(stim_curves) - 1))
        xc = info["x_centers"]
        rng = info["outer_range"]

        axs.plot(xc, info["model_mean"], color=col, lw=2)
        axs.fill_between(xc, info["model_mean"]-info["model_sem"], info["model_mean"]+info["model_sem"], color=col, alpha=0.15, lw=0)
        axs.errorbar(xc, info["data_mean"], yerr=info["data_sem"], fmt="o", color=col, ms=5, capsize=3, ls="none")

    axs.axhspan(0, 1 / 3, color="gray", alpha=0.2, zorder=0)
    axs.set_xlabel("Stimulus duration (s)")
    axs.set_title(f"{subject} - Stim (n={n_stim})")
    # axs.legend(frameon=False, fontsize=8)

    sns.despine()
    fig.tight_layout()
    fname = f"fig_delay_stim_nested_{subject}.png"
    out_path = get_plot_path("strat", fname, model_name)
    fig.savefig(out_path, dpi=300)
    # print(f"  Figure saved to {out_path}")
    plt.close(fig)



def plot_scatter_delay_stim(df_subj, model_name, subject=None, stim_edges=None, delay_edges=None, show_bins=True):
    cats_scatter = ["DS", "DM", "DL"]
    df_sc = df_subj[df_subj["ttype_c"].isin(cats_scatter)].copy()
    s_size = 5
    needed_cols = ["stim_duration", "delay_duration", "correct_bool", "p_model_correct"]
    df_sc = df_sc.dropna(subset=needed_cols)
    if stim_edges is None:
        _, stim_edges = pd.qcut(df_sc["stim_duration"], q=N_X_BINS, retbins=True, duplicates="drop")
    if delay_edges is None:
        _, delay_edges = pd.qcut(df_sc["delay_duration"], q=N_T4_BINS, retbins=True, duplicates="drop")

    if df_sc.empty:
        print("  (sin trials válidos para scatter delay/stim)")
        return

    delay_palette = {
        "DS": trunc_purples(0.25),
        "DM": trunc_purples(0.5),
        "DL": trunc_purples(0.75),
    }
    correct_palette = {True: "#2E7D32", False: "#C62828"}  # verde / rojo

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax_a, ax_b, ax_c = axes

    # ---------- Panel a: por tipo de delay ----------
    sns.scatterplot( data=df_sc, x="stim_duration", y="delay_duration", hue="ttype_c", hue_order=["DS", "DM", "DL"], palette=delay_palette, s=s_size, alpha=1, edgecolor="none", ax=ax_a,)
    ax_a.set_title("a) Por tipo de delay")
    ax_a.set_xlabel("Stimulus duration (s)")
    ax_a.set_ylabel("Delay duration (s)")
    ax_a.legend(title="Delay", frameon=False, fontsize=12)

    # ---------- Panel b: Data correcto / incorrecto ----------
    sns.scatterplot( data=df_sc, x="stim_duration", y="delay_duration", hue="correct_bool", palette=correct_palette, s=s_size, alpha=1, edgecolor="none", ax=ax_b, )
    ax_b.set_title("b) Data: correcto vs incorrecto")
    ax_b.set_xlabel("Stimulus duration (s)")
    ax_b.set_ylabel("")
    ax_b.legend(title="Correcto", frameon=False, fontsize=12)

    # ---------- Panel c: Modelo p(correct) ----------
    sns.scatterplot(data=df_sc, x="stim_duration", y="delay_duration", hue="p_model_correct", hue_norm=(0.0, 1.0), palette="RdYlGn", s=s_size, alpha=1, edgecolor="none", ax=ax_c)
    ax_c.set_title("c) Modelo: P(correct)")
    ax_c.set_xlabel("Stimulus duration (s)")
    ax_c.set_ylabel("")
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cbar = ax_c.figure.colorbar(sm, ax=ax_c)
    cbar.set_label("P(correct) modelo")
    ax_c.legend([], frameon=False)

    x_min = df_sc["stim_duration"].min()
    x_max = df_sc["stim_duration"].max()
    y_min = df_sc["delay_duration"].min()
    y_max = df_sc["delay_duration"].max()

    pad_x = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    pad_y = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

    for ax in axes:
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
    if show_bins:
        for x_edge in stim_edges[1:-1]:
            for ax in axes:
                ax.axvline(x_edge, ls="--", lw=0.7, color="black", alpha=0.4)

        for y_edge in delay_edges[1:-1]:
            for ax in axes:
                ax.axhline(y_edge, ls="--", lw=0.7, color="black", alpha=0.4)


    sns.despine()
    fig.tight_layout()
    fname = f"fig_scatter_delay_stim{'_' + subject if subject is not None else ''}.png"
    out_path = get_plot_path("scatter", fname, model_name)
    fig.savefig(out_path, dpi=300)
    # print(f"  Scatter guardado en {out_path}")
    plt.close(fig)

def plot_heatmaps_delay_stim(df_subj, model_name, subject=None, stim_edges=None, delay_edges=None, show_bins=True):
    cats_scatter = ["DS", "DM", "DL"]
    df_sc = df_subj[df_subj["ttype_c"].isin(cats_scatter)].copy()

    needed_cols = ["stim_duration", "delay_duration", "correct_bool", "p_model_correct", "ttype_c"]
    df_sc = df_sc.dropna(subset=needed_cols)

    if df_sc.empty:
        print("  (sin trials válidos para joint/heatmaps delay/stim)")
        return

    if stim_edges is None:
        _, stim_edges = pd.qcut(df_sc["stim_duration"], q=10, retbins=True, duplicates="drop")
    if delay_edges is None:
        _, delay_edges = pd.qcut(df_sc["delay_duration"], q=10, retbins=True, duplicates="drop")

    delay_palette = {"DS": trunc_purples(0.25), "DM": trunc_purples(0.5), "DL": trunc_purples(0.75),}

    # =========================
    # FIGURA A: JOINTPLOT (hue)
    # =========================
    g = sns.jointplot(data=df_sc, x="stim_duration", y="delay_duration", hue="ttype_c", hue_order=["DS", "DM", "DL"], palette=delay_palette, s=10, alpha=0.85, edgecolor="none", height=6, marginal_kws=dict(fill=True, alpha=0.35),)
    g.fig.suptitle("a) Por tipo de delay", y=1.02)
    g.set_axis_labels("Stimulus duration (s)", "Delay duration (s)")

    if show_bins:
        for x_edge in stim_edges[1:-1]:
            g.ax_joint.axvline(x_edge, ls="--", lw=0.7, color="black", alpha=0.35)
        for y_edge in delay_edges[1:-1]:
            g.ax_joint.axhline(y_edge, ls="--", lw=0.7, color="black", alpha=0.35)

    fname_a = f"fig_joint_delay_stim{'_' + subject if subject is not None else ''}.png"
    out_path_a = get_plot_path("scatter", fname_a, model_name)
    plt.legend(title="Delay type", frameon=False, fontsize=12, markerscale=2)
    g.fig.savefig(out_path_a, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    # =========================
    # FIGURA B+C: HEATMAPS
    # =========================
    df_hm = df_sc.copy()
    df_hm["stim_bin"]  = pd.cut(df_hm["stim_duration"], bins=stim_edges, include_lowest=True, labels=False)
    df_hm["delay_bin"] = pd.cut(df_hm["delay_duration"], bins=delay_edges, include_lowest=True, labels=False)

    df_hm["stim_bin"]  = df_hm["stim_bin"] + 1
    df_hm["delay_bin"] = df_hm["delay_bin"] + 1

    heat_data = (df_hm.groupby(["delay_bin", "stim_bin"], observed=True)["correct_bool"].mean().unstack("stim_bin").sort_index(ascending=True))
    heat_model = (df_hm.groupby(["delay_bin", "stim_bin"], observed=True)["p_model_correct"].mean().unstack("stim_bin").sort_index(ascending=True))

    n_x = len(stim_edges) - 1
    n_y = len(delay_edges) - 1
    heat_data  = heat_data.reindex(index=range(1, n_y+1), columns=range(1, n_x+1))
    heat_model = heat_model.reindex(index=range(1, n_y+1), columns=range(1, n_x+1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax_b, ax_c = axes

    sns.heatmap(heat_data, ax=ax_b,vmin=0.0, vmax=1.0, cmap="RdYlGn",cbar=True, cbar_kws={"label": "Data accuracy (mean)"},xticklabels=list(range(1, n_x+1)),yticklabels=list(range(1, n_y+1)),linewidths=0.0)
    ax_b.invert_yaxis()
    ax_b.set_title("b) Data: accuracy")
    ax_b.set_xlabel("Stim bins")
    ax_b.set_ylabel("Delay bins")
    ax_b.xaxis.tick_bottom()  # bins abajo
    ax_b.tick_params(axis="x", rotation=0)
    ax_b.tick_params(axis="y", rotation=0)

    sns.heatmap(heat_model, ax=ax_c,vmin=0.0, vmax=1.0, cmap="RdYlGn",cbar=True, cbar_kws={"label": "Model P(correct) (mean)"},xticklabels=list(range(1, n_x+1)),yticklabels=list(range(1, n_y+1)),linewidths=0.0)
    ax_c.invert_yaxis()
    ax_c.set_title("c) Model: P(correct)")
    ax_c.set_xlabel("Stim bins")
    ax_c.set_ylabel("")
    ax_c.xaxis.tick_bottom()  # bins abajo
    ax_c.tick_params(axis="x", rotation=0)
    ax_c.tick_params(axis="y", rotation=0)

    sns.despine()
    fig.tight_layout()

    fname_hm = f"fig_heatmaps_delay_stim{'_' + subject if subject is not None else ''}.png"
    out_path_hm = get_plot_path("scatter", fname_hm, model_name)
    fig.savefig(out_path_hm, dpi=300)
    plt.close(fig)

def _plot_cat_panel(ax, df, group_col, order, title, xlabel, ylabel=None, palette=None, labels=None):
    df = df[df[group_col].isin(order)].copy()
    if df.empty:
        ax.set_visible(False)
        return


    subj = (df.groupby([group_col, "subject"], observed=True).agg(correct_mean=("correct_bool", "mean"),model_mean=("p_model_correct", "mean"),).reset_index())
    g = subj.groupby(group_col, observed=True)

    mean_data  = g["correct_mean"].mean()
    std_data   = g["correct_mean"].std(ddof=1)
    n_subj_d   = g["correct_mean"].count().clip(lower=1)
    sem_data   = std_data / np.sqrt(n_subj_d)

    mean_model = g["model_mean"].mean()
    std_model  = g["model_mean"].std(ddof=1)
    n_subj_m   = g["model_mean"].count().clip(lower=1)
    sem_model  = std_model / np.sqrt(n_subj_m)

    tcrit_d = t.ppf(0.975, n_subj_d - 1)
    tcrit_m = t.ppf(0.975, n_subj_m - 1)
    ci_data  = sem_data  * tcrit_d
    ci_model = sem_model * tcrit_m

    cats = [c for c in order if c in mean_data.index]
    x = np.arange(len(cats))

    md = mean_data.loc[cats].to_numpy()
    sd = ci_data.loc[cats].to_numpy()
    mm = mean_model.loc[cats].to_numpy()
    sm = ci_model.loc[cats].to_numpy()

    ax.plot(x, mm, "-", color="black", lw=2, label="Model")
    ax.fill_between(x, mm-sm, mm+sm, color="black", alpha=0.15)

    colors = palette if palette else ["black"] * len(cats)
    for i, (xpos, yval, err) in enumerate(zip(x, md, sd)):
        ax.errorbar(xpos, yval, yerr=err, fmt="o",
                    color=colors[i], ms=7, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels else cats)

    ax.set_ylim(0.2, 1.05)
    ax.axhspan(0, 1/3, color="gray", alpha=0.15)
    ax.set_xlim(left=-0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

def plot_categorical_performance_all(df, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    ax1, ax2, ax3 = axes

    # =====================================================
    # a) Stimulus categories (TODOS los estímulos)
    # =====================================================
    palette_a = ['#230027','#C88FEC','#9C69A3','#C698CB','#EFD9F5']
    labels_a  = ["Visual","Easy","Medium","Hard"]
    order_a   = ["VG","SL","SM","SS","SIL"]

    df_a = df.copy()
    _plot_cat_panel(ax1, df_a, "stimd_c", order_a, title="a) Trial difficulty", xlabel="Trial difficulty",           ylabel="Accuracy", palette=palette_a, labels=labels_a)

    # =====================================================
    # b) Stim duration (solo DS, SS/SM/SL)
    # =====================================================
    palette_b = ["#FFB74D","#FB8C00","#EF6C00"]
    labels_b  = ['Short','Med','Long']
    order_b   = ["SS","SM","SL"]

    df_b = df[(df["ttype_c"] == "DS") & (df["stimd_c"].isin(order_b))]
    _plot_cat_panel(ax2, df_b, "stimd_c", order_b, title="b) Stim duration", xlabel="Stimulus type", palette=palette_b, labels=labels_b)

    # =====================================================
    # c) Delay duration (solo SS)
    # =====================================================
    palette_c = ['#5E2A7E','#9C69A3','#C698CB']
    labels_c  = ['Short','Med','Long']
    order_c   = ["DS","DM","DL"]

    df_c = df[df["stimd_c"] == "SS"]
    _plot_cat_panel(ax3, df_c, "ttype_c", order_c, title="c) Delay duration", xlabel="Delay type", palette=palette_c,labels=labels_c)

    sns.despine()
    fig.tight_layout()
    fname = f"fig_categorical_perf_all.png"
    out_path = get_plot_path("general", fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_categorical_strat_by_side(df, subject, model_name, df_silent = None, cond_col="stimd_c",
                                   cond_order=['VG', 'SL', 'SM', 'SS', 'SIL'], cond_labels=['Visual', 'Easy', 'Medium', 'Hard', 'Silent']):
    df = df.copy()
    df["x_c"] = (df["x_c"].astype("string").str.strip().str.upper())

    if cond_order is None:
        cond_order = list(df[cond_col].dropna().unique())
        cond_order = sorted(cond_order)

    if cond_labels is None:
        cond_labels = cond_order

    g = (df.groupby([cond_col, "x_c"], observed=True).agg(data_mean=("correct_bool", "mean"), model_mean=("p_model_correct", "mean"), n=("correct_bool", "size")).reset_index())

    g["data_sem"] = np.sqrt(g["data_mean"] * (1.0 - g["data_mean"]) / g["n"].clip(lower=1))

    if df_silent is not None:
        df_s = df_silent.copy()
        p_silent = {"L": df_s["pL_mean"], "C": df_s["pC_mean"], "R": df_s["pR_mean"]}

    cond_to_x = {c: i for i, c in enumerate(cond_order)}
    g["x_pos"] = g[cond_col].map(cond_to_x)

    side_palette = {'L': '#e41a1c', 'C': '#4daf4a', 'R': '#377eb8'}

    fig, ax = plt.subplots(figsize=(5,5))

    for side in ["L", "C", "R"]:
        sub = g[g["x_c"] == side].dropna(subset=["x_pos"])
        if sub.empty:
            continue

        sub = sub.sort_values("x_pos")

        ax.plot( sub["x_pos"], sub["model_mean"], "-", lw=2, color=side_palette.get(side, "gray"), label=f"Model {side}", zorder=2)

        ax.errorbar( sub["x_pos"], sub["data_mean"], yerr=sub["data_sem"], fmt="o", ms=5, capsize=3, color=side_palette.get(side, "gray"), linestyle="none", label=f"Data {side}", zorder=3)

        if df_silent is not None:
            ax.plot(len(cond_order)-1, p_silent[side],marker="D", ms=7,color=side_palette[side],linestyle="none",zorder=4)

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)

    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_labels)

    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel("Frac. correct responses")
    ax.set_xlabel("Trial difficulty")
    ax.set_title(f"{subject}")

    # ax.legend(frameon=False, fontsize=8, ncol=2)
    sns.despine()
    fig.tight_layout()

    fname = f"fig_categorical_strat_by_side_{subject}.png"
    out_path = get_plot_path("strat_by_side", fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_delay_binned_1d(df, model_name, subject=None, n_bins=N_X_BINS):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_delay = df[df['onset']==0.0].copy()
    df_stim = df[df['ttype_c']!='VG'].copy()
    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_duration", "correct_bool", "p_model_correct", "subject", 'stim_duration']
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim = df_stim.dropna(subset=needed_cols)

    if df_delay.empty:
        print(f"  (sin datos válidos para delay 1D en {subject})")
        return
    elif df_stim.empty:
        print(f"  (sin datos válidos para stim 1D en {subject})")
        return
    
    df_delay["delay_bin"], edges = pd.qcut(df_delay["delay_duration"], q=n_bins, retbins=True, duplicates="drop")
    df_stim["stim_bin"], edges_stim = pd.qcut(df_stim["stim_duration"], q=n_bins, retbins=True, duplicates="drop")
    centers_delay = (df_delay.groupby("delay_bin", observed=True)["delay_duration"].median().rename("center").reset_index().sort_values("center"))
    centers_stim = (df_stim.groupby("stim_bin", observed=True)["stim_duration"].median().rename("center").reset_index().sort_values("center"))
    order_bins_delay = list(centers_delay["delay_bin"])
    order_bins_stim = list(centers_stim["stim_bin"])
 
    subj_delay = (df_delay.groupby(["delay_bin", "subject"], observed=True).agg(data_acc=("correct_bool", "mean"),model_acc=("p_model_correct", "mean"),).reset_index().merge(centers_delay, on="delay_bin", how="left"))
    plot_delay = subj_delay.melt(id_vars=["delay_bin", "subject", "center"],value_vars=["data_acc", "model_acc"],var_name="kind",value_name="acc",)
    plot_delay["kind"] = plot_delay["kind"].map({"data_acc": "Data","model_acc": "Model"})

    subj_stim = (df_stim.groupby(["stim_bin", "subject"], observed=True).agg(data_acc=("correct_bool", "mean"),model_acc=("p_model_correct", "mean"),).reset_index().merge(centers_stim, on="stim_bin", how="left"))
    plot_stim = subj_stim.melt(id_vars=["stim_bin", "subject", "center"],value_vars=["data_acc", "model_acc"],var_name="kind",value_name="acc",)
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data","model_acc": "Model"})


    fig, ax = plt.subplots(figsize=(5, 5))

    sns.lineplot(data=plot_delay[plot_delay["kind"] == "Model"], x="center", y="acc",color="gray", linestyle="-",errorbar=("ci", 95),err_style="band",ax=ax)
    sns.lineplot(x="center", y="acc", hue="center",data=plot_delay[plot_delay["kind"] == "Data"],errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0, ax=ax, zorder=10, palette=trunc_purples, legend=False)

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)

    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Delay duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")

    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Delay (1D, {len(order_bins_delay)} bins)")

    sns.despine()
    fig.tight_layout()

    fname = f"fig_delay_1d_{title_subj}.png"
    out_path = get_plot_path("general", fname, model_name)
    fig.savefig(out_path, dpi=300)

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(data=plot_stim[plot_stim["kind"] == "Model"], x="center", y="acc",color="gray", linestyle="-",errorbar=("ci", 95),err_style="band",ax=ax)
    sns.lineplot(x="center", y="acc", hue="center",data=plot_stim[plot_stim["kind"] == "Data"],errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0, ax=ax, zorder=10, palette=trunc_oranges, legend=False)
    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Stimulus duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Stimulus (1D, {len(order_bins_stim)} bins)")
    sns.despine()
    fig.tight_layout()
    fname = f"fig_stim_1d_{title_subj}.png"
    out_path = get_plot_path("general", fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


DT_TRACES = 0.1 / 40.0  # el mismo dt que usas en la simulación
SIDE_TO_TRACE_COL = {"L": "trace_L", "C": "trace_C", "R": "trace_R"}


def _stack_traces(traces_list):
    """
    Dada una lista de arrays 1D de distinta longitud, devuelve
    un array (n_trials, max_len) rellenando con NaNs a la derecha.
    """
    traces_list = [np.asarray(t, dtype=float) for t in traces_list if t is not None]
    if len(traces_list) == 0:
        return None
    max_len = max(len(t) for t in traces_list)
    arr = np.full((len(traces_list), max_len), np.nan, dtype=float)
    for i, tr in enumerate(traces_list):
        L = len(tr)
        arr[i, :L] = tr
    return arr


def _time_axis_from_traces(traces_arr):
    """
    Genera eje temporal para un array de trazas (n_trials, n_time).
    """
    if traces_arr is None:
        return None
    n_t = traces_arr.shape[1]
    return np.arange(n_t, dtype=float) * DT_TRACES


ALL_SIDES = ["L", "C", "R"]

def _get_model_choice(row, thr=0.5):
    """
    Devuelve 'L', 'C' o 'R' según la traza con valor final máximo,
    siempre que ese valor final sea > thr.
    Si no hay decisión clara, devuelve None.
    """
    try:
        tr_L = row.get(SIDE_TO_TRACE_COL["L"], None)
        tr_C = row.get(SIDE_TO_TRACE_COL["C"], None)
        tr_R = row.get(SIDE_TO_TRACE_COL["R"], None)
    except KeyError:
        return None

    if tr_L is None or tr_C is None or tr_R is None:
        return None

    tr_L = np.asarray(tr_L, dtype=float)
    tr_C = np.asarray(tr_C, dtype=float)
    tr_R = np.asarray(tr_R, dtype=float)

    # valor al final del trial
    end_vals = np.array([tr_L[-1], tr_C[-1], tr_R[-1]], dtype=float)
    if not np.all(np.isfinite(end_vals)):
        return None

    idx_max = int(np.argmax(end_vals))
    chosen = ALL_SIDES[idx_max]
    chosen_end_val = end_vals[idx_max]

    if chosen_end_val <= thr:
        # el modelo no ha llegado a una decisión clara
        return None

    return chosen

# ============================================================
# 1) Correct trials: winning population vs delay_duration binned
# ============================================================
def plot_traces_correct_by_delay(df, subject, model_name, n_bins=3, kind_dir="traces"):
    """
    Para un sujeto:
      - selecciona trials correctos,
      - define 'winning population' como la población elegida (r_c),
      - agrupa por delay_duration en n_bins (qcut),
      - plotea la traza media de la población ganadora para cada bin de delay.
    """
    df_s = df[df["correct_bool"] == True].copy()

    # Nos aseguramos de tener lados válidos
    for col in ["x_c", "r_c"]:
        if col in df_s.columns:
            df_s[col] = df_s[col].astype("string").str.strip().str.upper()

    df_s = df_s[df_s["r_c"].isin(["L", "C", "R"])]

    if df_s.empty:
        print(f"[{subject}] Sin trials correctos para trazas por delay.")
        return

    # Binning por delay_duration
    df_s = df_s[df_s["delay_duration"].notna()].copy()
    df_s["delay_bin"], bins = pd.qcut(df_s["delay_duration"], q=n_bins, retbins=True, duplicates="drop")
    # Centro de cada bin
    bin_centers = (df_s.groupby("delay_bin", observed=True)["delay_duration"].median().sort_values())

    fig, ax = plt.subplots(figsize=(6, 4))

    palette = sns.color_palette("viridis", len(bin_centers))
    for idx, (bin_cat, center_val) in enumerate(bin_centers.items()):
        sub = df_s[df_s["delay_bin"] == bin_cat].copy()
        if sub.empty:
            continue

        # ganar = traza de la población elegida (r_c)
        traces_winning = []
        for _, row in sub.iterrows():
            side = row["r_c"]
            col_trace = SIDE_TO_TRACE_COL.get(side)
            tr = row.get(col_trace, None)
            if tr is not None:
                traces_winning.append(tr)

        arr = _stack_traces(traces_winning)
        if arr is None:
            continue

        t = _time_axis_from_traces(arr)
        mean_tr = np.nanmean(arr, axis=0)
        sem_tr = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(arr), axis=0).clip(min=1))

        col = palette[idx]
        label = f"Delay bin {idx+1}\n(med ~ {center_val:.2f}s)"
        ax.plot(t, mean_tr, color=col, lw=2, label=label)
        ax.fill_between(t, mean_tr - sem_tr, mean_tr + sem_tr, color=col, alpha=0.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population rate (winning choice)")
    ax.set_title(f"{subject} - Correct trials by delay")
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()

    fname = f"traces_correct_by_delay_{subject}.png"
    out_path = get_plot_path(kind_dir, fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ==================================================================
# 2) Error trials: winning vs good-choice vs other (3 poblaciones)
# ==================================================================
def plot_traces_errors_winning_good_bad(df, subject, model_name, kind_dir="traces", thr_model_choice=0.5):
    """
    Para un sujeto:
      - selecciona trials incorrectos del ANIMAL (r_c != x_c),
      - el modelo debe haber tomado una decisión (traza final > thr_model_choice),
      - winning = población elegida por el MODELO,
      - good-choice = población del lado correcto (x_c),
      - other = la tercera población,
      - plotea las trazas medias (±SEM) de las tres.
    """
    df_s = df.copy()

    # Normalizamos strings
    for col in ["x_c", "r_c"]:
        if col in df_s.columns:
            df_s[col] = df_s[col].astype("string").str.strip().str.upper()

    df_s = df_s[df_s["r_c"].isin(ALL_SIDES)]
    df_s = df_s[df_s["x_c"].isin(ALL_SIDES)]

    # Correcto/incorrecto según comportamiento
    df_s["behav_correct"] = (df_s["x_c"] == df_s["r_c"])

    # Solo errores del animal
    df_s = df_s[df_s["behav_correct"] == False].copy()

    if df_s.empty:
        print(f"[{subject}] Sin trials incorrectos (animal) para trazas.")
        return

    # Elección del modelo
    df_s["model_choice"] = df_s.apply(lambda r: _get_model_choice(r, thr_model_choice), axis=1)

    # Nos quedamos solo con trials donde el modelo realmente elige algo
    df_s = df_s[df_s["model_choice"].notna()].copy()
    df_s['model_correct'] = (df_s["x_c"] == df_s["model_choice"])
    # df_s = df_s[df_s["model_correct"] == False].copy()

    if df_s.empty:
        print(f"[{subject}] Sin trials donde el modelo tome decisión en errores.")
        return

    traces_winning = []
    traces_good    = []
    traces_other   = []

    for _, row in df_s.iterrows():
        chosen  = row["model_choice"]  # elección del modelo
        correct = row["x_c"]           # lado correcto real

        others = [s for s in ALL_SIDES if s not in (chosen, correct)]
        if len(others) != 1:
            continue
        other = others[0]

        tr_L = np.asarray(row[SIDE_TO_TRACE_COL["L"]], dtype=float)
        tr_C = np.asarray(row[SIDE_TO_TRACE_COL["C"]], dtype=float)
        tr_R = np.asarray(row[SIDE_TO_TRACE_COL["R"]], dtype=float)

        tr_chosen  = {"L": tr_L, "C": tr_C, "R": tr_R}[chosen]
        tr_correct = {"L": tr_L, "C": tr_C, "R": tr_R}[correct]
        tr_other   = {"L": tr_L, "C": tr_C, "R": tr_R}[other]

        traces_winning.append(tr_chosen)
        traces_good.append(tr_correct)
        traces_other.append(tr_other)

    arr_win   = _stack_traces(traces_winning)
    arr_good  = _stack_traces(traces_good)
    arr_other = _stack_traces(traces_other)

    if arr_win is None or arr_good is None or arr_other is None:
        print(f"[{subject}] No se pudieron acumular trazas de error.")
        return

    t = _time_axis_from_traces(arr_win)

    def _mean_sem(arr):
        mean = np.nanmean(arr, axis=0)
        n_eff = np.sum(np.isfinite(arr), axis=0).clip(min=1)
        sem  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_eff)
        return mean, sem

    mean_win, sem_win     = _mean_sem(arr_win)
    mean_good, sem_good   = _mean_sem(arr_good)
    mean_other, sem_other = _mean_sem(arr_other)

    fig, ax = plt.subplots(figsize=(6, 4))

    col_win   = "#d73027"  # rojo
    col_good  = "#1a9850"  # verde
    col_other = "#666666"  # gris

    ax.plot(t, mean_win,   color=col_win,   lw=2, label="Winning (model choice)")
    ax.fill_between(t, mean_win - sem_win, mean_win + sem_win, color=col_win, alpha=0.2)

    ax.plot(t, mean_good,  color=col_good,  lw=2, label="Good-choice (correct side)")
    ax.fill_between(t, mean_good - sem_good, mean_good + sem_good, color=col_good, alpha=0.2)

    ax.plot(t, mean_other, color=col_other, lw=2, label="Other")
    ax.fill_between(t, mean_other - sem_other, mean_other + sem_other, color=col_other, alpha=0.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population rate")
    ax.set_title(f"Error trials (animal)")
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()

    fname = f"traces_errors_winning_good_bad_{subject}_animal.png"
    out_path = get_plot_path(kind_dir, fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ============================================================
# 3) Winning population traces: correct vs error
# ============================================================
def plot_traces_winning_correct_vs_error(df, subject, model_name, kind_dir="traces"):
    """
    Para un sujeto:
      - winning = población elegida (r_c),
      - compara la traza media de winning en trials correctos vs incorrectos.
    """
    # df_s = df[df["subject"] == subject].copy()
    df_s = df

    for col in ["x_c", "r_c"]:
        if col in df_s.columns:
            df_s[col] = df_s[col].astype("string").str.strip().str.upper()

    df_s = df_s[df_s["r_c"].isin(["L", "C", "R"])]

    if df_s.empty:
        print(f"[{subject}] Sin trials válidos para winning correct vs error.")
        return

    df_corr = df_s[df_s["correct_bool"] == True].copy()
    df_err  = df_s[df_s["correct_bool"] == False].copy()

    traces_corr = []
    traces_err  = []

    for _, row in df_corr.iterrows():
        side = row["r_c"]
        tr = row.get(SIDE_TO_TRACE_COL[side], None)
        if tr is not None:
            traces_corr.append(tr)

    for _, row in df_err.iterrows():
        side = row["r_c"]
        tr = row.get(SIDE_TO_TRACE_COL[side], None)
        if tr is not None:
            traces_err.append(tr)

    arr_corr = _stack_traces(traces_corr)
    arr_err  = _stack_traces(traces_err)

    if arr_corr is None or arr_err is None:
        print(f"[{subject}] No se pudieron acumular trazas winning correct/error.")
        return

    n_t = max(arr_corr.shape[1], arr_err.shape[1])
    # Ajustamos longitud rellenando con NaN si hace falta
    def _pad_to(arr, T):
        if arr.shape[1] == T:
            return arr
        out = np.full((arr.shape[0], T), np.nan, dtype=float)
        out[:, : arr.shape[1]] = arr
        return out

    arr_corr = _pad_to(arr_corr, n_t)
    arr_err  = _pad_to(arr_err, n_t)
    t = np.arange(n_t, dtype=float) * DT_TRACES

    def _mean_sem(arr):
        mean = np.nanmean(arr, axis=0)
        n_eff = np.sum(np.isfinite(arr), axis=0).clip(min=1)
        sem  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_eff)
        return mean, sem

    mean_corr, sem_corr = _mean_sem(arr_corr)
    mean_err,  sem_err  = _mean_sem(arr_err)

    fig, ax = plt.subplots(figsize=(6, 4))

    col_corr = "#1a9850"  # verde
    col_err  = "#d73027"  # rojo

    ax.plot(t, mean_corr, color=col_corr, lw=2, label="Correct trials (winning)")
    ax.fill_between(t, mean_corr - sem_corr, mean_corr + sem_corr, color=col_corr, alpha=0.2)

    ax.plot(t, mean_err,  color=col_err,  lw=2, label="Error trials (winning)")
    ax.fill_between(t, mean_err - sem_err, mean_err + sem_err, color=col_err, alpha=0.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population rate (winning choice)")
    ax.set_title(f"{subject} – Winning population: correct vs error")
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()

    fname = f"traces_winning_correct_vs_error_{subject}.png"
    out_path = get_plot_path(kind_dir, fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ========= MAIN =========
if __name__ == "__main__":
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_context("talk", font_scale=1)

    MODEL_NAME = "spatial_reduced_cert"  # Cambiar aquí el modelo a analizar
    
    probs_path = os.path.join(paths.PARAMS_DIR, f"df_{MODEL_NAME}.parquet")
    df = pd.read_parquet(probs_path, engine="pyarrow")
    df_silent_path = os.path.join(paths.PARAMS_DIR, f"df_silent_{MODEL_NAME}.csv")
    df_silent = pd.read_csv(df_silent_path, sep=";")
    print(f"Leyendo probs de: {probs_path}")
    
    # probs_path = os.path.join(paths.PARAMS_DIR, f"df_trial_probs_{MODEL_NAME}.csv")
    # df = pd.read_csv(probs_path, sep=";")
    # if "onset" not in df.columns or "offset" not in df.columns:
    #     print("Añadiendo columnas onset/offset...")
    #     df[["onset", "offset"]] = df.apply(lambda r: pd.Series(get_onset_offset(r["stimd_c"], r["ttype_c"], r["timepoint_1"], r["timepoint_2"], r["timepoint_3"],r["timepoint_4"],)),axis=1,)

    # # duraciones
    # df["stim_duration"] = df["offset"] - df["onset"]
    # df["delay_duration"] = df["timepoint_4"] - df["offset"]

 
    # parsear columna 'model' -> pL/pC/pR/pMiss/p_model_correct
    
    df = parse_model_probs_column(df, col="model")
    subjects = sorted(df["subject"].unique())
    print("Subjects:", subjects)

    for subject in tqdm(subjects, desc="Plotting"):
        df_subj = df[df["subject"] == subject].copy()

        df_delay = df_subj[
            (df_subj["onset"] == 0)
        ].copy()
        n_delay = len(df_delay)
        if n_delay == 0:
            print("  (sin trials delay)")
        else:
            delay_curves, delay_outer_edges, delay_x_edges_dict = compute_nested_curves(df_delay, outer_var="offset", x_var="delay_duration",  n_t4_bins=N_T4_BINS, n_x_bins=N_X_BINS, subsample=SUBSAMPLE_PER_BIN,data_col="correct_bool", model_col="p_model_correct",)

        df_stim = df_subj[(df_subj["onset"] == 0)].copy()
        n_stim = len(df_stim)
        if n_stim == 0:
            print("  (sin trials stim)")
            continue

        stim_curves, stim_outer_edges, stim_x_edges_dict = compute_nested_curves(df_stim, outer_var="offset", x_var="stim_duration",n_t4_bins=N_T4_BINS, n_x_bins=N_X_BINS, subsample=SUBSAMPLE_PER_BIN, data_col="correct_bool", model_col="p_model_correct",)

        if n_delay == 0 or len(delay_curves) == 0 or len(stim_curves) == 0:
            print("  (curvas insuficientes para plot)")
            continue
        
        if len(delay_x_edges_dict):
            delay_edges = np.unique(np.concatenate(list(delay_x_edges_dict.values())))
        else:
            delay_edges = None

        if len(stim_x_edges_dict):
            stim_edges = np.unique(np.concatenate(list(stim_x_edges_dict.values())))
        else:
            stim_edges = None

        plot_delay_stim_nested(subject, delay_curves, stim_curves, n_delay, n_stim, outer_var_delay="offset", outer_var_stim="offset", model_name=MODEL_NAME)
        plot_scatter_delay_stim(df_subj, MODEL_NAME, subject=subject, stim_edges=stim_outer_edges, delay_edges=delay_outer_edges, show_bins=True)
        df_silent_subj = df_silent[df_silent["subject"] == subject].copy()
        plot_categorical_strat_by_side(df_subj, subject, MODEL_NAME, df_silent=df_silent_subj) # Para plot sin silent no pasar df_silent
    plot_scatter_delay_stim(df, MODEL_NAME, show_bins=False)
    plot_heatmaps_delay_stim(df, MODEL_NAME)
    plot_categorical_performance_all(df, MODEL_NAME)
    plot_delay_binned_1d(df=df, model_name=MODEL_NAME, n_bins=N_X_BINS)

    traces_path = os.path.join(paths.PARAMS_DIR, f"df_traces_{MODEL_NAME}.parquet")
    df_traces = pd.read_parquet(traces_path)
    subjects = sorted(df_traces["subject"].unique())
    for subject in tqdm(subjects, desc="Plotting traces"):
        df_subj = df_traces[df_traces["subject"] == subject].copy()
        plot_traces_correct_by_delay(df_subj, subject=subject, model_name=MODEL_NAME, n_bins=7)
        plot_traces_winning_correct_vs_error(df_subj, subject=subject, model_name=MODEL_NAME)
        plot_traces_errors_winning_good_bad(df_subj, subject=subject, model_name=MODEL_NAME)
    # plot_traces_correct_by_delay(df_traces, subject="All_Subjects", model_name=MODEL_NAME, n_bins=7)
    # plot_traces_winning_correct_vs_error(df_traces, subject="All_Subjects", model_name=MODEL_NAME)
    # plot_traces_errors_winning_good_bad(df_traces, subject="All_Subjects", model_name=MODEL_NAME)
    
    print("\nListo.")
