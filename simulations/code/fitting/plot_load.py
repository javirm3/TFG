# rsync -avP mini:code/fitting/df_traces_correct_error_spatial_reduced3.parquet df_traces_correct_error_spatial_reduced3.parquet

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
    fname = f"fig_delay_stim_nested_{subject}.pdf"
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
    fname = f"fig_scatter_delay_stim{'_' + subject if subject is not None else ''}.pdf"
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
        _, stim_edges = pd.qcut(df_sc["stim_duration"], q=7, retbins=True, duplicates="drop")
    if delay_edges is None:
        _, delay_edges = pd.qcut(df_sc["delay_duration"], q=7, retbins=True, duplicates="drop")

    delay_palette = {"DS": trunc_purples(0.25), "DM": trunc_purples(0.5), "DL": trunc_purples(0.75),}
    stim_palette = {"SS": trunc_oranges(0.25), "SM": trunc_oranges(0.5), "SL": trunc_oranges(0.75)}
    # =========================
    # FIGURA A: JOINTPLOT (hue)
    # =========================
    delay_edges_by_cat = {}
    for delay in ["DS", "DM", "DL"]:
        df_s = df_sc[df_sc["ttype_c"] == delay]
        df_s = df_s[df_s["stimd_c"] == "SS"] 

        # Delay bins
        _, delay_edges = pd.qcut(
            df_s["delay_duration"],
            q=7,
            retbins=True,
            duplicates="drop"
        )
        delay_edges_by_cat[delay] = delay_edges

    df_plot = df_sc[df_sc["stimd_c"] == "SS"] 
    g = sns.jointplot(data=df_plot, x="stim_duration", y="delay_duration", hue="ttype_c", hue_order=["DS", "DM", "DL"], palette=delay_palette, s=5, alpha=0.85, edgecolor="none", height=6, marginal_kws=dict(fill=True, alpha=0.35),)
    g.fig.suptitle("a) Por tipo de delay", y=1.02)
    g.set_axis_labels("Stimulus duration (s)", "Delay duration (s)")

    if show_bins:
        for delay in ["DS", "DM", "DL"]:
            color = delay_palette[delay]
            for x_edge in delay_edges_by_cat[delay][1:-1]:
                g.ax_joint.axhline( x_edge, ls="--", lw=1.2, color=color, alpha=1,zorder=1)

    fname_a = f"fig_joint_stim{'_' + subject if subject is not None else ''}.pdf"
    out_path_a = get_plot_path("scatter", fname_a, model_name)
    plt.legend(title="Stim type", frameon=False, fontsize=12, markerscale=2)
    g.fig.savefig(out_path_a, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    # ===========================
    # With bins cataegorical stim
    # ===========================
    stim_edges_by_cat = {}

    for stim in ["SS", "SM", "SL"]:
        df_s = df_sc[df_sc["ttype_c"] == "DS"]
        df_s = df_s[df_s["stimd_c"] == stim]

        # Stimulus bins
        _, stim_edges = pd.qcut(
            df_s["stim_duration"],
            q=7,
            retbins=True,
            duplicates="drop"
        )
        stim_edges_by_cat[stim] = stim_edges
    df_plot = df_sc[df_sc["ttype_c"] == "DS"]
    g = sns.jointplot(data=df_plot, x="stim_duration", y="delay_duration", hue="stimd_c", hue_order=["SS", "SM", "SL"], palette=stim_palette, s=5, alpha=0.85, edgecolor="none", height=6, marginal_kws=dict(fill=True, alpha=0.35),)
    g.fig.suptitle("a') Por tipo de estímulo", y=1.02)
    g.set_axis_labels("Stimulus duration (s)", "Delay duration (s)")

    if show_bins:
        for stim in ["SS", "SM", "SL"]:
            color = stim_palette[stim]

            for x_edge in stim_edges_by_cat[stim][1:-1]:
                g.ax_joint.axvline( x_edge, ls="--", lw=1.2, color=color, alpha=1,zorder=1)

    fname_a = f"fig_joint_delay_stim{'_' + subject if subject is not None else ''}.pdf"
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

    fname_hm = f"fig_heatmaps_delay_stim{'_' + subject if subject is not None else ''}.pdf"
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
    labels_a  = ["Visual","Easy","Medium","Hard", "Silent"]
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
    fname = f"fig_categorical_perf_all.pdf"
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

    fname = f"fig_categorical_strat_by_side_{subject}.pdf"
    out_path = get_plot_path("strat_by_side", fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_delay_binned_1d(df, model_name, subject=None, n_bins=N_X_BINS):
    # n_bins=3
    # df_delay = df[df['onset']==0.0].copy()
    df_delay = df[df['stimd_c'] == 'SS']
    # df_stim = df[df['ttype_c']!='VG'].copy()
    df_stim = df[df['ttype_c']=='DS'].copy()
    
    
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
    
    # df_delay["delay_bin"], edges = pd.qcut(df_delay["delay_duration"], q=n_bins, retbins=True, duplicates="drop")
    # df_stim["stim_bin"], edges_stim = pd.qcut(df_stim["stim_duration"], q=n_bins, retbins=True, duplicates="drop")
    # centers_delay = (df_delay.groupby("delay_bin", observed=True)["delay_duration"].median().rename("center").reset_index().sort_values("center"))
    # centers_stim = (df_stim.groupby("stim_bin", observed=True)["stim_duration"].median().rename("center").reset_index().sort_values("center"))
    # order_bins_delay = list(centers_delay["delay_bin"])
    # order_bins_stim = list(centers_stim["stim_bin"])
    
    df_delay["delay_bin"] = (
    df_delay.groupby("ttype_c", observed=True)["delay_duration"]
    .transform(lambda s: pd.qcut(s, q=n_bins, duplicates="drop"))
    )

    # centers por ttype_c y bin
    centers_delay = (
        df_delay.groupby(["ttype_c", "delay_bin"], observed=True)["delay_duration"]
        .median()
        .rename("center")
        .reset_index()
    )

    # (opcional) order de bins dentro de cada ttype_c según center
    centers_delay["bin_order"] = centers_delay.groupby("ttype_c")["center"].rank(method="dense")
    order_bins_delay = list(centers_delay["delay_bin"])
    # agregación por bin+subject+ttype_c
    subj_delay = (
        df_delay.groupby(["ttype_c", "delay_bin", "subject"], observed=True)
        .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
        )
        .reset_index()
        .merge(centers_delay, on=["ttype_c", "delay_bin"], how="left")
    )
    df_stim["stim_bin"] = (
    df_stim.groupby("stimd_c", observed=True)["stim_duration"]
    .transform(lambda s: pd.qcut(s, q=n_bins, duplicates="drop"))   
    )

    centers_stim = (
        df_stim.groupby(["stimd_c", "stim_bin"], observed=True)["stim_duration"]
        .median()
        .rename("center")
        .reset_index()
    )
    order_bins_stim = list(centers_stim["stim_bin"])
    subj_stim = (
        df_stim.groupby(["stimd_c", "stim_bin", "subject"], observed=True)
        .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
        )
        .reset_index()
        .merge(centers_stim, on=["stimd_c", "stim_bin"], how="left")
    )

    plot_stim = subj_stim.melt(
        id_vars=["stimd_c", "stim_bin", "subject", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data", "model_acc": "Model"})


    # subj_delay = (df_delay.groupby(["delay_bin", "subject", "ttype_c"], observed=True).agg(data_acc=("correct_bool", "mean"),model_acc=("p_model_correct", "mean"),).reset_index().merge(centers_delay, on="delay_bin", how="left"))
    plot_delay = subj_delay.melt(id_vars=["delay_bin", "subject", "ttype_c", "center"],value_vars=["data_acc", "model_acc"],var_name="kind",value_name="acc",)
    plot_delay["kind"] = plot_delay["kind"].map({"data_acc": "Data","model_acc": "Model"})

    # subj_stim = (df_stim.groupby(["stim_bin", "subject", "stimd_c"], observed=True).agg(data_acc=("correct_bool", "mean"),model_acc=("p_model_correct", "mean"),).reset_index().merge(centers_stim, on="stim_bin", how="left"))
    plot_stim = subj_stim.melt(id_vars=["stim_bin", "subject", "center", "stimd_c"],value_vars=["data_acc", "model_acc"],var_name="kind",value_name="acc",)
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data","model_acc": "Model"})


    fig, ax = plt.subplots(figsize=(6, 6))

    sns.lineplot(data=plot_delay[plot_delay["kind"] == "Model"], x="center", y="acc",color="gray", hue='ttype_c', linestyle="-",errorbar=("ci", 95),err_style="band",ax=ax)
    sns.lineplot(x="center", y="acc", hue="ttype_c",data=plot_delay[plot_delay["kind"] == "Data"], errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0, ax=ax, zorder=10, legend=False)

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)

    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Delay duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")

    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Delay (1D, {len(order_bins_delay)} bins)")

    sns.despine()
    fig.tight_layout()

    fname = f"fig_delay_1d_{title_subj}.pdf"
    out_path = get_plot_path("binning", fname, model_name)
    fig.savefig(out_path, dpi=300)

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(data=plot_stim[plot_stim["kind"] == "Model"], x="center", y="acc",color="gray", hue = "stimd_c", linestyle="-",errorbar=("ci", 95),err_style="band",ax=ax)
    sns.lineplot(x="center", y="acc", hue="stimd_c",data=plot_stim[plot_stim["kind"] == "Data"],errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0, ax=ax, zorder=10, legend=False)
    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Stimulus duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Stimulus (1D, {len(order_bins_stim)} bins)")
    sns.despine()
    fig.tight_layout()
    fname = f"fig_stim_1d_{title_subj}.pdf"
    out_path = get_plot_path("binning", fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_delay_binned_1d_1(df, model_name, subject=None, n_bins=N_X_BINS):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_delay = df[df['stimd_c'] == 'SS']
    # df_stim = df[df['ttype_c']!='VG'].copy()
    df_stim = df[df['ttype_c']=='DS'].copy()
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

    fname = f"fig_delay_1d_{title_subj}.pdf"
    out_path = get_plot_path("no binning", fname, model_name)
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
    fname = f"fig_stim_1d_{title_subj}.pdf"
    out_path = get_plot_path("no binning", fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_timepoint_deltas_binned_1d_overlay(df, model_name, subject=None, n_bins=N_X_BINS):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- filtros como en tu función (ajusta si hace falta) ---
    df_base = df.copy()
    # df_A = df_base[df_base["stimd_c"] == "SS"].copy()   # para TP3-TP1
    # df_B = df_base[df_base["ttype_c"] == "DS"].copy()   # para TP4-TP3
    df_A = df_base.copy()   # para TP3-TP1
    df_B = df_base.copy()   # para TP4-TP3 
    if subject is not None:
        df_A = df_A[df_A["subject"] == subject].copy()
        df_B = df_B[df_B["subject"] == subject].copy()

    needed_cols = ["correct_bool", "p_model_correct", "subject", "timepoint_1", "timepoint_3", "timepoint_4"]
    df_A = df_A.dropna(subset=needed_cols)
    df_B = df_B.dropna(subset=needed_cols)

    if df_A.empty or df_B.empty:
        print(f"  (sin datos válidos para overlay en {subject})")
        return

    # --- deltas ---
    df_A["delta"] = df_A["timepoint_3"] - df_A["timepoint_1"]
    df_A["delta_kind"] = "TP3-TP1"

    df_B["delta"] = df_B["timepoint_4"] - df_B["timepoint_3"]
    df_B["delta_kind"] = "TP4-TP3"

    # limpiar inf/nan
    df_A["delta"] = df_A["delta"].replace([np.inf, -np.inf], np.nan)
    df_A = df_A.dropna(subset=["delta"])
    df_B["delta"] = df_B["delta"].replace([np.inf, -np.inf], np.nan)
    df_B = df_B.dropna(subset=["delta"])
    if df_A.empty or df_B.empty:
        print(f"  (sin datos tras limpiar inf/nan en {subject})")
        return

    # --- bins comunes (comparables) ---
    all_delta = pd.concat([df_A["delta"], df_B["delta"]], ignore_index=True)
    _, edges = pd.qcut(all_delta, q=n_bins, retbins=True, duplicates="drop")

    # asignamos esos edges a ambos datasets
    df_A["x_bin"] = pd.cut(df_A["delta"], bins=edges, include_lowest=True)
    df_B["x_bin"] = pd.cut(df_B["delta"], bins=edges, include_lowest=True)

    df_long = pd.concat([df_A, df_B], ignore_index=True)

    # centros por bin (usamos el delta mediano en cada bin, por delta_kind)
    centers = (
        df_long.groupby(["delta_kind", "x_bin"], observed=True)["delta"]
        .median().rename("center").reset_index()
        .sort_values(["delta_kind", "center"])
    )

    subj = (
        df_long.groupby(["delta_kind", "x_bin", "subject"], observed=True)
        .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
        )
        .reset_index()
        .merge(centers, on=["delta_kind", "x_bin"], how="left")
    )

    plot_df = subj.melt(
        id_vars=["delta_kind", "x_bin", "subject", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_df["kind"] = plot_df["kind"].map({"data_acc": "Data", "model_acc": "Model"})

    # --- plot único ---
    fig, ax = plt.subplots(figsize=(6, 5))

    # Modelo: líneas + banda CI, dos curvas (hue=delta_kind)
    label_map = {
    "TP3-TP1": "Time in corridor",
    "TP4-TP3": "Time out of corridor",
    }
    sns.lineplot(data=plot_df[plot_df["kind"] == "Model"],x="center", y="acc",hue = "delta_kind", hue_order=list(label_map.keys()), errorbar=("ci", 95), err_style="band",ax=ax)

    # Data: puntos + barras CI, dos curvas
    sns.lineplot(data=plot_df[plot_df["kind"] == "Data"],x="center", y="acc",hue="delta_kind",errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0,ax=ax, zorder=10, legend=False, hue_order=list(label_map.keys()),)

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frac. correct responses")

    title_subj = subject if subject is not None else ""

    sns.despine()
    fig.tight_layout()

    fname = f"fig_tp_deltas_overlay_1d_All_{title_subj}.pdf"
    out_path = get_plot_path("extended_corridor", fname, model_name)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [label_map.get(l, l) for l in labels]
    ax.legend(handles, new_labels, title=None)
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

def debug_alignment(df_sub, traces_col, align_col, dt, n_show=5, label=""):
    """
    df_sub: dataframe con las filas (trials) que estás intentando plotear (p.ej. un bin)
    traces_col: columna del trace que estás usando ("trace_L"/"trace_C"/"trace_R" ya escogida)
    align_col:  "timepoint_3" o "timepoint_4"
    dt:         DT_TRACES
    """
    # extraer pares (trace, tp)
    traces = df_sub[traces_col].tolist()
    tps    = df_sub[align_col].tolist()

    n = len(traces)
    n_trace_none = 0
    n_trace_empty = 0
    n_tp_none = 0
    n_tp_nan = 0
    n_k_oob = 0
    n_ok = 0

    examples = []

    for tr, tp in zip(traces, tps):
        if tr is None:
            n_trace_none += 1
            continue

        a = np.asarray(tr, dtype=float).ravel()
        if a.size == 0:
            n_trace_empty += 1
            continue

        if tp is None:
            n_tp_none += 1
            continue

        try:
            tp = float(tp)
        except Exception:
            n_tp_nan += 1
            continue

        if not np.isfinite(tp):
            n_tp_nan += 1
            continue

        k = int(np.round(tp / dt))
        if k < 0 or k >= a.size:
            n_k_oob += 1
            if len(examples) < n_show:
                examples.append((a.size, tp, k, tp/dt))
            continue

        n_ok += 1
        if len(examples) < n_show:
            examples.append((a.size, tp, k, tp/dt))

    print(f"\n=== DEBUG ALIGN {label} ===")
    print(f"rows={n}")
    print(f"trace None={n_trace_none}, trace empty={n_trace_empty}")
    print(f"tp None={n_tp_none}, tp nonfinite/bad={n_tp_nan}")
    print(f"k out-of-bounds={n_k_oob}")
    print(f"OK={n_ok}")
    if examples:
        print("Examples: (len_trace, tp, k, tp/dt)")
        for e in examples:
            print("  ", e)

def _build_align_layout(traces, align_times, dt=DT_TRACES, clip=True):
    """
    traces:      lista de arrays 1D (uno por trial)
    align_times: lista de floats (segundos desde inicio de trial), misma longitud
    clip:        si True, fuerza k dentro [0, len-1] en vez de descartar

    Devuelve layout + lista de índices (keep_idx) de trials válidos.
    """
    keep_idx = []
    lens     = []
    ks       = []

    for i, (tr, tp) in enumerate(zip(traces, align_times)):
        if tr is None or tp is None:
            continue
        a = np.asarray(tr, dtype=float).ravel()
        if a.size == 0:
            continue

        k = int(np.floor(float(tp) / dt))

        if clip:
            k = max(0, min(k, a.size - 1))
        else:
            if k < 0 or k >= a.size:
                continue

        keep_idx.append(i)
        lens.append(a.size)
        ks.append(k)

    if len(keep_idx) == 0:
        return None

    left_len  = max(ks)                             # samples antes del ancla
    right_len = max(L - k for L, k in zip(lens, ks))# desde ancla hasta final
    T = left_len + right_len
    anchor_col = left_len

    starts = [anchor_col - k for k in ks]  # start col para cada trial válido

    return {
        "T": T,
        "anchor_col": anchor_col,
        "starts": starts,
        "keep_idx": keep_idx,
        "dt": dt,
    }

def _mean_sem_from_aligned(traces, align_times, dt=DT_TRACES, clip=True):
    layout = _build_align_layout(traces, align_times, dt=dt, clip=clip)
    if layout is None:
        return None

    arr = _apply_align_layout(traces, layout)   # (n_trials, T)
    t   = _t_axis_from_layout(layout)           # (T,)

    mean = np.nanmean(arr, axis=0)
    n_eff = np.sum(np.isfinite(arr), axis=0).clip(min=1)
    sem  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_eff)

    return dict(t=t, mean=mean, sem=sem, layout=layout, n_trials=arr.shape[0])

def _stack_subject_means_no_interp(subject_summaries, dt=DT_TRACES):
    S = [s for s in subject_summaries if s is not None]
    if len(S) == 0:
        return None

    lefts  = [s["layout"]["anchor_col"] for s in S]
    rights = [s["layout"]["T"] - s["layout"]["anchor_col"] for s in S]

    left_max  = int(max(lefts))
    right_max = int(max(rights))
    T = left_max + right_max
    anchor = left_max

    M = np.full((len(S), T), np.nan, dtype=float)

    for i, s in enumerate(S):
        y = np.asarray(s["mean"], dtype=float)
        L = y.size
        subj_left = int(s["layout"]["anchor_col"])
        start = anchor - subj_left
        M[i, start:start+L] = y

    t = (np.arange(T, dtype=float) - anchor) * dt

    mean = np.nanmean(M, axis=0)
    n_eff = np.sum(np.isfinite(M), axis=0).clip(min=1)
    sem  = np.nanstd(M, axis=0, ddof=1) / np.sqrt(n_eff)

    return dict(t=t, mean=mean, sem=sem, n_subjects=len(S), subj_matrix=M)

def _apply_align_layout(traces, layout):
    """
    Aplica el layout a una lista de trazas (misma longitud original que align_times).
    Devuelve arr (n_trials_valid, T) con NaNs.
    """
    T = layout["T"]
    out = np.full((len(layout["keep_idx"]), T), np.nan, dtype=float)

    for j, (i, start) in enumerate(zip(layout["keep_idx"], layout["starts"])):
        a = np.asarray(traces[i], dtype=float).ravel()
        out[j, start:start + a.size] = a

    return out


def _t_axis_from_layout(layout):
    T = layout["T"]
    anchor_col = layout["anchor_col"]
    dt = layout["dt"]
    return (np.arange(T, dtype=float) - anchor_col) * dt  # t=0 en el timepoint elegido

# ============================================================
# 1) Correct trials: winning population vs delay_duration binned
# ============================================================
def plot_traces_correct_by_delay(df, model_name, n_bins=7, subject=None ,kind_dir="traces", align="timepoint_4"):
    df["model_choice"] = df.apply(lambda r: _get_model_choice(r, 0.5), axis=1)
    print(f"Total trials with model choice: {df['model_choice'].notna().sum()} / {len(df)}")
    df = df[df["model_choice"].notna()].copy()
    df['correct_model'] = df["x_c"] == df["model_choice"]
    df_s = df[df["correct_model"] == True].copy()
    
    for col in ["x_c", "r_c"]:
        if col in df_s.columns:
            df_s[col] = df_s[col].astype("string").str.strip().str.upper()

    df_s = df_s[df_s["r_c"].isin(["L","C","R"])].copy()
    df_s = df_s[df_s["delay_duration"].notna()].copy()

    if subject is not None:
        df_s = df_s[df_s["subject"] == subject].copy()
        if df_s.empty:
            print(f"[{subject}] Sin trials correctos para trazas por delay.")
            return

    else:
        if df_s.empty:
            print("[GROUP] Sin trials correctos para trazas por delay.")
            return

    # Binning (para group lo hago GLOBAL, que es lo más estable)
    df_s["delay_bin"], _ = pd.qcut(df_s["delay_duration"], q=n_bins, retbins=True, duplicates="drop")
    bin_centers = df_s.groupby("delay_bin", observed=True)["delay_duration"].median().sort_values()

    fig, ax = plt.subplots(figsize=(5,4))
    palette = sns.color_palette("viridis", len(bin_centers))

    for idx, (bin_cat, center_val) in enumerate(bin_centers.items()):
        df_bin = df_s[df_s["delay_bin"] == bin_cat].copy()
        if df_bin.empty:
            continue

        if subject is not None:
            # ==== single subject: alinea trials y plotea ====
            traces_winning, align_times = [], []

            for _, row in df_bin.iterrows():
                side = row["r_c"]
                tr = row.get(SIDE_TO_TRACE_COL.get(side), None)
                tp = row.get(align, None)
                if tr is None or tp is None:
                    continue
                traces_winning.append(tr)
                align_times.append(tp)

            summ = _mean_sem_from_aligned(traces_winning, align_times, dt=DT_TRACES)
            if summ is None:
                print(f"[{subject}] No se pudieron alinear trazas para bin {idx+1}.")
                continue

            t, mean_tr, sem_tr = summ["t"], summ["mean"], summ["sem"]
            label = f"Delay bin {idx+1}\n(med ~ {center_val:.2f}s)"

        else:
            # ==== group: mean por sujeto -> stack sin interpolar ====
            per_subj = []
            for s in sorted(df_bin["subject"].unique()):
                sub = df_bin[df_bin["subject"] == s]
                traces_winning, align_times = [], []

                for _, row in sub.iterrows():
                    side = row["r_c"]
                    tr = row.get(SIDE_TO_TRACE_COL.get(side), None)
                    tp = row.get(align, None)
                    if tr is None or tp is None:
                        continue
                    traces_winning.append(tr)
                    align_times.append(tp)

                per_subj.append(_mean_sem_from_aligned(traces_winning, align_times, dt=DT_TRACES))

            grp = _stack_subject_means_no_interp(per_subj, dt=DT_TRACES)
            if grp is None:
                print(f"[GROUP] Bin {idx+1}: sin sujetos alineables.")
                continue

            t, mean_tr, sem_tr = grp["t"], grp["mean"], grp["sem"]
            label = f"Delay bin {idx+1}\n(med ~ {center_val:.2f}s)"

        col = palette[idx]
        ax.plot(t, mean_tr, color=col, lw=2, label=label)
        ax.fill_between(t, mean_tr-sem_tr, mean_tr+sem_tr, color=col, alpha=0.2)

    ax.set_xlabel(f"Time (s) aligned to {'Response' if align == 'timepoint_4' else 'Corridor end'} (t=0)", fontsize=12)
    ax.set_ylabel("Population rate", fontsize=12)
    # ax.set_title(f"{subject if subject is not None else f'{grp['n_subjects']} subjects'} - Correct trials by delay", fontsize=12)
    if align == "timepoint_4":
        ax.set_xlim([-2.0, 0.0])
    elif align == "timepoint_3":
        ax.set_xlim([-1, 1])
    ax.set_ylim(-0.2,4)
    n_cols = len(bin_centers)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=n_cols//2 + n_cols%2, frameon=False, fontsize=10, handlelength=2.5, columnspacing=1.5)
    sns.despine()
    fig.subplots_adjust(bottom=0.4)

    fname = f"traces_correct_by_delay{('_'+ subject) if subject is not None else ''}_{align}.pdf"
    out_path = get_plot_path(kind_dir, fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ==================================================================
# 2) Error trials: winning vs good-choice vs other (3 poblaciones)
# ==================================================================
def plot_traces_errors_chosen(df, model_name, kind_dir="traces",
                                        thr_model_choice=0.5, align="timepoint_4",
                                        subject=None, dt=DT_TRACES, clip=True):
    """
    single:
      - usa trials donde el MODELO elige (end>thr) y además el modelo se equivoca (x_c != model_choice)
      - plotea 3 trazas: chosen(model), correct(side), other

    group:
      - por sujeto: calcula mean de las 3 (alineando trials)
      - luego media entre sujetos SIN interpolar (padding NaNs)
    """
    df_s = df.copy()

    for col in ["x_c", "r_c"]:
        if col in df_s.columns:
            df_s[col] = df_s[col].astype("string").str.strip().str.upper()

    df_s = df_s[df_s["x_c"].isin(ALL_SIDES)].copy()

    if subject is not None:
        df_s = df_s[df_s["subject"] == subject].copy()
        if df_s.empty:
            print(f"[{subject}] Sin trials para error-traces.")
            return
    else:
        if df_s.empty:
            print("[GROUP] Sin trials para error-traces.")
            return

    def _summaries_one_subject(df_one):
        # model choice + filtrar donde el modelo decide y se equivoca
        df_one = df_one.copy()
        df_one["model_choice"] = df_one.apply(lambda r: _get_model_choice(r, thr_model_choice), axis=1)
        df_one = df_one[df_one["model_choice"].notna()].copy()
        df_one = df_one[df_one["x_c"] != df_one["model_choice"]].copy()  # errores del modelo
        df_one = df_one[df_one['stimd_c'] == 'SS']
        df_one = df_one[df_one['ttype_c'] == 'DL']
        if df_one.empty:
            return None, None, None

        traces_win, traces_good, traces_other, tps = [], [], [], []

        for _, row in df_one.iterrows():
            tp = row.get(align, None)
            if tp is None:
                continue

            chosen  = row["model_choice"]
            correct = row["x_c"]
            others = [s for s in ALL_SIDES if s not in (chosen, correct)]
            if len(others) != 1:
                continue
            other = others[0]

            tr_L = row.get(SIDE_TO_TRACE_COL["L"], None)
            tr_C = row.get(SIDE_TO_TRACE_COL["C"], None)
            tr_R = row.get(SIDE_TO_TRACE_COL["R"], None)
            if tr_L is None or tr_C is None or tr_R is None:
                continue

            tr_map = {"L": np.asarray(tr_L, float), "C": np.asarray(tr_C, float), "R": np.asarray(tr_R, float)}
            traces_win.append(tr_map[chosen])
            traces_good.append(tr_map[correct])
            traces_other.append(tr_map[other])
            tps.append(tp)

        # mismo layout para las 3 poblaciones (usamos winning como referencia)
        summ_win  = _mean_sem_from_aligned(traces_win,  tps, dt=dt, clip=clip)
        if summ_win is None:
            return None, None, None

        layout = summ_win["layout"]
        arr_good  = _apply_align_layout(traces_good,  layout)
        arr_other = _apply_align_layout(traces_other, layout)
        t = summ_win["t"]

        def _mean_sem_arr(arr):
            mean = np.nanmean(arr, axis=0)
            n_eff = np.sum(np.isfinite(arr), axis=0).clip(min=1)
            sem  = np.nanstd(arr, ddof=1, axis=0) / np.sqrt(n_eff)
            return mean, sem

        mean_good,  sem_good  = _mean_sem_arr(arr_good)
        mean_other, sem_other = _mean_sem_arr(arr_other)

        summ_good  = dict(t=t, mean=mean_good,  sem=sem_good,  layout=layout)
        summ_other = dict(t=t, mean=mean_other, sem=sem_other, layout=layout)

        return summ_win, summ_good, summ_other

    # ---------- SINGLE ----------
    if subject is not None:
        summ_win, summ_good, summ_other = _summaries_one_subject(df_s)
        if summ_win is None:
            print(f"[{subject}] No se pudieron alinear trazas de error.")
            return

        t = summ_win["t"]
        fig, ax = plt.subplots(figsize=(4,4))

        # ax.plot(t, summ_win["mean"],   color="#d73027", lw=2, label="Winning (model choice)")
        # ax.fill_between(t, summ_win["mean"]-summ_win["sem"], summ_win["mean"]+summ_win["sem"], color="#d73027", alpha=0.2)

        ax.plot(t, summ_good["mean"],  color="#F5A262", lw=2, label="Non-chosen correct")
        ax.fill_between(t, summ_good["mean"]-summ_good["sem"], summ_good["mean"]+summ_good["sem"], color="#F5A262", alpha=0.2)

        ax.plot(t, summ_other["mean"], color="#A67C52", lw=2, label="Non-chosen error")
        ax.fill_between(t, summ_other["mean"]-summ_other["sem"], summ_other["mean"]+summ_other["sem"], color="#A67C52", alpha=0.2)

        ax.set_xlabel(f"Time (s) aligned to {'Response' if align == 'timepoint_4' else 'Corridor end'} (t=0)", fontsize=12)
        ax.set_ylabel("Population rate", fontsize=12)
        ax.set_title(f"{subject} - Error trials", fontsize=12)
        ax.legend(frameon=False, fontsize=8)
        sns.despine()
        fig.tight_layout()
        if align == "timepoint_4":
            ax.set_xlim([-4, 0.0])
        elif align == "timepoint_3":
            ax.set_xlim([-4, 4])
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.axhline(0, color="gray", ls="--", lw=1)
        fname = f"traces_errors_winning_good_bad{'_' + subject if subject is not None else ''}_{align}.pdf"
        out_path = get_plot_path(kind_dir, fname, model_name)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    # ---------- GROUP ----------
    per_subj_win, per_subj_good, per_subj_other = [], [], []

    for s in sorted(df_s["subject"].unique()):
        df_one = df_s[df_s["subject"] == s]
        sw, sg, so = _summaries_one_subject(df_one)
        per_subj_win.append(sw)
        per_subj_good.append(sg)
        per_subj_other.append(so)

    grp_win   = _stack_subject_means_no_interp(per_subj_win,   dt=dt)
    grp_good  = _stack_subject_means_no_interp(per_subj_good,  dt=dt)
    grp_other = _stack_subject_means_no_interp(per_subj_other, dt=dt)

    if grp_win is None or grp_good is None or grp_other is None:
        print("[GROUP] No se pudieron alinear trazas de error (modelo).")
        return

    # Rejilla común (sin interpolar): usamos la que tenga mayor soporte.
    # Aquí: creamos una común con padding mecánico sobre las 3
    def _pad_to_common(vec, src_t, t_common):
        out = np.full_like(t_common, np.nan, dtype=float)
        a_c = np.where(t_common == 0)[0][0]
        a_s = np.where(src_t == 0)[0][0]
        start = a_c - a_s
        out[start:start+len(vec)] = vec
        return out

    # común = stack de los tres "means" como si fueran sujetos
    fake = [
        dict(t=grp_win["t"],   mean=grp_win["mean"],   sem=grp_win["sem"],   layout={"anchor_col": np.where(grp_win["t"]==0)[0][0],   "T": len(grp_win["t"])}),
        dict(t=grp_good["t"],  mean=grp_good["mean"],  sem=grp_good["sem"],  layout={"anchor_col": np.where(grp_good["t"]==0)[0][0],  "T": len(grp_good["t"])}),
        dict(t=grp_other["t"], mean=grp_other["mean"], sem=grp_other["sem"], layout={"anchor_col": np.where(grp_other["t"]==0)[0][0], "T": len(grp_other["t"])}),
    ]
    common = _stack_subject_means_no_interp(fake, dt=dt)
    t = common["t"]

    m_win   = _pad_to_common(grp_win["mean"],   grp_win["t"],   t)
    m_good  = _pad_to_common(grp_good["mean"],  grp_good["t"],  t)
    m_other = _pad_to_common(grp_other["mean"], grp_other["t"], t)

    s_win   = _pad_to_common(grp_win["sem"],   grp_win["t"],   t)
    s_good  = _pad_to_common(grp_good["sem"],  grp_good["t"],  t)
    s_other = _pad_to_common(grp_other["sem"], grp_other["t"], t)

    fig, ax = plt.subplots(figsize=(4,4))

    ax.plot(t, m_good,  color="#F5A262", lw=2, label=f"Non chosen correct")
    ax.fill_between(t, m_good-s_good, m_good+s_good, color="#F5A262", alpha=0.2)

    ax.plot(t, m_other, color="#A67C52", lw=2, label=f"Non chosen error")
    ax.fill_between(t, m_other-s_other, m_other+s_other, color="#A67C52", alpha=0.2)

    ax.set_xlabel(f"Time (s) aligned to {'Response' if align == 'timepoint_4' else 'Corridor end'} (t=0)", fontsize=12)
    ax.set_ylabel("Population rate", fontsize=12)
    ax.set_title(f"{len(df_s['subject'].unique())} subjects - Error trials", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()
    if align == "timepoint_4":
            ax.set_xlim([-3.5, 0.0])
    elif align == "timepoint_3":
        ax.set_xlim([-3, 3])
    ax.set_ylim(-2,0.2)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axhline(0, color="gray", ls="--", lw=1)
    fname = f"traces_errors_winning_good_bad{'_' + subject if subject is not None else ''}_{align}.pdf"
    out_path = get_plot_path(kind_dir, fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ============================================================
# 3) Winning population traces: correct vs error
# ============================================================
def plot_traces_winning_correct_vs_error(df, model_name,subject=None, kind_dir="traces",
                                        align="timepoint_4",
                                        dt=DT_TRACES, clip=True):
    """
    single:
      - winning = población elegida (r_c) en cada trial
      - compara traza media de winning en trials correctos vs incorrectos (animal)

    group:
      - por cada sujeto: calcula mean(trace winning) para correct y para error (alineando trials)
      - luego media entre sujetos SIN interpolar (padding NaNs)
    """
    df_s = df.copy()
    df_s["model_choice"] = df_s.apply(lambda r: _get_model_choice(r, 0.5), axis=1)
    df_s = df_s[df_s["model_choice"].notna()].copy()
    df_s['correct_model'] = df_s["x_c"] == df_s["model_choice"]
    for col in ["x_c", "r_c"]:
        if col in df_s.columns:
            df_s[col] = df_s[col].astype("string").str.strip().str.upper()

    df_s = df_s[df_s["r_c"].isin(["L", "C", "R"])].copy()

    if subject is not None:
        df_s = df_s[df_s["subject"] == subject].copy()
        if df_s.empty:
            print(f"[{subject}] Sin trials válidos para winning correct vs error.")
            return
    else:
        if df_s.empty:
            print("[GROUP] Sin trials válidos para winning correct vs error.")
            return

    def _collect_traces(df_block):
        traces, tps = [], []
        for _, row in df_block.iterrows():
            tp = row.get(align, None)
            if tp is None:
                continue
            side = row['model_choice']
            tr = row.get(SIDE_TO_TRACE_COL.get(side), None)
            if tr is None:
                continue
            traces.append(tr); tps.append(tp)
        return traces, tps

    # ---------- SINGLE ----------
    if subject is not None:
        df_corr = df_s[df_s["correct_model"] == True]
        df_err  = df_s[df_s["correct_model"] == False]

        tr_corr, tp_corr = _collect_traces(df_corr)
        tr_err,  tp_err  = _collect_traces(df_err)

        summ_corr = _mean_sem_from_aligned(tr_corr, tp_corr, dt=dt, clip=clip)
        summ_err  = _mean_sem_from_aligned(tr_err,  tp_err,  dt=dt, clip=clip)

        if summ_corr is None or summ_err is None:
            print(f"[{subject}] No se pudieron alinear trazas correct/error.")
            return

        t_corr, m_corr, s_corr = summ_corr["t"], summ_corr["mean"], summ_corr["sem"]
        t_err,  m_err,  s_err  = summ_err["t"],  summ_err["mean"],  summ_err["sem"]

        # Para plot conjunto sin interpolar: apilamos 2 “sujetos” ficticios usando padding
        # (si prefieres, puedes construir un layout global como antes; esto es más mecánico)
        # -> aquí lo hago simple: reuso el stacker con dos summaries
        grp_corr = _stack_subject_means_no_interp([summ_corr], dt=dt)
        grp_err  = _stack_subject_means_no_interp([summ_err],  dt=dt)
        t = grp_corr["t"]  # mismo dt y t=0, pero longitudes pueden variar: usamos t de corr y pad en plot

        # Re-embed err en el mismo eje t (padding)
        # (mecánico: creo una rejilla común con left/right máximos entre ambos)
        both = _stack_subject_means_no_interp([summ_corr, summ_err], dt=dt)
        t = both["t"]
        # reconstruyo dos series en esa rejilla:
        # summ_corr y summ_err ya están en both["subj_matrix"] en filas 0 y 1
        m_corr2 = both["subj_matrix"][0]
        m_err2  = both["subj_matrix"][1]

        # sem single-subject: usamos sem original, pero hay NaNs por padding -> las rellenamos alineando igual
        # (aprox: ponemos sem en las mismas posiciones; fuera NaN)
        def _pad_sem(summ, target_layout_like):
            # target_layout_like: the "both" layout values encoded in left/right (via anchor)
            # easiest: reconstruct by start shift:
            # Lo hacemos bien:
            T = both["subj_matrix"].shape[1]
            out = np.full(T, np.nan, float)
            subj_left = int(summ["layout"]["anchor_col"])
            anchor = int(np.where(t == 0)[0][0])
            start = anchor - subj_left
            out[start:start+len(summ["sem"])] = summ["sem"]
            return out

        s_corr2 = _pad_sem(summ_corr, both)
        s_err2  = _pad_sem(summ_err,  both)

        # PLOT
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, m_corr2, color="#1a9850", lw=2, label="Correct trials")
        ax.fill_between(t, m_corr2 - s_corr2, m_corr2 + s_corr2, color="#1a9850", alpha=0.2)

        ax.plot(t, m_err2,  color="#d73027", lw=2, label="Error trials")
        ax.fill_between(t, m_err2 - s_err2, m_err2 + s_err2, color="#d73027", alpha=0.2)
        if align == "timepoint_4":
            ax.set_xlim([-2.0, 0.0])
        elif align == "timepoint_3":
            ax.set_xlim([-1, 1])
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel(f"Time (s) aligned to {'Response' if align == 'timepoint_4' else 'Corridor end'} (t=0)", fontsize=12)
        ax.set_ylabel("Population rate", fontsize=12)
        ax.set_title(f"{subject} - Choice population: correct vs error", fontsize=12)
        ax.legend(frameon=False, fontsize=8)
        sns.despine()
        fig.tight_layout()

        fname = f"traces_winning_correct_vs_error{'_' + subject if subject is not None else ''}_{align}.pdf"
        out_path = get_plot_path(kind_dir, fname, model_name)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    # ---------- GROUP ----------
    per_subj_corr = []
    per_subj_err  = []

    for s in sorted(df_s["subject"].unique()):
        df_sub = df_s[df_s["subject"] == s]
        df_corr = df_sub[df_sub["correct_model"] == True]
        df_err  = df_sub[df_sub["correct_model"] == False]

        tr_corr, tp_corr = _collect_traces(df_corr)
        tr_err,  tp_err  = _collect_traces(df_err)

        per_subj_corr.append(_mean_sem_from_aligned(tr_corr, tp_corr, dt=dt, clip=clip))
        per_subj_err.append (_mean_sem_from_aligned(tr_err,  tp_err,  dt=dt, clip=clip))

    grp_corr = _stack_subject_means_no_interp(per_subj_corr, dt=dt)
    grp_err  = _stack_subject_means_no_interp(per_subj_err,  dt=dt)

    if grp_corr is None or grp_err is None:
        print("[GROUP] No se pudieron alinear trazas correct/error.")
        return

    # Rejilla común entre las dos curvas (corr vs err) sin interpolar: padding mecánico
    # -> usamos el mismo truco: tratamos corr y err como “dos sujetos” y stackeamos
    fake_corr = dict(t=grp_corr["t"], mean=grp_corr["mean"], sem=grp_corr["sem"], layout={"anchor_col": np.where(grp_corr["t"]==0)[0][0], "T": len(grp_corr["t"])})
    fake_err  = dict(t=grp_err["t"],  mean=grp_err["mean"],  sem=grp_err["sem"],  layout={"anchor_col": np.where(grp_err["t"]==0)[0][0],  "T": len(grp_err["t"])})
    both = _stack_subject_means_no_interp([fake_corr, fake_err], dt=dt)

    t = both["t"]
    m_corr = both["subj_matrix"][0]
    m_err  = both["subj_matrix"][1]

    # Para SEM: igual, padding mecánico
    def _pad_vec_to_both(vec, src_t):
        out = np.full_like(t, np.nan, dtype=float)
        anchor = np.where(t == 0)[0][0]
        src_anchor = np.where(src_t == 0)[0][0]
        start = anchor - src_anchor
        out[start:start+len(vec)] = vec
        return out

    s_corr = _pad_vec_to_both(grp_corr["sem"], grp_corr["t"])
    s_err  = _pad_vec_to_both(grp_err["sem"],  grp_err["t"])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(t, m_corr, color="#1a9850", lw=2, label=f"Correct")
    ax.fill_between(t, m_corr - s_corr, m_corr + s_corr, color="#1a9850", alpha=0.2)
    ax.plot(t, m_err,  color="#d73027", lw=2, label=f"Error")
    ax.fill_between(t, m_err - s_err, m_err + s_err, color="#d73027", alpha=0.2)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(f"Time (s) aligned to {'Response' if align == 'timepoint_4' else 'Corridor end'} (t=0)", fontsize=12)
    ax.set_ylabel("Population rate", fontsize=12)
    ax.set_title(f"{grp_corr['n_subjects']} subjects - Choice population: correct vs error", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    if align == "timepoint_4":
            ax.set_xlim([-2.0, 0.0])
    elif align == "timepoint_3":
        ax.set_xlim([-1, 1])
    ax.set_ylim(-0.2,4.2)
    
    sns.despine()
    fig.tight_layout()
    
    fname = f"traces_winning_correct_vs_error{'_' + subject if subject is not None else ''}_{align}.pdf"
    out_path = get_plot_path(kind_dir, fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_traces_winning_correct_vs_error(df, model_name, subject=None, kind_dir="traces",
                                         align="timepoint_4", delayd=None, stimd=None,
                                         dt=DT_TRACES, clip=True):
    """
    Nuevo DF:
      - trace_correct: traza media de la población GANADORA (winner) en reps correctas
      - trace_error:   traza media de la población GANADORA (winner) en reps incorrectas

    single:
      - usa directamente trace_correct vs trace_error del sujeto

    group:
      - por sujeto: alinea trace_correct y trace_error (padding NaNs, sin interpolar)
      - luego media entre sujetos
    """
    df_s = df.copy()

    # normaliza columnas si existen (no dependemos ya de model_choice)
    if "r_c" in df_s.columns:
        df_s["r_c"] = df_s["r_c"].astype("string").str.strip().str.upper()
        df_s = df_s[df_s["r_c"].isin(["L", "C", "R"])].copy()

    if subject is not None:
        df_s = df_s[df_s["subject"] == subject].copy()
        if df_s.empty:
            print(f"[{subject}] Sin trials válidos para correct vs error.")
            return
    else:
        if df_s.empty:
            print("[GROUP] Sin trials válidos para correct vs error.")
            return

    def _collect_traces(df_block, trace_col):
        traces, tps = [], []
        for _, row in df_block.iterrows():
            if delayd is not None and row.get("ttype_c", None) != delayd:
                continue
            if stimd is not None and row.get("stimd_c", None) != stimd:
                continue
            tp = row.get(align, None)
            if tp is None:
                continue
            tr = row.get(trace_col, None)
            if tr is None:
                continue
            traces.append(tr); tps.append(tp)
        return traces, tps

    # ---------- SINGLE ----------
    if subject is not None:
        tr_corr, tp_corr = _collect_traces(df_s, "trace_correct")
        tr_err,  tp_err  = _collect_traces(df_s, "trace_error")

        summ_corr = _mean_sem_from_aligned(tr_corr, tp_corr, dt=dt, clip=clip)
        summ_err  = _mean_sem_from_aligned(tr_err,  tp_err,  dt=dt, clip=clip)

        if summ_corr is None or summ_err is None:
            print(f"[{subject}] No se pudieron alinear trazas correct/error.")
            return

        # rejilla común sin interpolar (padding mecánico)
        both = _stack_subject_means_no_interp([summ_corr, summ_err], dt=dt)
        t = both["t"]
        m_corr = both["subj_matrix"][0]
        m_err  = both["subj_matrix"][1]

        # padding de SEM a la rejilla común
        def _pad_vec_to_both(vec, src_t):
            out = np.full_like(t, np.nan, dtype=float)
            anchor = np.where(t == 0)[0][0]
            src_anchor = np.where(src_t == 0)[0][0]
            start = anchor - src_anchor
            out[start:start+len(vec)] = vec
            return out

        s_corr = _pad_vec_to_both(summ_corr["sem"], summ_corr["t"])
        s_err  = _pad_vec_to_both(summ_err["sem"],  summ_err["t"])

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, m_corr, color="#1a9850", lw=2, label="Correct reps (winner)")
        ax.fill_between(t, m_corr - s_corr, m_corr + s_corr, color="#1a9850", alpha=0.2)

        ax.plot(t, m_err,  color="#d73027", lw=2, label="Error reps (winner)")
        ax.fill_between(t, m_err - s_err, m_err + s_err, color="#d73027", alpha=0.2)

        if align == "timepoint_4":
            ax.set_xlim([-2.0, 0.0])
        elif align == "timepoint_3":
            ax.set_xlim([-1, 1])

        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel(f"Time (s) aligned to {'Response' if align == 'timepoint_4' else 'Corridor end'} (t=0)", fontsize=12)
        ax.set_ylabel("Population rate", fontsize=12)
        ax.set_title(f"{subject} - Winner population: correct vs error", fontsize=12)
        ax.legend(frameon=False, fontsize=8)
        sns.despine()
        fig.tight_layout()

        fname = f"traces_winning_correct_vs_error_{subject}_{align}{f'_{delayd}' if delayd is not None else ''}{f'_{stimd}' if stimd is not None else ''}.pdf"
        out_path = get_plot_path(kind_dir, fname, model_name)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    # ---------- GROUP ----------
    per_subj_corr = []
    per_subj_err  = []

    for s in sorted(df_s["subject"].unique()):
        df_sub = df_s[df_s["subject"] == s]

        tr_corr, tp_corr = _collect_traces(df_sub, "trace_correct")
        tr_err,  tp_err  = _collect_traces(df_sub, "trace_error")

        per_subj_corr.append(_mean_sem_from_aligned(tr_corr, tp_corr, dt=dt, clip=clip))
        per_subj_err.append (_mean_sem_from_aligned(tr_err,  tp_err,  dt=dt, clip=clip))

    grp_corr = _stack_subject_means_no_interp(per_subj_corr, dt=dt)
    grp_err  = _stack_subject_means_no_interp(per_subj_err,  dt=dt)

    if grp_corr is None or grp_err is None:
        print("[GROUP] No se pudieron alinear trazas correct/error.")
        return

    # rejilla común entre corr y err (sin interpolar)
    fake_corr = dict(t=grp_corr["t"], mean=grp_corr["mean"], sem=grp_corr["sem"],
                     layout={"anchor_col": np.where(grp_corr["t"] == 0)[0][0], "T": len(grp_corr["t"])})
    fake_err  = dict(t=grp_err["t"],  mean=grp_err["mean"],  sem=grp_err["sem"],
                     layout={"anchor_col": np.where(grp_err["t"] == 0)[0][0],  "T": len(grp_err["t"])})
    both = _stack_subject_means_no_interp([fake_corr, fake_err], dt=dt)

    t = both["t"]
    m_corr = both["subj_matrix"][0]
    m_err  = both["subj_matrix"][1]

    def _pad_vec_to_both(vec, src_t):
        out = np.full_like(t, np.nan, dtype=float)
        anchor = np.where(t == 0)[0][0]
        src_anchor = np.where(src_t == 0)[0][0]
        start = anchor - src_anchor
        out[start:start+len(vec)] = vec
        return out

    s_corr = _pad_vec_to_both(grp_corr["sem"], grp_corr["t"])
    s_err  = _pad_vec_to_both(grp_err["sem"],  grp_err["t"])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(t, m_corr, color="#1a9850", lw=2, label="Correct reps (winner)")
    ax.fill_between(t, m_corr - s_corr, m_corr + s_corr, color="#1a9850", alpha=0.2)

    ax.plot(t, m_err,  color="#d73027", lw=2, label="Error reps (winner)")
    ax.fill_between(t, m_err - s_err, m_err + s_err, color="#d73027", alpha=0.2)

    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(f"Time (s) aligned to {'Response' if align == 'timepoint_4' else 'Corridor end'} (t=0)", fontsize=12)
    ax.set_ylabel("Population rate", fontsize=12)
    ax.set_title(f"{grp_corr['n_subjects']} subjects - Winner population: correct vs error", fontsize=12)
    ax.legend(frameon=False, fontsize=8)

    if align == "timepoint_4":
        ax.set_xlim([-2.0, 0.0])
    elif align == "timepoint_3":
        ax.set_xlim([-1, 1])
    ax.set_ylim(-0.2, 4.2)

    sns.despine()
    fig.tight_layout()

    fname = f"traces_winning_correct_vs_error_{align}{f'_{delayd}' if delayd is not None else ''}{f'_{stimd}' if stimd is not None else ''}.pdf"
    out_path = get_plot_path(kind_dir, fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ========= MAIN =========
if __name__ == "__main__":
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_context("talk", font_scale=1)

    MODEL_NAME = "spatial_reduced3"  # Cambiar aquí el modelo a analizar
    
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
        plot_delay_binned_1d(df=df_subj, subject=subject, model_name=MODEL_NAME, n_bins=N_X_BINS)
        plot_timepoint_deltas_binned_1d_overlay(df=df_subj, subject=subject, model_name=MODEL_NAME, n_bins=N_X_BINS)
    plot_scatter_delay_stim(df, MODEL_NAME, show_bins=False)
    plot_heatmaps_delay_stim(df, MODEL_NAME)
    # plot_categorical_performance_all(df, MODEL_NAME)
    plot_delay_binned_1d(df=df, model_name=MODEL_NAME, n_bins=N_X_BINS)
    plot_delay_binned_1d_1(df=df, model_name=MODEL_NAME, n_bins=N_X_BINS)
    plot_timepoint_deltas_binned_1d_overlay(df=df, model_name=MODEL_NAME, n_bins=N_X_BINS)
    from helpers.plots import plot_delay_stim_1d_multipanel_all_subjects, plot_delay_binned_1d_two_models
    plot_delay_stim_1d_multipanel_all_subjects(df=df, model_name=MODEL_NAME, n_bins=N_X_BINS, max_cols=5)

    pathB = os.path.join(paths.PARAMS_DIR, f"df_externalU2_randomx0.parquet")
    pathA = os.path.join(paths.PARAMS_DIR, f"df_spatial_reduced3.parquet")
    dfA = parse_model_probs_column(pd.read_parquet(pathA), col="model")
    dfB = parse_model_probs_column(pd.read_parquet(pathB), col="model")

    plot_delay_binned_1d_two_models(dfA, dfB, model_name_A="spatial_reduced3", model_name_B="externalU2_randomx0",
        # subject="A83",
        n_bins=N_X_BINS, color_A="#1f77b4", color_B="#d62728",)


    traces_path = os.path.join(paths.PARAMS_DIR, f"df_traces_{MODEL_NAME}.parquet")
    traces_ce_path = os.path.join(paths.PARAMS_DIR, f"df_traces_correct_error_{MODEL_NAME}.parquet")
    df_traces = pd.read_parquet(traces_path)
    df_traces_ce = pd.read_parquet(traces_ce_path)
    subjects = sorted(df_traces["subject"].unique())
    for subject in tqdm(subjects, desc="Plotting traces"):
        df_subj = df_traces[df_traces["subject"] == subject].copy()
        df_subj_ce = df_traces_ce[df_traces_ce["subject"] == subject].copy()
        for align in ["timepoint_3", "timepoint_4"]:
            plot_traces_correct_by_delay(df_subj, subject=subject, model_name=MODEL_NAME, n_bins=7, align=align)
            plot_traces_winning_correct_vs_error(df_subj_ce, subject=subject, model_name=MODEL_NAME, align=align)
            plot_traces_errors_chosen(df_subj, subject=subject, model_name=MODEL_NAME, align=align)
    for align in ["timepoint_3", "timepoint_4"]:
        plot_traces_correct_by_delay(df_traces,  model_name=MODEL_NAME, n_bins=7, align=align)
        plot_traces_winning_correct_vs_error(df_traces_ce, model_name=MODEL_NAME, align=align)
        plot_traces_errors_chosen(df_traces, model_name=MODEL_NAME, align=align)
    
    for delayd in ['DS', 'DM', 'DL']:
        plot_traces_winning_correct_vs_error(df_traces, model_name=MODEL_NAME, delayd=delayd, align="timepoint_4")
        plot_traces_winning_correct_vs_error(df_traces, model_name=MODEL_NAME, delayd=delayd, align="timepoint_3")
    # plot_traces_correct_by_delay(df_traces, subject="All_Subjects", model_name=MODEL_NAME, n_bins=7)
    # plot_traces_winning_correct_vs_error(df_traces, subject="All_Subjects", model_name=MODEL_NAME)
    # plot_traces_errors_winning_good_bad(df_traces, subject="All_Subjects", model_name=MODEL_NAME)
    
    print("\nListo.")
