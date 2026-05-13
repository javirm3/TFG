import pandas as pd
import numpy as np
import sys, os, json, re, math
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import seaborn as sns
from numba import set_num_threads, get_num_threads
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import paths 
from helpers.sim_core_numba import simulate_counts_one_trial_heun

side_map = {'L': 0, 'C': 1, 'R': 2}
dt = 0.1 / 40
th = (0.5, 0.5, 0.5)
M = 300
SUBSAMPLE_PER_BIN = 10000

# ================== UTILIDADES PLOT ==================
def truncate_colormap(cmap_name, minval=0.2, maxval=0.9, n=256):
    """Trunca un colormap a un subrango."""
    cmap = cm.get_cmap(cmap_name, n)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

trunc_purples = truncate_colormap("Purples_r", 0, 0.7)
trunc_oranges = truncate_colormap("Oranges", 0.3, 1.0)

# ================== FUNCIONES DEL MODELO ==================
def get_onset_offset(stim_dur, delay_dur, t1, t2, t3, t4):
    if stim_dur == 'VG': return 0.0, t4
    if stim_dur == 'SS':
        if delay_dur == 'DS': return t2, t3
        if delay_dur == 'DM': return t1, t2
        if delay_dur == 'DL': return 0.0, t1
    if stim_dur == 'SM':
        if delay_dur == 'DS': return t1, t3
        if delay_dur == 'DM': return 0.0, t2
    if stim_dur == 'SL': return 0.0, t3
    if stim_dur == 'SIL': return 0.0, 0.0
    raise ValueError

def get_U_fn(amplitude, baseline, onset, offset):
    def U(t):
        t = np.asarray(t, dtype=np.float64)
        D = float(offset) - float(onset)
        u = np.full_like(t, float(baseline), dtype=np.float64)
        if D <= 0.0:
            return u
        active = (t >= onset) & (t <= offset)
        u[active] = baseline + amplitude * (t[active] - onset) / D
        return u
    return U

def get_U_spatial_fn(U_amp, U_base, t1, t2, t3, t4):
    def U(t):
        w1 = 1.0 / (t1 - 0.0)
        w2 = 1.0 / (t2 - t1)
        w3 = 1.0 / (t3 - t2)
        w4 = 1.0 / (t4 - t3)
        r1 = np.clip(t * w1,            0.0, 1.0)
        r2 = np.clip((t - t1) * w2,     0.0, 1.0)
        r3 = np.clip((t - t2) * w3,     0.0, 1.0)
        r4 = np.clip((t - t3) * w4,     0.0, 1.0)
        return U_base + 0.25 * U_amp * (r1 + r2 + r3 + r4)
    return U

def get_S_fn(amplitude, d, onset, offset):
    def S(t):
        S_base = np.where((t >= onset) & (t <= offset), amplitude, 0.0)
        tail   = np.where((t > offset) & (t <= offset + d),
                          amplitude * (1.0 - (t - offset) / d), 0.0)
        return np.maximum(S_base, tail)
    return S

def build_SU_for_trial(r, theta, type="temporal"):
    on, off = get_onset_offset(r['stimd_c'], r['ttype_c'],
                               r['timepoint_1'], r['timepoint_2'],
                               r['timepoint_3'], r['timepoint_4'])
    T = float(r['timepoint_4'])
    N = int(T / dt)
    if N <= 0:
        return None, None, 0
    t = np.linspace(0.0, T, N, endpoint=False)
    S_t = get_S_fn(theta['S_amp'], theta['S_d'], float(on), float(off))(t).astype(np.float64, copy=False)
    if type == "temporal":
        U_t = get_U_fn(theta['U_amp'], theta['U_base'], theta['U_on'], T)(t).astype(np.float64, copy=False)
    else:
        U_t = get_U_spatial_fn(theta['U_amp'], theta['U_base'],
                               r['timepoint_1'], r['timepoint_2'],
                               r['timepoint_3'], r['timepoint_4'])(t).astype(np.float64, copy=False)
    return np.ascontiguousarray(S_t), np.ascontiguousarray(U_t), N

def model_mean_for_trials(trials, theta, type="temporal"):
    pc = []
    for _, r in trials.iterrows():
        side_true = r['x_c']
        code = side_map[side_true]
        S_t, U_t, N = build_SU_for_trial(r, theta, type)
        if N <= 0:
            continue
        mL, mC, mR = simulate_counts_one_trial_heun(
            S_t, U_t, code,
            theta['sL'], theta['sC'], theta['sR'],
            theta['noise'], dt, th[0], th[1], th[2], M
        )
        pc.append((mL if side_true == 'L' else mC if side_true == 'C' else mR) / M)
    if len(pc) == 0:
        return np.nan, np.nan
    pc = np.asarray(pc, float)
    mean = float(pc.mean())
    sem  = float(pc.std(ddof=1) / np.sqrt(len(pc))) if len(pc) > 1 else 0.0
    return mean, sem



def get_plot_path(kind: str, filename: str, model_name: str) -> str:
    base_model_dir = os.path.join(paths.PLOTS, "fitting", model_name)
    kind_dir = os.path.join(base_model_dir, kind)
    os.makedirs(kind_dir, exist_ok=True)
    return os.path.join(kind_dir, filename)


def _plot_delay_or_stim_1d_on_ax(
    ax,
    df,
    subject,
    n_bins,
    which,                 # "delay" o "stim"
    palette_data,          # trunc_purples o trunc_oranges
):
    """
    Pinta en 'ax' la curva 1D binned (Data vs Model) para un subject.
    Devuelve True si ha ploteado algo, False si no hay datos.
    """

    # --- filtros como en tu función ---
    df_delay = df[df["stimd_c"] == "SS"]
    df_stim  = df[df["ttype_c"] == "DS"].copy()

    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim  = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_duration", "correct_bool", "p_model_correct", "subject", "stim_duration"]
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim  = df_stim.dropna(subset=needed_cols)

    # elegir dataset
    if which == "delay":
        d = df_delay
        xcol = "delay_duration"
        xlabel = "Delay duration"
        title_suffix = "Delay"
        band_floor = 1/3
    elif which == "stim":
        d = df_stim
        xcol = "stim_duration"
        xlabel = "Stimulus duration"
        title_suffix = "Stimulus"
        band_floor = 1/3
    else:
        raise ValueError("which must be 'delay' or 'stim'")

    if d.empty:
        # panel vacío pero con título para saber cuál falta
        ax.set_title(f"{subject} - {title_suffix}\n(no data)", fontsize=9)
        ax.axis("off")
        return False

    # --- binning por subject (qcut) ---
    d = d.copy()
    d["x_bin"], edges = pd.qcut(d[xcol], q=n_bins, retbins=True, duplicates="drop")

    centers = (
        d.groupby("x_bin", observed=True)[xcol]
         .median()
         .rename("center")
         .reset_index()
         .sort_values("center")
    )
    order_bins = list(centers["x_bin"])

    subj = (
        d.groupby(["x_bin", "subject"], observed=True)
         .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
         )
         .reset_index()
         .merge(centers, on="x_bin", how="left")
    )

    plot_df = subj.melt(
        id_vars=["x_bin", "subject", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_df["kind"] = plot_df["kind"].map({"data_acc": "Data", "model_acc": "Model"})

    # --- plot ---
    sns.lineplot(
        data=plot_df[plot_df["kind"] == "Model"],
        x="center", y="acc",
        color="gray", linestyle="-",
        errorbar=("ci", 95), err_style="band",
        ax=ax,
    )

    sns.lineplot(
        data=plot_df[plot_df["kind"] == "Data"],
        x="center", y="acc",
        hue="center",
        palette=palette_data,
        marker="o",
        linewidth=0,
        errorbar=("ci", 95), err_style="bars",
        legend=False,
        ax=ax,
        zorder=10,
    )

    ax.axhspan(0, band_floor, color="gray", alpha=0.15, zorder=0)

    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frac. correct responses", fontsize=12)
    ax.set_title(f"{subject}", fontsize=12)
    ax.tick_params(labelsize=12)

    return True


def plot_delay_stim_1d_multipanel_all_subjects(
    df,
    model_name,
    n_bins,
    subjects=None,
    max_cols=5,
    save=True,
):
    """
    Crea 2 multipanel: uno para Delay y otro para Stim, con todos los subjects.
    """
    if subjects is None:
        subjects = sorted(df["subject"].dropna().unique())

    n = len(subjects)
    ncols = min(max_cols, n) if n > 0 else 1
    nrows = math.ceil(n / ncols) if n > 0 else 1
    PANEL = 1.75
    GAP   = 0.75
    fig_width  = ncols * PANEL + (ncols - 1) * GAP
    fig_height = nrows * PANEL + (nrows - 1) * GAP
    fig_d, axes_d = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True)
    axes_d = np.array(axes_d).reshape(-1)

    for i, subj in enumerate(subjects):
        _plot_delay_or_stim_1d_on_ax(
            axes_d[i], df, subj, n_bins,
            which="delay",
            palette_data=trunc_purples,
        )

    # apagar axes sobrantes
    for idx, ax in enumerate(axes_d):
        row = idx // ncols
        col = idx % ncols

        if row != nrows - 1:
            ax.set_xlabel("")
        if col != 0:
            ax.set_ylabel("")
    for j in range(len(subjects), len(axes_d)):
        axes_d[j].axis("off")
    sns.despine(fig=fig_d)
    # fig_d.suptitle(f"{model_name} - Delay 1D (all subjects)", y=1.01, fontsize=14)
    # fig_d.tight_layout()
    fig_d.subplots_adjust(
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    wspace=GAP / PANEL,
    hspace=GAP / PANEL,
    )

    if save:
        fname = f"fig_delay_1d_ALLSUBJ_multipanel.pdf"
        out_path = get_plot_path("no binning", fname, model_name)
        fig_d.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig_d)

    # --- STIM multipanel ---
    fig_s, axes_s = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True)
    axes_s = np.array(axes_s).reshape(-1)

    for i, subj in enumerate(subjects):
        _plot_delay_or_stim_1d_on_ax(
            axes_s[i], df, subj, n_bins,
            which="stim",
            palette_data=trunc_oranges,
        )

     # apagar axes sobrantes
    for idx, ax in enumerate(axes_s):
        row = idx // ncols
        col = idx % ncols
        if row != nrows - 1:
            ax.set_xlabel("")
        if col != 0:
            ax.set_ylabel("")

    for j in range(len(subjects), len(axes_s)):
        axes_s[j].axis("off")
    sns.despine(fig=fig_s)
    # fig_s.suptitle(f"{model_name} - Stimulus 1D (all subjects)", y=1.01, fontsize=14)
    # fig_s.tight_layout()
    fig_s.subplots_adjust(
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    wspace=GAP / PANEL,
    hspace=GAP / PANEL,
    )

    if save:
        fname = f"fig_stim_1d_ALLSUBJ_multipanel.pdf"
        out_path = get_plot_path("no binning", fname, model_name)
        fig_s.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig_s)



def plot_delay_binned_1d_two_models(dfA, dfB, dfC, model_name_A, model_name_B, model_name_C, model_labels, subject=None, n_bins=7,  color_A="tab:blue", color_B="tab:orange", color_C="tab:green", save=True, base_for_bins="A"):
    def _filter_delay_stim(df, subject):
        df_delay = df[df["stimd_c"] == "SS"].copy()
        df_stim  = df[df["ttype_c"] == "DS"].copy()
        if subject is not None:
            df_delay = df_delay[df_delay["subject"] == subject].copy()
            df_stim  = df_stim[df_stim["subject"] == subject].copy()

        needed = ["delay_duration", "stim_duration", "correct_bool", "p_model_correct", "subject"]
        df_delay = df_delay.dropna(subset=needed)
        df_stim  = df_stim.dropna(subset=needed)
        return df_delay, df_stim

    def _compute_edges_and_centers(d_base, xcol, n_bins):
        # qcut devuelve edges (np.ndarray)
        _, edges = pd.qcut(d_base[xcol], q=n_bins, retbins=True, duplicates="drop")
        # asignar bins con esos edges
        bin_ser = pd.cut(d_base[xcol], bins=edges, include_lowest=True)
        centers = (
            d_base.assign(_bin=bin_ser)
                  .groupby("_bin", observed=True)[xcol]
                  .median()
                  .rename("center")
                  .reset_index()
        )
        centers = centers.dropna(subset=["center"]).copy()
        centers["center"] = centers["center"].astype(float)
        order_bins = list(centers.sort_values("center")["_bin"])
        return edges, centers.rename(columns={"_bin": "bin"}), order_bins

    def _agg_data(d, edges, centers, xcol):
        b = pd.cut(d[xcol], bins=edges, include_lowest=True)
        out = (
            d.assign(bin=b)
             .groupby(["bin", "subject"], observed=True)
             .agg(data_acc=("correct_bool", "mean"))
             .reset_index()
             .merge(centers, on="bin", how="left")
        )
        out = out.dropna(subset=["center"]).copy()
        out["center"] = out["center"].astype(float)
        return out

    def _agg_model(d, edges, centers, xcol):
        b = pd.cut(d[xcol], bins=edges, include_lowest=True)
        out = (
            d.assign(bin=b)
             .groupby(["bin", "subject"], observed=True)
             .agg(model_acc=("p_model_correct", "mean"))
             .reset_index()
             .merge(centers, on="bin", how="left")
        )
        out = out.dropna(subset=["center"]).copy()
        out["center"] = out["center"].astype(float)
        return out

    # ---------- preparar ----------
    A_delay, A_stim = _filter_delay_stim(dfA, subject)
    B_delay, B_stim = _filter_delay_stim(dfB, subject)
    C_delay, C_stim = _filter_delay_stim(dfC, subject)

    if A_delay.empty or A_stim.empty:
        print(f"(sin datos válidos en A para {subject})")
        return
    if B_delay.empty or B_stim.empty:
        print(f"(sin datos válidos en B para {subject})")
        return
    if C_delay.empty or C_stim.empty:
        print(f"(sin datos válidos en C para {subject})")
        return

    # base para bins
    if base_for_bins == "A":
        base_delay = A_delay
        base_stim  = A_stim
    else:
        base_delay = pd.concat([A_delay, B_delay, C_delay], ignore_index=True)
        base_stim  = pd.concat([A_stim,  B_stim, C_stim],  ignore_index=True)

    title_subj = subject if subject is not None else "All subjects"

    # ---------- DELAY ----------
    edges_delay, centers_delay, order_bins_delay = _compute_edges_and_centers(
        base_delay, xcol="delay_duration", n_bins=n_bins
    )
    data_delay  = _agg_data(base_delay, edges_delay, centers_delay, xcol="delay_duration")
    modelA_delay = _agg_model(A_delay, edges_delay, centers_delay, xcol="delay_duration")
    modelB_delay = _agg_model(B_delay, edges_delay, centers_delay, xcol="delay_duration")
    modelC_delay = _agg_model(C_delay, edges_delay, centers_delay, xcol="delay_duration")
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.lineplot(
        data=modelA_delay, x="center", y="model_acc",
        color=color_A, errorbar=("ci", 95), err_style="band", label=model_labels[0],
        ax=ax
    )
    sns.lineplot(
        data=modelB_delay, x="center", y="model_acc",
        color=color_B, errorbar=("ci", 95), err_style="band", label=model_labels[1],
        ax=ax
    )
    sns.lineplot(
        data=modelC_delay, x="center", y="model_acc",
        color=color_C, errorbar=("ci", 95), err_style="band", label=model_labels[2],
        ax=ax
    )

    sns.lineplot(
        data=data_delay, x="center", y="data_acc",
        color="purple",
        marker="o", linewidth=0,
        errorbar=("ci", 95), err_style="bars",
        legend=False, ax=ax, zorder=10
    )

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Delay duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    ax.set_title(f"{title_subj} - Delay (1D, {len(order_bins_delay)} bins)")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()

    if save:
        fname = f"fig_delay_1d_{title_subj}_{model_name_A}_vs_{model_name_B}_vs_{model_name_C}.pdf"
        out_path = get_plot_path("no binning", fname, model_name_A)
        fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # ---------- STIM ----------
    edges_stim, centers_stim, order_bins_stim = _compute_edges_and_centers(
        base_stim, xcol="stim_duration", n_bins=n_bins
    )
    data_stim   = _agg_data(base_stim, edges_stim, centers_stim, xcol="stim_duration")
    modelA_stim = _agg_model(A_stim, edges_stim, centers_stim, xcol="stim_duration")
    modelB_stim = _agg_model(B_stim, edges_stim, centers_stim, xcol="stim_duration")
    modelC_stim = _agg_model(C_stim, edges_stim, centers_stim, xcol="stim_duration")

    fig, ax = plt.subplots(figsize=(5, 5))

    sns.lineplot(
        data=modelA_stim, x="center", y="model_acc",
        color=color_A, errorbar=("ci", 95), err_style="band", label=model_labels[0],
        ax=ax
    )
    sns.lineplot(
        data=modelB_stim, x="center", y="model_acc",
        color=color_B, errorbar=("ci", 95), err_style="band", label=model_labels[1],
        ax=ax
    )
    sns.lineplot(
        data=modelC_stim, x="center", y="model_acc",
        color=color_C, errorbar=("ci", 95), err_style="band", label=model_labels[2],
        ax=ax
    )

    sns.lineplot(
        data=data_stim, x="center", y="data_acc",
        color="orange",
        marker="o", linewidth=0,
        errorbar=("ci", 95), err_style="bars",
        legend=False, ax=ax, zorder=10
    )

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Stimulus duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    ax.set_title(f"{title_subj} - Stimulus (1D, {len(order_bins_stim)} bins)")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()

    if save:
        fname = f"fig_stim_1d_{title_subj}_{model_name_A}_vs_{model_name_B}.pdf"
        out_path = get_plot_path("no binning", fname, model_name_A)
        fig.savefig(out_path, dpi=300)
    plt.close(fig)