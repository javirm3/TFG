import pandas as pd
import numpy as np
import sys, os, json, re
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
from numba import set_num_threads, get_num_threads
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import paths 

from helpers.plots import truncate_colormap, get_onset_offset, get_U_fn, get_U_spatial_fn, get_S_fn, build_SU_for_trial, model_mean_for_trials

import seaborn as sns
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import paths


N_BINS = 8           # nº de bins en x_var
X_VAR = "delay_duration"   # "delay_duration", "offset" o "stim_duration"
MODEL_TYPE = "spatialU_notrandom"     # para tu modelo
SUBSAMPLE_PER_BIN = 600  # nº de trials a submuestrear por bin
side_map = {'L': 0, 'C': 1, 'R': 2}
dt = 0.1 / 40
th = (0.5, 0.5, 0.5)
M = 300

from helpers.sim_core_numba import simulate_counts_one_trial_heun_misses
def model_mean_for_trials(trials, theta, type="temporal"):
    pc = []
    total_misses = 0
    for _, r in trials.iterrows():
        side_true = r['x_c']
        code = side_map[side_true]
        S_t, U_t, N = build_SU_for_trial(r, theta, type)
        if N <= 0:
            continue
        mL, mC, mR, miss = simulate_counts_one_trial_heun_misses(
            S_t, U_t, code,
            theta['sL'], theta['sC'], theta['sR'],
            theta['noise'], dt, th[0], th[1], th[2], M
        )
        total_misses += miss
        pc.append((mL if side_true == 'L' else mC if side_true == 'C' else mR) / M)
    if len(pc) == 0:
        return np.nan, np.nan, 0
    pc = np.asarray(pc, float)
    mean = float(pc.mean())
    sem  = float(pc.std(ddof=1) / np.sqrt(len(pc))) if len(pc) > 1 else 0.0
    return mean, sem, total_misses

# ================== WORKER PARA EL POOL ==================
def model_bin_side_worker(args):
    side, cat, df_subj, theta, subsample, model_type = args

    bin_trials = df_subj[(df_subj['x_bin'] == cat) & (df_subj['x_c'] == side)].copy()

    if len(bin_trials) == 0:
        return side, cat, np.nan, 0.0

    if len(bin_trials) > subsample:
        bin_trials = bin_trials.sample(subsample, random_state=0)

    m, s, miss = model_mean_for_trials(bin_trials, theta, type=model_type)
    return side, cat, m, s, miss

# ================== FUNCIÓN PRINCIPAL POR SUJETO ==================
def compute_side_curves_for_subject(df_subj, theta, x_var, n_bins, subsample, model_type):
    df_subj['x_bin'], x_edges = pd.qcut(df_subj[x_var], q=n_bins, retbins=True, duplicates='drop')

    x_centros = (df_subj.groupby('x_bin', observed=True)[x_var].median().rename('x_center').reset_index().sort_values('x_center'))
    x_centers = x_centros['x_center'].to_numpy()
    bin_order = list(x_centros['x_bin'])

    sides = ['L', 'C', 'R']

    p_data      = np.full((len(sides), len(bin_order)), np.nan, float)
    p_data_sem  = np.zeros_like(p_data)

    for i, side in enumerate(sides):
        for j, cat in enumerate(bin_order):
            bin_trials = df_subj[(df_subj['x_bin'] == cat) &(df_subj['x_c'] == side)].copy()

            if len(bin_trials) == 0:
                continue

            if len(bin_trials) > subsample:
                bin_trials = bin_trials.sample(subsample, random_state=0)

            acc = bin_trials['correct_bool'].to_numpy(dtype=float)
            p_hat = float(acc.mean())
            p_data[i, j] = p_hat

            if len(acc) > 1:
                n = len(acc)
                p_data_sem[i, j] = np.sqrt(p_hat * (1 - p_hat) / n)
            else:
                p_data_sem[i, j] = 0.0

    # --- Modelo en paralelo ---
    p_model     = np.full_like(p_data, np.nan, float)
    p_model_sem = np.zeros_like(p_model)
    miss_model   = np.zeros_like(p_model, dtype=int)

    jobs = []
    for side in sides:
        for cat in bin_order:
            jobs.append((side, cat, df_subj, theta, subsample, model_type))

    side_to_idx = {s: i for i, s in enumerate(sides)}
    cat_to_idx  = {cat: j for j, cat in enumerate(bin_order)}

    n_workers = max(1, mp.cpu_count() - 5)
    print(f"  Lanzando pool con {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = [exe.submit(model_bin_side_worker, job) for job in jobs]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="  Modelo lado×bin", leave=False):
            side, cat, m, s, misses = f.result()
            i = side_to_idx[side]
            j = cat_to_idx[cat]
            p_model[i, j]     = m
            p_model_sem[i, j] = s
            miss_model[i, j]   = misses

    print("\n  Misses del modelo por condición (side, bin):")
    for i, side in enumerate(sides):
        for j, cat in enumerate(bin_order):
            misses = miss_model[i, j]
            if not np.isnan(p_model[i, j]):
                n_trials_bin = len(df_subj[(df_subj['x_bin'] == cat) & (df_subj['x_c'] == side)])

                if n_trials_bin > 0:
                    total_paths = n_trials_bin * M
                    miss_pct = 100.0 * misses / total_paths
                else:
                    miss_pct = np.nan

                print(f"    side={side}, bin={j}:  {miss_pct:5.1f}% misses   (n={n_trials_bin})")

    # ==== Misses agregados por lado ====
    print("\n  Misses totales por lado:")
    for i, side in enumerate(sides):
        misses = miss_model[i].sum()
        # número total de trials del lado
        n_trials_side = len(df_subj[df_subj['x_c'] == side])
        total_paths_side = n_trials_side * M
        if total_paths_side > 0:
            miss_pct_side = 100.0 * misses / total_paths_side
        else:
            miss_pct_side = np.nan

        print(f"    {side}: {miss_pct_side:5.1f}% misses   (n={n_trials_side})\n")

    return x_centers, p_data, p_data_sem, p_model, p_model_sem

def plot_side_curves(subject, x_centers, theta,
                     p_data, p_data_sem, p_model, p_model_sem,
                     x_var):
    sides = ['L', 'C', 'R']
    colors = ['#e41a1c', '#4daf4a', '#377eb8']
    side_labels = [
        fr'Left ($s_L$ = {theta["sL"]:.3f})',
        fr'Center ($s_C$ = {theta["sC"]:.3f})',
        fr'Right ($s_R$ = {theta["sR"]:.3f})'
    ]

    plt.figure(figsize=(7, 5))

    for i, (side, col, lab) in enumerate(zip(sides, colors, side_labels)):
        # Modelo
        plt.errorbar(x_centers, p_model[i], yerr=p_model_sem[i], fmt='-o', color=col, capsize=0, markersize=5, elinewidth=2, label=f'Model {lab}')
        # Datos
        plt.errorbar(x_centers, p_data[i], yerr=p_data_sem[i], fmt='--o', color=col, capsize=0, markersize=5, elinewidth=1.5, label=f'Data {lab}')

    plt.axhspan(0, 1/3, color='gray', alpha=0.1, zorder=0)

    xlabel_map = {
        'offset':        'Stimulus offset (s)',
        'stim_duration': 'Stimulus duration (s)',
        'delay_duration':'Delay duration (s)'
    }
    plt.xlabel(xlabel_map.get(x_var, x_var))
    plt.ylabel('Accuracy  p(correct)')
    plt.ylim(0.2, 1.05)
    sns.despine()
    plt.legend(ncol=2, fontsize=8)
    plt.title(f'{subject}: accuracy vs {x_var} by side')
    plt.tight_layout()

    fname = f'fig_side_{x_var}_{subject}.png'
    plt.savefig(fname, dpi=300)
    print(f'  Figure saved to {fname}')
    plt.close()

# ================== MAIN ==================
if __name__ == "__main__":

    # --- Cargar parámetros ---
    params_df=pd.read_csv(f'{paths.PARAMS_DIR}/params_best_models.csv',sep=';')
    # params_df = params_df[params_df['model'] == MODEL_TYPE]
    params_plot = params_df
    # Nos quedamos con el mejor fit por sujeto
    # params_df  = params_df[params_df['subject']== 'A92']
    params_plot = params_df.loc[params_df.groupby("subject")["nll/trial"].idxmin()]

    # --- Cargar datos de trials ---
    df = pd.read_csv(f'{paths.DATA_PATH}/df_filtered.csv')

    if 'onset' not in df.columns or 'offset' not in df.columns:
        df[['onset', 'offset']] = df.apply(
            lambda r: pd.Series(get_onset_offset(r['stimd_c'], r['ttype_c'],
                                                 r['timepoint_1'], r['timepoint_2'],
                                                 r['timepoint_3'], r['timepoint_4'])),
            axis=1
        )

    df['stim_duration']  = df['offset'] - df['onset']
    df['delay_duration'] = df['timepoint_4'] - df['offset']

    # Filtramos como en tu script de delay (onset==0)
    to_plot = df[df['onset'] == 0].copy()

    subjects = list(params_plot['subject'].unique())
    print("Subjects:", subjects)

    sns.set()
    sns.set_style('white')
    sns.set_style('ticks')
    sns.set_context("talk", font_scale=1)

    for subject in subjects:
        print(f'\nProcesando sujeto {subject}')
        row = params_plot.loc[params_plot['subject'] == subject].iloc[0]
        if 'spatial' in str(row['model']):
            model_type= 'spatial'
        else:
            model_type = 'temporal'
        # Defaults por si hay NaNs
        if pd.isna(row['U_int_baseline']):
            row['U_int_baseline'] = -1.0
        if pd.isna(row['U_int_onset']):
            row['U_int_onset'] = 0.0
        if pd.isna(row['noise_amp']):
            row['noise_amp'] = 1.0
        if pd.isna(row['U_ext_amplitude']):
            row['U_ext_amplitude'] = 0.0

        theta = dict(
            sL=float(row['sL']),
            sC=float(row['sC']),
            sR=float(row['sR']),
            noise=float(row['noise_amp']),
            S_amp=float(row['S_amplitude']),
            S_d=float(row['S_d']),
            U_amp=float(row['U_int_amplitude']),
            U_base=float(row['U_int_baseline']),
            U_on=float(row['U_int_onset']),
            U_ext_amp=float(row['U_ext_amplitude'])
        )

        print("  theta =", theta)

        df_subj = to_plot[to_plot['subject'] == subject].copy()
        if df_subj.empty:
            print("  (sin trials para este sujeto en to_plot, se salta)")
            continue

        x_centers, p_data, p_data_sem, p_model, p_model_sem = compute_side_curves_for_subject( df_subj, theta, X_VAR, N_BINS, SUBSAMPLE_PER_BIN, model_type)

        plot_side_curves(subject, x_centers, theta, p_data, p_data_sem, p_model, p_model_sem, X_VAR)