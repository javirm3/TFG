import pandas as pd
import numpy as np
import sys, os, json, re
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
from numba import set_num_threads, get_num_threads

# justo después de los imports, ANTES de llamar a model_data_mean(...)
set_num_threads(10)        # o el número de cores que quieras
print("Numba threads =", get_num_threads())
def truncate_colormap(cmap_name, minval=0.2, maxval=0.9, n=256):
    """Truncate a colormap to a subset of its range."""
    cmap = cm.get_cmap(cmap_name, n)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

trunc_purples = truncate_colormap('Purples_r', 0, 0.7)  # skip the lightest 30%
trunc_oranges = truncate_colormap('Oranges', 0.3, 1)  # skip the lightest 30%

import seaborn as sns
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import paths 

from helpers.sim_core_numba import simulate_counts_one_trial_heun

# --- parámetros del sujeto ---
# row = params_df.loc[params_df['subject'] == f'{subject}_crn'].iloc[0]
# theta = dict(
#     sL=float(row['sL']), sC=float(row['sC']), sR=float(row['sR']),
#     noise=float(row['noise_amp']),
#     S_amp=float(row['S_amplitude']), S_d=float(row['S_d']),
#     U_amp=float(row['U_int_amplitude']), U_base=float(row['U_int_baseline']),
#     U_on=float(row['U_int_onset']),
# )

# ===== utilidades =====
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
        if D <= 0.0: return u
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

side_map = {'L':0, 'C':1, 'R':2}

def build_SU_for_trial(r, theta, type="temporal"):
    on, off = get_onset_offset(r['stimd_c'], r['ttype_c'],
                               r['timepoint_1'], r['timepoint_2'],
                               r['timepoint_3'], r['timepoint_4'])
    T = float(r['timepoint_4'])
    N = int(T / dt)
    if N <= 0: return None, None, 0
    t = np.linspace(0.0, T, N, endpoint=False)
    S_t = get_S_fn(theta['S_amp'], theta['S_d'], float(on), float(off))(t).astype(np.float64, copy=False)
    if type =="temporal":
        U_t = get_U_fn(theta['U_amp'], theta['U_base'], theta['U_on'], T)(t).astype(np.float64, copy=False)
    else:
        U_t = get_U_spatial_fn(theta['U_amp'], theta['U_base'], r['timepoint_1'], r['timepoint_2'], r['timepoint_3'], r['timepoint_4'])(t).astype(np.float64, copy=False)
    return np.ascontiguousarray(S_t), np.ascontiguousarray(U_t), N

def sem_from_bool(x):
    n = len(x)
    if n <= 1: return 0.0
    p = float(np.mean(x))
    return np.sqrt(p*(1-p)/n)

def model_mean_for_trials(trials, theta, type="temporal"):
    pc = []
    for _, r in trials.iterrows():
        side_true = r['x_c']; code = side_map[side_true]
        S_t, U_t, N = build_SU_for_trial(r, theta, type)
        if N <= 0: 
            continue
        mL, mC, mR = simulate_counts_one_trial_heun(
            S_t, U_t, code,
            theta['sL'], theta['sC'], theta['sR'],
            theta['noise'], dt, th[0], th[1], th[2], M
        )
        # mL, mC, mR = simulate_counts_one_trial_heun_jl(
        #     S_t, U_t, code,
        #     theta, dt, th, M, seed=12345
        # )
        pc.append( (mL if side_true=='L' else mC if side_true=='C' else mR) / M )
    if len(pc) == 0:
        return np.nan, np.nan
    pc = np.asarray(pc, float)
    mean = float(pc.mean())
    sem  = float(pc.std(ddof=1) / np.sqrt(len(pc))) if len(pc) > 1 else 0.0
    return mean, sem


subject = 'A92'

dt = 0.1/40
th = (0.5, 0.5, 0.5)
SUBSAMPLE_PER_BIN = 600
stim_colors = {'SS': '#EF6C00', 'SM': '#FB8C00', 'SL': '#FFB74D'}
stim_order  = ['SS','SM','SL']
delay_types = ['VG', 'DS', 'DM', 'DL', 'SIL']
delay_colors = dict(zip(delay_types, ['#230027', '#5E2A7E', '#9C69A3', '#C698CB', '#EFD9F5']))

delay_duration_colors = {k: delay_colors[k] for k in ['DS', 'DM', 'DL']}
colors = {'delay_duration': delay_duration_colors, 'offset': delay_colors, 'stim_duration': stim_colors}
order = {'delay_duration': ['DS', 'DM', 'DL'], 'offset': delay_types, 'stim_duration': stim_order}

n_bins = 10

M = 300
SUBSAMPLE_PER_BIN = 600
def _nansem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    dd = np.where(n > 1, 1, 0)
    s = np.nanstd(a, axis=axis, ddof=0) 
    s = s * np.sqrt(np.maximum(n, 1) / np.maximum(n - dd, 1))
    return s / np.sqrt(np.maximum(n, 1))

def model_data_mean(to_plot, params_plot, bin_var, n_bins, type):
    
    to_plot['off_bin'], off_edges = pd.qcut(to_plot[bin_var], n_bins, retbins=True, duplicates='drop')
    centros = (to_plot.groupby('off_bin', observed=True)[bin_var]
            .median().rename('center').reset_index().sort_values('center'))
    bin_order = list(centros['off_bin'])
    x_centers = centros['center'].to_numpy()

    y_data, y_data_sem = [], []
    y_model, y_model_sem = [], []


    data_mat, model_mat = [], []
    sil_xs, sil_data, sil_model = [], [], []

    for subject in tqdm(params_plot['subject'].unique()):
        df_subj = to_plot[to_plot['subject'] == subject].copy()

        row = params_plot.loc[params_plot['subject'] == subject].iloc[0]
        theta = dict(
            sL=float(row['sL']), sC=float(row['sC']), sR=float(row['sR']), noise=float(row['noise_amp']), S_amp=float(row['S_amplitude']), S_d=float(row['S_d']), U_amp=float(row['U_int_amplitude']), U_base=float(row['U_int_baseline']), U_on=float(row['U_int_onset']))
        print(f'Processing subject {subject} with params: {theta}')
        subj_data, subj_model = [], []

        for cat in bin_order:
            bin_trials = df_subj[df_subj['off_bin'] == cat]

            if len(bin_trials) > SUBSAMPLE_PER_BIN:
                parts = []
                for s in ['L','C','R']:
                    g = bin_trials[bin_trials['x_c'] == s]
                    k = min(len(g), SUBSAMPLE_PER_BIN//3)
                    if k > 0:
                        parts.append(g.sample(k, random_state=0))
                if len(parts):
                    bin_trials = pd.concat(parts, axis=0)

            if len(bin_trials) == 0:
                subj_data.append(np.nan)
                subj_model.append(np.nan)
            else:
                acc = bin_trials['correct_bool'].to_numpy(dtype=float)
                subj_data.append(float(np.mean(acc)))

                m, _s = model_mean_for_trials(bin_trials, theta)
                subj_model.append(m)

        data_mat.append(subj_data)
        model_mat.append(subj_model)


    data_mat   = np.asarray(data_mat, float)   
    model_mat  = np.asarray(model_mat, float) 

    y_data_mean   = np.nanmean(data_mat, axis=0)
    y_model_mean  = np.nanmean(model_mat, axis=0)
    y_data_sem_s  = _nansem(data_mat, axis=0)
    y_model_sem_s = _nansem(model_mat, axis=0)

    x_plot_mean      = x_centers
    y_data_plot_mean  = y_data_mean
    y_data_sem_mean   = y_data_sem_s
    y_model_plot_mean = y_model_mean
    y_model_sem_mean  = y_model_sem_s

    return (data_mat, model_mat, x_centers)

params_df = pd.read_csv(f'{paths.PARAMS_DIR}/params_evaluated.csv', sep=';')
params_df = params_df[params_df['model'] == 'spatial_reduced_cert']
params_plot = params_df.loc[params_df.groupby("subject")["nll/trial"].idxmin()]
params_plot['U_int_onset']=0.0
params_plot['U_int_baseline'] = -1.0
params_plot['noise_amp'] = 1.0
# params_plot = params_plot[params_plot['subject']=='A92']

df = pd.read_csv(f'{paths.DATA_PATH}/df_filtered.csv')
if 'onset' not in df.columns or 'offset' not in df.columns:
    df[['onset','offset']] = df.apply(
        lambda r: pd.Series(get_onset_offset(r['stimd_c'], r['ttype_c'],
                                             r['timepoint_1'], r['timepoint_2'],
                                             r['timepoint_3'], r['timepoint_4'])),
        axis=1
    )

df['stim_duration']  = df['offset'] - df['onset']
df['delay_duration'] = df['timepoint_4'] - df['offset']
to_plot = df[df['onset']==0]
to_plot2 = df[df['ttype_c']!='VG']
print(f'Number of sessions delay {len(to_plot[["subject", "session"]].drop_duplicates())}')
print(f'Number of trials delay: {len(to_plot["trial"])}')
print(f'Number of sessions stimulus {len(to_plot2[["subject", "session"]].drop_duplicates())}')
print(f'Number of trials delay: {len(to_plot2["trial"])}')
data_stim, model_stim,bins_stim = model_data_mean(to_plot2, params_plot, 'offset', 8, "spatial")
data_delay, model_delay, bins_delay = model_data_mean(to_plot, params_plot, 'delay_duration', 8, "spatial")

rows = []
subjects = list(params_plot['subject'].unique())
for si, subj in enumerate(subjects):
    for bi, x in enumerate(bins_stim):
        d = data_stim[si, bi]
        m = model_stim[si, bi]
        if not np.isnan(d):
            rows.append({'center': float(x), 'acc': float(d), 'kind': 'Data',  'subject': subj})
        if not np.isnan(m):
            rows.append({'center': float(x), 'acc': float(m), 'kind': 'Model', 'subject': subj})

plot_stim= pd.DataFrame(rows)

rows = []
subjects = list(params_plot['subject'].unique())
for si, subj in enumerate(subjects):
    for bi, x in enumerate(bins_delay):
        d = data_delay[si, bi]
        m = model_delay[si, bi]
        if not np.isnan(d):
            rows.append({'center': float(x), 'acc': float(d), 'kind': 'Data',  'subject': subj})
        if not np.isnan(m):
            rows.append({'center': float(x), 'acc': float(m), 'kind': 'Model', 'subject': subj})


plot_delay = pd.DataFrame(rows)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=False, sharey=True)
axes=axes.flatten()

sns.lineplot(x='center',  y='acc',  ax=axes[0], data=plot_delay[plot_delay['kind']=='Data'], ci=False,  
              zorder=1, color='gray',  linestyle='--')
sns.lineplot(x='center',  y='acc',  ax=axes[0], data=plot_delay[plot_delay['kind']=='Model'], ci=False,  
              zorder=1, color='gray')
sns.lineplot(x='center',  y='acc', hue='center',  ax=axes[0], data=plot_delay[plot_delay['kind']=='Model'], errorbar=('ci', 95),  
             err_style="bars", marker='s', zorder=10, palette=trunc_purples, legend=False)
sns.lineplot(x='center',  y='acc', hue='center',  ax=axes[0], data=plot_delay[plot_delay['kind']=='Data'], errorbar=('ci', 95),  
             err_style="bars", marker='o', zorder=10, palette=trunc_purples, legend=False)


sns.lineplot(x='center',  y='acc',  ax=axes[1], data=plot_stim[plot_stim['kind']=='Data'], ci=False,  
              zorder=1, color='gray', linestyle='--')
sns.lineplot(x='center',  y='acc',  ax=axes[1], data=plot_stim[plot_stim['kind']=='Model'], ci=False,  
              zorder=1, color='gray')
sns.lineplot(x='center',  y='acc', hue='center',  ax=axes[1], data=plot_stim[plot_stim['kind']=='Model'], errorbar=('ci', 95),  
             err_style="bars", marker='s', zorder=10, palette=trunc_oranges, legend=False)
sns.lineplot(x='center',  y='acc', hue='center',  ax=axes[1], data=plot_stim[plot_stim['kind']=='Data'], errorbar=('ci', 95),  
             err_style="bars", marker='o', zorder=10, palette=trunc_oranges, legend=False)

# axes    
xmin = -0.4
lines_c = 'silver'
axes[0].set_xlabel('Delay duration (s)')
axes[0].set_ylabel('Frac. correct responses')
# axes[0].set_ylim(0.2, 1.05)
axes[0].text(0.3, 0.05, 'n= '+str(len(subjects)), verticalalignment='bottom', horizontalalignment='right',
        transform=axes[0].transAxes, color='black', fontweight='bold', fontsize = 20)
xmax= plot_delay.center.max()+0.5
axes[0].fill_between(np.arange(xmin, xmax+1), 0.3333, 0.2, facecolor=lines_c, alpha=0.4)
axes[0].set_xlim(xmin, xmax)


from matplotlib.lines import Line2D

legend_delay = [
    Line2D([], [], color=trunc_purples(0.15), marker='o', linestyle='None',
           markersize=7, label='Data'),
    Line2D([], [], color=trunc_purples(0.15), marker='s', linestyle='None',
           markersize=7, label='Model')
]
axes[0].legend(handles=legend_delay, loc='best', frameon=False)
legend_stim = [
    Line2D([], [], color=trunc_oranges(0.3), marker='o', linestyle='None',
           markersize=7, label='Data'),
    Line2D([], [], color=trunc_oranges(0.3), marker='s', linestyle='None',
           markersize=7, label='Model')
]
axes[1].legend(handles=legend_stim, loc='best', frameon=False)

axes[1].set_xlabel('Stimulus duration (s)')
xmax= plot_stim.center.max()+0.5
axes[1].fill_between(np.arange(xmin, xmax+1), 0.3333, 0.2, facecolor=lines_c, alpha=0.4)
axes[1].set_xlim(-0.4, xmax)
sns.set()
sns.set_style('white')
sns.set_style('ticks')
sns.set_context("poster", font_scale=1)
sns.despine()
plt.tight_layout()
plt.savefig('fig_model_vs_data_all_subjects.png', dpi=300)
plt.savefig('fig_model_vs_data_all_subjects.svg')