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

trunc_purples = truncate_colormap('Purples_r', 0, 0.7)

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