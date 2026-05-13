#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib, sys, os
# ==== PATHS DEL PROYECTO ====
base_path = pathlib.Path().resolve().parents[1]
PROJECT_ROOT = pathlib.Path("../").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import paths

set2 = sns.color_palette("Set1")
COLORS = {
    "L": set2[0],  # rojo
    "C": set2[2],  # verde
    "R": set2[1],  # azul
}

U_color = "#9c69a2"

def get_plot_path(kind: str, filename: str, model_name: str) -> str:
    base_model_dir = os.path.join(paths.PLOTS, "fitting", model_name)
    kind_dir = os.path.join(base_model_dir, kind)
    os.makedirs(kind_dir, exist_ok=True)
    return os.path.join(kind_dir, filename)


def onset_offset_from_codes(stim, delay, t1, t2, t3, t4):
    # stim: 0 VG,1 SS,2 SM,3 SL,4 SIL ; delay: 0 DS,1 DM,2 DL
    if stim == 0:
        return 0.0, t4
    elif stim == 1:
        return (t2, t3) if delay == 0 else ((t1, t2) if delay == 1 else (0.0, t1))
    elif stim == 2:
        return (t1, t3) if delay == 0 else (0.0, t2)
    elif stim == 3:
        return 0.0, t3
    else:
        return 0.0, 0.0

def S_value(t, amp, d, onset, offset):
    if t < onset:
        return 0.0
    elif t <= offset:
        return amp
    else:
        tail_end = offset + d
        if (d > 0.0) and (abs(offset - onset) >= 1e-5) and (t <= tail_end):
            return amp * (1.0 - (t - offset) / d)
        return 0.0

def U_spatial_value(t, U_amp, U_base, t1, t2, t3, t4, w1, w2, w3, w4):
    r1 = np.clip(t * w1, 0.0, 1.0)
    r2 = np.clip((t - t1) * w2, 0.0, 1.0)
    r3 = np.clip((t - t2) * w3, 0.0, 1.0)
    r4 = np.clip((t - t3) * w4, 0.0, 1.0)
    return (0.25 * U_amp) * (r1 + r2 + r3 + r4) + U_base


# ====== Mapeos (ajusta si tus códigos difieren) ======
stim_map  = {'VG':0,'SS':1,'SM':2,'SL':3,'SIL':4}
side_map  = {'L':0,'C':1,'R':2,'SIL':3}
resp_map  = {'L':0,'C':1,'R':2}
delay_map = {'DS':0,'DM':1,'DL':2}


def _row_codes(row):
    """Devuelve stim_code, delay_code (Int8) desde columnas string."""
    stim_s = str(row["stimd_c"]).strip().upper()
    delay_s = str(row.get("ttype_c", "DS")).strip().upper()

    stim = np.int8(stim_map[stim_s])
    # delay solo importa en SS/SM; en el resto ponemos 0 (DS)
    if stim_s in ("SS", "SM"):
        delay = np.int8(delay_map[delay_s])
    else:
        delay = np.int8(0)
    return stim, delay


def _pick_trace_from_row(row, trace_col=None):
    """
    Si trace_col se pasa, usa esa columna.
    Si no, intenta usar la 'winning' definida por r_c eligiendo trace_L/C/R.
    """
    if trace_col is not None:
        tr = row.get(trace_col, None)
        if tr is None:
            raise KeyError(f"No existe la columna {trace_col} en el DF.")
        return np.asarray(tr, dtype=np.float32), trace_col

    # inferir por r_c
    rc = str(row.get("r_c", "")).strip().upper()
    if rc == "L":
        return np.asarray(row["trace_L"], dtype=np.float32), "trace_L"
    if rc == "C":
        return np.asarray(row["trace_C"], dtype=np.float32), "trace_C"
    if rc == "R":
        return np.asarray(row["trace_R"], dtype=np.float32), "trace_R"

    raise ValueError("No puedo inferir la traza: pasa trace_col='trace_L'/'trace_C'/'trace_R' o asegúrate de que r_c sea L/C/R.")


def _theta_from_params(params_df, subject, model_name="spatial_reduced3"):
    """
    Coge la fila con menor nll para ese subject y modelo.
    Devuelve parámetros en el orden canónico (10), como en tu pipeline.
    """
    CANONICAL = [
        "sL", "sC", "sR", "noise_amp", "S_amplitude", "S_d",
        "U_int_amplitude", "U_int_baseline", "U_int_onset", "U_ext_amplitude"
    ]
    TEMPLATE = np.array([0, 0, 0, 1, 0, 0, 0, -1, 0, 0], dtype=float)

    df = params_df[(params_df["subject"] == subject) & (params_df["model"] == model_name)].copy()
    if df.empty:
        raise ValueError(f"No hay params para subject={subject} y model={model_name} en params_best_models.csv")

    # intenta usar nll_eval si está; si no, usa nll
    nll_col = "nll_eval" if "nll_eval" in df.columns else ("nll" if "nll" in df.columns else None)
    if nll_col is None:
        # si no hay nll, cogemos primera fila
        row = df.iloc[0]
    else:
        row = df.loc[df[nll_col].astype(float).idxmin()]

    theta = TEMPLATE.copy()
    name_to_idx = {n:i for i,n in enumerate(CANONICAL)}
    for n in CANONICAL:
        if n in row.index and pd.notna(row[n]):
            theta[name_to_idx[n]] = float(row[n])
    return theta


def plot_single_trial_trace_with_SU(df_traces, params_df, subject, trial_selector, delayd = None, stimd = None,
                                    model_name="spatial_reduced3",
                                    dt=0.1/40.0,
                                    trace_col=None,
                                    show_onset_offset=True,
                                    savepath=None):
    """
    trial_selector: puede ser
      - int (índice posicional dentro del df del subject, tras reset_index)
      - dict con filtros (ej. {"trial": 123} o {"session":..., "trial_in_session":...})
    """
    df_sub = df_traces[df_traces["subject"] == subject].copy()
    if df_sub.empty:
        raise ValueError(f"No hay filas para subject={subject} en df_traces")

    if delayd is not None:
        df_sub = df_sub[df_sub["ttype_c"] == delayd]

    if stimd is not None:
        df_sub = df_sub[df_sub["stimd_c"] == stimd]
    df_sub = df_sub.reset_index(drop=True)


    if isinstance(trial_selector, int):
        row = df_sub.iloc[trial_selector]
    elif isinstance(trial_selector, dict):
        m = np.ones(len(df_sub), dtype=bool)
        for k, v in trial_selector.items():
            m &= (df_sub[k] == v)
        if not m.any():
            raise ValueError(f"No hay filas que cumplan {trial_selector}")
        row = df_sub[m].iloc[0]
    else:
        raise TypeError("trial_selector debe ser int o dict.")

    theta = _theta_from_params(params_df, subject, model_name=model_name)

    # params
    sL, sC, sR = theta[0], theta[1], theta[2]
    noise_amp  = theta[3]
    S_amp, dS  = theta[4], theta[5]
    U_amp, Ubase = theta[6], theta[7]
    U_on = theta[8]          # quizá no lo uses en spatial
    U_ext_amp = theta[9]     # quizá 0 en reduced3

    # timepoints del trial
    t1 = float(row["timepoint_1"])
    t2 = float(row["timepoint_2"])
    t3 = float(row["timepoint_3"])
    t4 = float(row["timepoint_4"])

    stim_code, delay_code = _row_codes(row)
    onset, offset = onset_offset_from_codes(np.int8(stim_code), np.int8(delay_code), t1, t2, t3, t4)
    
    stim_side = str(row.get("x_c", "")).strip().upper()
    stim_color = COLORS.get(stim_side, "gray")

    N = int(np.floor(t4 / dt))
    t = np.arange(N, dtype=float) * dt

    
    trL = np.asarray(row["trace_L"], dtype=np.float32)
    trC = np.asarray(row["trace_C"], dtype=np.float32)
    trR = np.asarray(row["trace_R"], dtype=np.float32)

    def _pad_to_N(tr, N):
        return tr[:N] if len(tr) >= N else np.pad(tr, (0, N - len(tr)), constant_values=np.nan)

    trL = _pad_to_N(trL, N)
    trC = _pad_to_N(trC, N)
    trR = _pad_to_N(trR, N)

    # S(t)
    S = np.array([S_value(tt, S_amp, dS, onset, offset) for tt in t], dtype=float)

    # U(t) (spatial)
    def _inv(x): return 1.0/x if abs(x) > 1e-12 else 0.0
    w1 = _inv(t1 - 0.0)
    w2 = _inv(t2 - t1)
    w3 = _inv(t3 - t2)
    w4 = _inv(t4 - t3)

    U = np.array([
        U_spatial_value(float(tt), U_amp, Ubase, t1, t2, t3, t4, w1, w2, w3, w4)
        for tt in t
    ], dtype=float)

    # ---- PLOT 2 PANELES ----
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(5, 3), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]}
    )
    
    # arriba: 3 trazas
    ax0.plot(t, trL, lw=1.2, color=COLORS["L"], label="L")
    ax0.plot(t, trC, lw=1.2, color=COLORS["C"], label="C")
    ax0.plot(t, trR, lw=1.2, color=COLORS["R"], label="R")
    
    ax0.set_ylabel("Population rate")
    ax0.legend(frameon=False, fontsize=9)
    # title = f"{subject} | {model_name} | {tr_name} | stim={row['stimd_c']} delay={row.get('ttype_c','')} | r_c={row.get('r_c','')}"
    # ax0.set_title(title, fontsize=10)

    # abajo: S y U
    ax1.plot(t, S, lw=2, color=stim_color, label="S(t)")
    ax1.plot(t, U, lw=2, color=U_color, label="U(t)")
    ax1.set_ylabel("Input")
    ax1.set_xlabel("Time (s)")
    ax1.legend(frameon=False, fontsize=9)

    # vlines timepoints
    for ax in (ax0, ax1):
        for x, lab in [(t1,"t1"), (t2,"t2"), (t3,"t3"), (t4,"t4")]:
            ax.axvline(x, ls="--", lw=1, alpha=0.6)
        if show_onset_offset:
            ax.axvline(onset,  ls=":", lw=1.2, alpha=0.8)
            ax.axvline(offset, ls=":", lw=1.2, alpha=0.8)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# =======================
# EJEMPLO DE USO
# =======================
if __name__ == "__main__":
    MODEL_NAME = "spatial_reduced3"
    traces_path =os.path.join(paths.PARAMS_DIR, f"df_traces_{MODEL_NAME}.parquet")
    params_path = os.path.join(paths.PARAMS_DIR, f"params_evaluated.csv")

    df_traces = pd.read_parquet(traces_path)
    params_df = pd.read_csv(params_path, sep =";")


    # 1) por índice dentro del sujeto (0 = primera fila de ese subject)
    plot_single_trial_trace_with_SU(
        df_traces, params_df,
        delayd="DS",
        stimd ="SS",
        subject="A83",
        trial_selector=0,
        model_name=MODEL_NAME,
        dt=0.1/40.0,
        trace_col=None,
        savepath=get_plot_path("single_traces", "single_trial_trace_A83_idx0.pdf", MODEL_NAME)
    )

    # 2) o por filtro (si tienes columnas tipo "trial" / "session" etc.)
    # plot_single_trial_trace_with_SU(
    #     df_traces, params_df,
    #     subject="A83",
    #     trial_selector={"trial": 123},
    #     model_name="spatial_reduced3",
    #     dt=0.1/40.0,
    #     trace_col="trace_L",
    #     savepath="single_trial_trace.pdf"
    # )