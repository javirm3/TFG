import os, time, warnings, pickle
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from numba import set_num_threads
from pybads import BADS
from send_not import notify_telegram
from sim_core_numba import nll_trials_numba

# ========= CONFIG =========
DT = 0.1 / 40.0
M_REPS = 200                 # prueba rápida; sube luego a 500–1000
N_THREADS = int(os.environ.get("NUMBA_NUM_THREADS", "8"))
ALPHA = 0.5                  # Jeffreys prior
STORAGE_PATH = "./datasets"
CSV_PATH = f"{STORAGE_PATH}/df_filtered.csv"
SUBJECTS = ['A89']

# Columnas esperadas
STIM_COL  = 'stimd_c'
DELAY_COL = 'ttype_c'
SIDE_COL  = 'x_c'
RESP_COL  = 'r_c'

# ========= Fit por sujeto =========
def fit_subject(df_all, subject, save_suffix="nll_heun_trial"):
    # Filtrado
    df = df_all[(df_all['subject'] == subject) & (df_all['timepoint_4'] <= 5.0)].copy()
    if df.empty:
        print(f"⚠️ No hay datos para subject {subject}.")
        return None

    # Codificación arrays compactos
    stim_map  = {'VG':0,'SS':1,'SM':2,'SL':3,'SIL':4}
    delay_map = {'DS':0,'DM':1,'DL':2}
    side_map  = {'L':0,'C':1,'R':2,'SIL':3}
    resp_map  = {'L':0,'C':1,'R':2}

    stimd  = df[STIM_COL].map(stim_map).to_numpy(dtype=np.int8)
    delayd = df[DELAY_COL].map(delay_map).to_numpy(dtype=np.int8)
    side   = df[SIDE_COL].map(side_map).to_numpy(dtype=np.int8)
    resp   = df[RESP_COL].map(resp_map).to_numpy(dtype=np.int8)

    t1 = df['timepoint_1'].to_numpy(dtype=np.float32)
    t2 = df['timepoint_2'].to_numpy(dtype=np.float32)
    t3 = df['timepoint_3'].to_numpy(dtype=np.float32)
    t4 = df['timepoint_4'].to_numpy(dtype=np.float32)

    # Diagnóstico
    dist = (df[RESP_COL].value_counts() / len(df) * 100.0).round(2).to_dict()
    print(f"Subject {subject} — distribución de {RESP_COL}: {dist}")

    # Límites
    lb  = np.array([-1.0, -1.0, -1.0, 0.01, 0.0, 0.0, 0.0, -5.0, 0.0])
    ub  = np.array([ 2.0,  2.0,  2.0, 2.00, 2.0, 1.0, 10.0,  2.0, 2.0])
    plb = np.array([-0.25, -0.25, -0.25, 0.5, 0.01, 0.1, 1.0, -1.25, 0.01])
    pub = np.array([ 0.25,  0.25,  0.25, 1.0, 0.50, 0.8, 3.0, -0.75, 0.50])

    set_num_threads(N_THREADS)
    print(f"Numba threads: {N_THREADS}")

    # Objetivo
    def objective(theta):
        return nll_trials_numba(stimd, delayd, side, resp,
                            t1.astype(np.float64), t2.astype(np.float64),
                            t3.astype(np.float64), t4.astype(np.float64),
                            theta.astype(np.float64),
                            M_REPS, DT, ALPHA, 0.5, 0.5, 0.5)

    # Notificación inicio
    msg_start = f"🚀 Empezando optimización (Heun) para subject {subject}"
    print(msg_start)
    try:
        notify_telegram(msg_start)
    except Exception as e:
        print(f"(Aviso) No se pudo enviar telegram: {e}")

    t0 = time.time()
    bads = BADS(
        fun=objective,
        lower_bounds=lb,
        upper_bounds=ub,
        plausible_lower_bounds=plb,
        plausible_upper_bounds=pub,
        options={
            "uncertainty_handling": True,
            "max_fun_evals": 1000,  # baja para pruebas rápidas
        }
    )
    result = bads.optimize()
    t1 = time.time()
    elapsed = t1 - t0

    # Guardar
    out_name = f"result_{subject}_{save_suffix}.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(result, f)

    # Salida
    print("\n--- Resultado ---")
    print("Parámetros óptimos:", result.x)
    print("NLL óptimo:", result.fval)
    print(f"Guardado en {out_name}")

    # Notificación fin
    msg_end = f"✅ Subject {subject} terminado (Heun). Tiempo total: {elapsed:.1f} s"
    print(msg_end)
    try:
        notify_telegram(msg_end)
    except Exception as e:
        print(f"(Aviso) No se pudo enviar telegram: {e}")

    return result


# ========= MAIN =========
if __name__ == "__main__":
    df_all = pd.read_csv(CSV_PATH, sep=';')
    for subj in SUBJECTS:
        fit_subject(df_all, subj, save_suffix="nll_heun_trial")