import pandas as pd
import numpy as np
from pybads import BADS
from potencial import get_expressions
from sim_helpers_fit import simulate_path, make_drift
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from sim_helpers_fit_numba import run_trial_numba as simulate_path_nb
from send_not import notify_telegram
import pickle

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)




# --- Función auxiliar ---
def get_onset_offset(stim_dur, delay_dur, timepoint_1, timepoint_2, timepoint_3, timepoint_4):
    if stim_dur == 'VG':
        onset = 0
        offset = timepoint_4
    elif stim_dur == 'SS':
        if delay_dur == 'DS':
            onset = timepoint_2
            offset = timepoint_3
        elif delay_dur == 'DM':
            onset = timepoint_1
            offset = timepoint_2
        elif delay_dur == 'DL':
            onset = 0
            offset = timepoint_1
    elif stim_dur == 'SM':
        if delay_dur == 'DS':
            onset = timepoint_1
            offset = timepoint_3
        elif delay_dur == 'DM':
            onset = 0
            offset = timepoint_2
    elif stim_dur == 'SL':
        onset = 0
        offset = timepoint_3
    elif stim_dur == 'SIL':
        onset = 0
        offset = 0
    else:
        raise ValueError(f"stim_dur desconocido: {stim_dur}")
    return onset, offset

# --- Simulación paralelizable por fila ---
values = {
        "tau": 0.1,
        "c": 1,
        "g": 1,
        "s0": 1,
        "IL": 1/3,
        "IC": 1,
        "IR": 1,
        "I_I": 1/3,
        "sL": 0,
        "sC": 0,
        "sR": 0,
    }
_, F1, F2, _ = get_expressions(values, type = "numeric", substituted_I = False, substituted_S= False)
def drift(X, I_L, I_C, I_R, sL, sC, sR):
    x, y = X
    return np.array([F1(x, y, I_L, I_C, I_R, sL, sC, sR),
                        F2(x, y, I_L, I_C, I_R, sL, sC, sR)])


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
x0 = np.array([
    0.5,   # sL
    0.5,   # sC
    0.5,   # sR
    0.75,  # sigma (ruido mínimo razonable)
    0.1,   # A_stim
    0.5,   # Delta_stim
    2.0,   # A_ui
    -1.0,   # B_ui
    0.0,   # t_on
    0.0    # A_u_ext
])

# x0 = np.array([
#     -0.0830118656,  # sL
#     -0.4959077240,  # sC
#     -0.0971766114,  # sR
#      0.0412979053,  # sigma
#      1.1493691300,  # A_stim
#      0.8829399820,  # Delta_stim
#      1.8544063600,  # A_ui
#     -0.5748256440,  # B_ui
#      0.1083602340,  # t_on
#      0.0003704977   # A_u_ext
# ])

# Global params para evitar pasar objetos no serializables
global_params = None
global_drift = None

def init_worker(params, drift):
    global global_params
    global global_drift
    global_params = params
    global_drift = drift

def simulate_row_pure(args):
    # Desempaquetar fila en formato primitivo (sin pandas)
    (
        stim_dur, delay_dur, timepoint_1, timepoint_2, timepoint_3, timepoint_4,
        x_c, r_c, ttype_c
    ) = args

    onset, offset = get_onset_offset(stim_dur, delay_dur, timepoint_1, timepoint_2, timepoint_3, timepoint_4)

    U_int_params = {
        'offset': timepoint_4,
        'amplitude': global_params[6],
        'baseline': global_params[7],
        'onset': global_params[8],
    }

    S_params = {
        'offset': offset,
        'onset': onset,
        'd': global_params[5],
        'amplitude': global_params[4],
    }

    U_ext_params = {
        'offset': timepoint_4,
        'onset': onset,
        'amplitude': global_params[9],
    }

    s_params = {
        'sL': global_params[0],
        'sC': global_params[1],
        'sR': global_params[2],
    }

    side = x_c if ttype_c != 'SIL' else 'SIL'

    sim_choice = simulate_path_nb(
        side=side, 
        S_params=S_params,
        U_int_params=U_int_params,
        U_ext_params=U_ext_params,
        s_params=s_params,
        noise_amp=global_params[3],
        Tmax=timepoint_4,
        dt=0.1 / 40
    )

    return (1 if sim_choice == r_c else 0)


def objective_function(params):
    try:
        rows = [
            (
                row.stimd_c, row.ttype_c, row.timepoint_1, row.timepoint_2, row.timepoint_3,
                row.timepoint_4, row.x_c, row.r_c, row.ttype_c
            )
            for row in df.itertuples()
        ]

        with ProcessPoolExecutor(max_workers=8, initializer=init_worker, initargs=(params, drift)) as executor:
            errors = list(executor.map(simulate_row_pure, rows))
        sse = sum(errors)
        obj = 1 - (sse / len(rows))

        # --- Protección contra valores peligrosos ---
        if not np.isfinite(obj):
            print("⚠️  Valor no finito detectado. Penalizando...")
            return 1.0  # Penalización máxima

        if obj == 0 or obj == 1.0:
            obj = obj - 1e-6 if obj == 1.0 else obj + 1e-6  # Quitar exactitud

        print(f"[Eval] SSE={sse}, Objective={obj:.6f}")
        return obj

    except Exception as e:
        print(f"❌ Error durante la simulación: {e}")
        return 1.0  # Penaliza si algo falla


# --- Test: solo para verificar que funciona ---
if __name__ == '__main__':
    # --- Carga de datos y filtrado ---
    df = pd.read_csv('./datasets/df_filtered.csv', sep=';')
    df = df[df['subject'].isin(['A92'])]
    df = df[df['timepoint_4'] <= 5]
    print('Done loading and filtering ata')
    print("Iniciando optimización...")
    # Valor inicial
    x0 = np.array([0.5, 0.5, 0.5, 0.75, 0.1, 0.5, 2.0, -1.0, 0.0, 0.0])

    lb = np.array([-1.0, -1.0, -1.0, 0.01, 0.0, 0.0, 0.0, -5, 0.0, 0.0])
    ub = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1, 10.0, 2, 2.0, 10.0])

    plb = np.array([-0.75, -0.75, -0.75, 0.01, 1, 0.1, 1, -3, 0.0, 0])
    pub = np.array([0.75, 0.75, 0.75, 1, 1.25, 0.8, 5, 0, 0.5, 5])

    options = {
        "uncertainty_handling": True,
        "max_fun_evals": 5000,
    }

    bads = BADS(
        fun=objective_function,
        x0=x0,
        lower_bounds=lb,
        upper_bounds=ub,
        plausible_lower_bounds=plb,
        plausible_upper_bounds=pub,
        options=options
    )

    # Ejecutar la optimización
    result = bads.optimize()

    subject = 'A92' 
    with open(f"result_{subject}.pkl", "wb") as f:
                pickle.dump(result, f)
    print(f"✅ Subject {subject} terminado")
    notify_telegram(f"✅ Subject {subject} terminado. Tiempo total: {result.total_time:.1f} s")

    # Mostrar resultados
    print("\n--- Resultado de la optimización ---")
    print("Parámetros óptimos:", result.x)
    print("Valor óptimo de la función objetivo:", result.fval)