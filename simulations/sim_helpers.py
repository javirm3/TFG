# sim_helpers.py
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from potencial import get_expressions

# ———————— estímulos y señal U/S ————————
def U_t(t, onset=0.5, offset=1.5, duration=-1, amplitude=2):
    if duration <= 0:
        duration = offset - onset
    return np.where((t < onset) | (t > onset + duration),
                    -1,
                    -1 + amplitude * (t - onset) / duration)

def S_t(t, onset=0.30, offset=2, duration=-1, amplitude=0.1):
    if duration < 0:
        duration = offset - onset
    return np.where((t < onset) | (t > onset + duration),
                    0,
                    amplitude)

# ———————— simulador de una sola trayectoria ————————
def simulate_path(x0,
                  S_params,
                  U_params,
                  drift,
                  noise_amp=1,
                  Tmax=2,
                  dt=0.1/40):
    """
    Euler–Maruyama con drift dinámico:
      - drift(X, US, U1, U2) debe estar disponible.
      - S_params y U_params dict para las funciones S_t/U_t.
    Devuelve (X_truncated, winner_str).
    """
    N = int(Tmax / dt)
    X = np.empty((N+1, 2))
    X[0] = x0

    # umbrales fijos
    th1 = th2 = th3 = 0.5

    for i in range(N):
        dW = np.random.randn(3) * np.sqrt(dt)
        dB1 = (dW[0] - dW[1]) / 2
        dB2 = (dW[0] + dW[1] - 2*dW[2]) / 6

        U = U_t(i*dt, **U_params)
        S = S_t(i*dt, **S_params)

        X[i+1] = X[i] + drift(X[i], U+S, U, U) * dt + noise_amp * np.array([dB1, dB2])

        r1 =  (X[i+1, 0] +    X[i+1, 1])
        r2 = -X[i+1, 0] +    X[i+1, 1]
        r3 =       -2 *      X[i+1, 1]

    if r1> max(r2, r3, th1):
        return X[:i+2], 'r1'
    elif r2 > max(r1, r3, th2):
        return X[:i+2], 'r2'
    elif r3 > max(r1, r2, th3):
        return X[:i+2], 'r3'

    return X, 'none'


def simulate_batch_general(args):
    """
    args = (
      j,
      S_params, U_params,
      init_drift_params,
      drift_factory,
      update_drift_fn,
      x0,
      n_trajs
    )
    """
    (j,
     S_params, U_params,
     drift_params,
     drift_factory,
     update_drift_fn,
     x0, n_trajs, Tmax) = args

    wins = {'r1':0, 'r2':0, 'r3':0, 'none':0}
    if update_drift_fn is None:
            drift_fn = drift_factory(drift_params)

    for _ in range(n_trajs):
        if update_drift_fn is not None:
            drift_fn = drift_factory(drift_params)

        X, w = simulate_path(
            x0,
            S_params=S_params,
            U_params=U_params,
            drift=drift_fn,
            noise_amp=0.5,
            Tmax=Tmax
        )
        wins[w] += 1

        if update_drift_fn is not None:
            drift_params = update_drift_fn(drift_params, X, w)

    return j, wins['r1'], wins['r2'], wins['r3'], wins['none']


def parallel_sweep_general(
    sweep_list,                # lista de valores para el sweep
    target,                    # 'S' o 'U'
    param_key,                 # la clave que cambias en S_params o U_params
    S_params_def,
    U_params_def,
    x0,
    Tmax,
    n_trajs,
    init_drift_params,         # dict inicial para drift_factory
    drift_factory,            # fn(drift_params)->drift_fn
    update_drift_fn=None,     # fn(drift_params, X, winner)->drift_params
    n_workers=None
):
    """
    Devuelve:
      p     : ndarray (4, len(sweep_list))
      p_err : ndarray igual que p, con errores estándar.
    """
    n = len(sweep_list)
    p     = np.zeros((4, n))
    p_err = np.zeros_like(p)

    # construyo jobs
    jobs = []
    for j, val in enumerate(sweep_list):
        if target == 'S':
            S_p = {**S_params_def, param_key: val}
            U_p = {**U_params_def}
        else:
            S_p = {**S_params_def}
            U_p = {**U_params_def, param_key: val}

        jobs.append((
            j,
            S_p, U_p,
            init_drift_params.copy(),
            drift_factory,
            update_drift_fn,
            x0, n_trajs, Tmax
        ))

    # workers por defecto = cores físicos - 1
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    ctx = mp.get_context('fork')

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=ctx) as exe:
        # envío de jobs
        futures = [exe.submit(simulate_batch_general, job) for job in jobs]
        # recogida con progreso
        for f in tqdm(as_completed(futures), total=n, desc=f"Parallel sweep using {n_workers} workers"):
            j, r1, r2, r3, none = f.result()
            total = float(n_trajs)
            p[:, j]     = [r1/total, r2/total, r3/total, none/total]
            p_err[:, j] = np.sqrt(p[:, j] * (1 - p[:, j]) / total)

    return p, p_err

