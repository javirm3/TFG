# sim_helpers.py
import numpy as np
from potencial import get_expressions
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def drift(X, I_L, I_C, I_R):
        x, y = X
        return np.array([F1(x, y, I_L, I_C, I_R),
                         F2(x, y, I_L, I_C, I_R)])

def U_t(t, onset=0.5, offset=1.5, duration=-1, amplitude=2):
    if duration <= 0:
        duration = offset - onset
    return np.where((t < onset) | (t > onset + duration), -1, -1 + amplitude * (t - onset) / duration)

def S_t(t, onset=0.30, offset=2, duration = -1 , amplitude=0.1):
    if duration < 0:
        duration = offset - onset
    return np.where(np.logical_or(t < onset, t > onset + duration), 0, amplitude)


def simulate_path(x0, S_params, U_params, Tmax = 2, dt = 0.1/40, noise_amp = 1):
    N = int(Tmax / dt)
    X = np.empty((N+1, 2))
    X[0] = x0
    for i in range(N):
        x, y = X[i]
        dW = np.random.randn(3) * np.sqrt(dt)

        dB1 = (dW[0] - dW[1]) / 2
        dB2 = (dW[0] + dW[1] - 2*dW[2]) / 6
        U = U_t(i*dt, **U_params)
        S = S_t(i*dt, **S_params)
        
        # Euler–Maruyama
        X[i+1] = X[i] + drift(X[i], U+S, U, U) * dt + noise_amp * np.array([dB1, dB2])

        r1 = (X[i+1, 0] + X[i+1, 1])  # r₁
        r2 = -X[i+1, 0] + X[i+1, 1] # r₂
        r3 = -2*X[i+1, 1]  # r₃
    th1,th2,th3 = 0.5, 0.5, 0.5
    # th1, th2, th3 = compute_thresholds(U+S, U, U)ç
   
    if th1 is None:
        th1 = 0.5
    if th2 is None:
        th2 = 0.5
    if th3 is None:
        th3 = 0.5
    if   (r1>r2 and r1>r3 and r1>th1):
        winner = 'r1'
    elif (r2>r1 and r2>r3 and r2>th2):
        winner = 'r2'
    elif (r3>r1 and r3>r2 and r3>th3):
        winner = 'r3'
    else:
        winner = 'none'
    return X, winner  


def simulate_batch(args):
    """
    args = (j, value, target, param_key,
            S_params_def, U_params_def,
            x0, n_trajs)
    """
    j, value, target, param_key, S_params_def, U_params_def, x0, n_trajs = args

    if target == 'S':
        S_params = {**S_params_def, param_key: value}
        U_params = U_params_def
    else:
        S_params = S_params_def
        U_params = {**U_params_def, param_key: value}

    wins = {'r1':0, 'r2':0, 'r3':0, 'none':0}
    for _ in range(n_trajs):
        X, w = simulate_path(
            x0,
            S_params=S_params,
            U_params=U_params,
            noise_amp=0.5,
            Tmax=2
        )
        wins[w] += 1

    return j, wins['r1'], wins['r2'], wins['r3'], wins['none']


def parallel_sweep(values_list,
                   target,
                   param_key,
                   x0,
                   S_params_def,
                   U_params_def,
                   n_trajs,
                   n_workers=None):
    """
    Corre en paralelo las simulaciones variando `param_key` en S o U.
    Devuelve:
       p     : array (4, len(values_list))
       p_err : igual que p, con errores estándar.
    """
    n = len(values_list)
    p     = np.zeros((4, n))
    p_err = np.zeros_like(p)

    jobs = []
    for j, val in enumerate(values_list):
        jobs.append((j, val, target, param_key,
                     S_params_def, U_params_def,
                     x0, n_trajs))

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 5)

    ctx = mp.get_context('fork')

    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=ctx) as exe:

        futures = [exe.submit(simulate_batch, job) for job in jobs]
        for j, r1, r2, r3, none in tqdm(as_completed(futures),
                                        total=n,
                                        desc="Parallel sweep"):
            total = float(n_trajs)
            p[:, j]     = [r1/total, r2/total, r3/total, none/total]
            p_err[:, j] = np.sqrt(p[:, j] * (1 - p[:, j]) / total)

    return p, p_err
