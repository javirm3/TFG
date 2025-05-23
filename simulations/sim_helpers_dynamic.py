# sim_helpers.py
import numpy as np
from potencial import get_expressions
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def make_drift(F1, F2):
    def drift(X, I_L, I_C, I_R):
        x, y = X
        return np.array([F1(x, y, I_L, I_C, I_R),
                         F2(x, y, I_L, I_C, I_R)])
    return drift

def U_t(t, onset=0.5, offset=1.5, duration=-1, amplitude=2):
    if duration <= 0:
        duration = offset - onset
    return np.where((t < onset) | (t > onset + duration), -1, -1 + amplitude * (t - onset) / duration)

def S_t(t, onset=0.30, offset=2, duration = -1 , amplitude=0.1):
    if duration < 0:
        duration = offset - onset
    return np.where(np.logical_or(t < onset, t > onset + duration), 0, amplitude)


def simulate_path(x0, S_params, U_params, Tmax = 2, dt = 0.1/40, noise_amp = 1, drift = None):
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

def simulate_pair(i, j, k, offset, values, S_params_def, U_params_def, x0, n_trajs, Tmax = 2.1):
    _vals    = {**values, 'sL': k}
    S_params = {**S_params_def, 'offset': offset, 'onset': 0.0}
    _, F1_num, F2_num, _ = get_expressions(_vals,
                                           type="numeric",
                                           substituted_I=False)
    drift = make_drift(F1_num, F2_num)

    wins = {'r1':0,'r2':0,'r3':0,'none':0}
    for _ in range(n_trajs):
        X, w = simulate_path(x0,
                             S_params=S_params,
                             U_params=U_params_def,
                             drift=drift,       # si tu simulate_path acepta drift
                             noise_amp=0.5,
                             Tmax=Tmax)
        wins[w] += 1
    return i, j, wins['r1'], n_trajs


def simulate_pair_side(i, j, side, offset,
                       drift_params,
                       S_params_def, U_params_def,
                       x0, n_trajs, Tmax, dt, noise_amp):
    _, F1n, F2n, _ = get_expressions(
        drift_params, type="numeric", substituted_I=False
    )
    drift = make_drift(F1n, F2n)
    mapping = {'r1':'sL', 'r2':'sC', 'r3':'sR'}
    S_params = {**S_params_def, 'offset':   offset,'onset': 0.0}

    wins = 0
    for _ in range(n_trajs):
        _, winner = simulate_path_side(
            x0, side,
            S_params, U_params_def,
            drift,
            noise_amp=noise_amp,
            Tmax=Tmax,
            dt=dt
        )
        if mapping[winner] == side:
            wins += 1

    return i, j, wins, n_trajs



def simulate_path_side(x0, side, S_params, U_params, drift_fn,
                       noise_amp=0.5, Tmax=2.1, dt=0.1/40):
    N = int(Tmax / dt)
    X = np.empty((N+1,2));  X[0] = x0
    th1 = th2 = th3 = 0.5

    for i in range(N):
        dW  = np.random.randn(3) * np.sqrt(dt)
        dB1 = (dW[0] - dW[1]) / 2
        dB2 = (dW[0] + dW[1] - 2*dW[2]) / 6
        U   = U_t(i*dt, **U_params)
        S   = S_t(i*dt, **S_params)

        # solo el canal `side` recibe S
        if side=='sL':
            iL, iC, iR = U+S, U,   U
        elif side=='sC':
            iL, iC, iR = U,   U+S, U
        else:
            iL, iC, iR = U,   U,   U+S

        X[i+1] = X[i] + drift_fn(X[i], iL, iC, iR)*dt + noise_amp*np.array([dB1,dB2])

        r1 =  X[i+1,0] +    X[i+1,1]
        r2 = -X[i+1,0] +    X[i+1,1]
        r3 =      -2 *      X[i+1,1]

    if r1>max(r2,r3,th1): return X[:i+2], 'r1'
    if r2>max(r1,r3,th2): return X[:i+2], 'r2'
    if r3>max(r1,r2,th3): return X[:i+2], 'r3'

    return X, 'none'
