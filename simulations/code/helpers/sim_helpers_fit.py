# sim_helpers.py
import numpy as np
from math_analysis.potencial import get_expressions
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def make_drift():
    tau = 0.1
    values = {
        "tau": tau,
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
    return drift

def U_t(t, onset=0.5, offset=1.5, duration=-1, baseline = -1,amplitude=2):
    if duration <= 0:
        duration = offset - onset
    return np.where((t < onset) | (t > onset + duration), baseline, baseline + amplitude * (t - onset) / duration)

def U_ext_t(t, onset=0.5, offset=1.5, duration=-1, amplitude=2):
    if duration < 0:
        duration = offset - onset
    return np.where(np.logical_or(t < onset, t > onset + duration), 0, amplitude)

def S_t(t, onset=0.30, offset=2, duration = -1 , amplitude=0.1):
    if duration < 0:
        duration = offset - onset
    return np.where(np.logical_or(t < onset, t > onset + duration), 0, amplitude)
def S_t(t, onset=0.30, offset=2, duration=-1, amplitude=0.1, d=0.5):
    if duration < 0:
        duration = offset - onset
    S_base = np.where((t >= onset) & (t <= onset + duration), amplitude, 0)
    tail_start = onset + duration
    tail = np.where(
        (t>=tail_start)&(t<=tail_start+d),
        amplitude*np.exp(-3*(t-tail_start)/d),
        0
    )       
    return np.maximum(S_base, tail)


def S_t(t, onset=0.30, offset=2, duration=-1, amplitude=0.1, d=0.5):# Linear decrease from amplitude to 0
    if duration < 0:
        duration = offset - onset
    S_base = np.where((t >= onset) & (t <= onset + duration), amplitude, 0)
    tail_start = onset + duration
    tail = np.where(
        (t>=tail_start)&(t<=tail_start+d),
        amplitude*(1-(t-tail_start)/d),  
        0
    )
    return np.maximum(S_base, tail)



def simulate_path(side, S_params, U_int_params, U_ext_params, s_params, drift_fn,
                       noise_amp=0.5, Tmax=2.1, dt=0.1/40):
    x0 = np.array([0, 0])
    N = int(Tmax / dt)
    X = np.empty((N+1,2));  X[0] = x0
    th1 = th2 = th3 = 0

    for i in range(N):
        dW  = np.random.randn(3) * np.sqrt(dt)
        dB1 = (dW[0] - dW[1]) / 2
        dB2 = (dW[0] + dW[1] - 2*dW[2]) / 6
        U_int   = U_t(i*dt, **U_int_params)
        U_ext   =  U_ext_t(i*dt, **U_ext_params)
        S   = S_t(i*dt, **S_params)
        U = U_int + U_ext
        if side=='L':
            iL, iC, iR = U+S, U,   U
        elif side=='C':
            iL, iC, iR = U,   U+S, U
        elif side=='R':
            iL, iC, iR = U,   U,   U+S
        else:
            iL, iC, iR = U, U, U

        X[i+1] = X[i] + drift_fn(X[i], iL, iC, iR, **s_params)*dt + noise_amp*np.array([dB1,dB2])

        r1 =  X[i+1,0] +    X[i+1,1]
        r2 = -X[i+1,0] +    X[i+1,1]
        r3 =      -2 *      X[i+1,1]

    if r1>max(r2,r3,th1): return 'L'
    if r2>max(r1,r3,th2): return 'C'
    if r3>max(r1,r2,th3): return 'R'

    return 'none'
