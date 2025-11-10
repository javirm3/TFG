from numba import njit
import numpy as np

@njit
def drift_numba(X, IL, IC, IR, sL, sC, sR):
    X1, X2 = X[0], X[1]

    term1_F1 = -5.0 * IC + 5.0 * IL
    term2_F1 = 20.0 * X1 * X2
    term3_F1 = -1.9047619047619 * X1 * (X1**2 + 3 * X2**2)
    term4_F1 = 5.0 * X1 * (sC + sL)
    term5_F1 = 10.0 * X1 * (
        0.904761904761905 * IC +
        0.904761904761905 * IL -
        0.0952380952380951 * IR +
        0.226190476190476 * sC +
        0.226190476190476 * sL -
        0.0238095238095238 * sR
    )
    term6_F1 = -10.0 * X2 * (IC - IL + 0.25 * sC - 0.25 * sL)
    term7_F1 = -5.0 * (X2 + 0.25) * (sC - sL)

    F1 = term1_F1 + term2_F1 + term3_F1 + term4_F1 + term5_F1 + term6_F1 + term7_F1

    F2 = (
        -3.33333333333333 * IC * X1 +
         3.09523809523809 * IC * X2 +
         1.66666666666667 * IC +
         3.33333333333333 * IL * X1 +
         3.09523809523809 * IL * X2 +
         1.66666666666667 * IL +
        13.0952380952381 * IR * X2 -
         3.33333333333333 * IR -
         1.9047619047619 * X1**2 * X2 +
         3.33333333333333 * X1**2 -
         10.0 * X2**2 -
         1.9047619047619 * X2 * (X1**2 + 3 * X2**2) +
         2.44047619047619 * X2 * sC +
         2.44047619047619 * X2 * sL +
         9.94047619047619 * X2 * sR +
         0.416666666666667 * sC +
         0.416666666666667 * sL -
         0.833333333333333 * sR
    )

    return np.array([F1, F2])


@njit
def simulate_path_numba(side_code, IL, IC, IR, sL, sC, sR, Tmax, dt=0.1 / 40, noise_amp=0.5):
    x0 = np.array([0.0, 0.0])
    N = int(Tmax / dt)
    X = np.empty((N + 1, 2))
    X[0] = x0
    th1 = th2 = th3 = 0.0

    for i in range(N):
        dW = np.random.randn(3) * np.sqrt(dt)
        dB1 = (dW[0] - dW[1]) / 2
        dB2 = (dW[0] + dW[1] - 2 * dW[2]) / 6

        d = drift_numba(X[i], IL[i], IC[i], IR[i], sL, sC, sR)
        X[i + 1] = X[i] + d * dt + noise_amp * np.array([dB1, dB2])

    r1 = X[-1, 0] + X[-1, 1]
    r2 = -X[-1, 0] + X[-1, 1]
    r3 = -2.0 * X[-1, 1]

    if r1 > max(r2, r3, th1):
        return 0
    elif r2 > max(r1, r3, th2):
        return 1
    elif r3 > max(r1, r2, th3):
        return 2
    else:
        return -1

import numpy as np

def get_U_fn(amplitude, baseline, onset, offset):
    def U(t):
        u = np.zeros_like(t)
        active = (t >= onset) & (t <= offset)
        u[active] = amplitude + baseline
        u[~active] = baseline
        return u
    return U

def get_U_ext_fn(amplitude, onset, offset):
    def U_ext(t):
        u = np.zeros_like(t)
        active = (t >= onset) & (t <= offset)
        u[active] = amplitude
        return u
    return U_ext

def get_S_fn(amplitude, d, onset, offset):
    def S(t):
        duration = offset - onset
        S_base = np.where((t >= onset) & (t <= onset + duration), amplitude, 0)
        tail_start = onset + duration
        tail = np.where(
            (t >= tail_start) & (t <= tail_start + d),
            amplitude * (1 - (t - tail_start) / d),
            0
        )
        return np.maximum(S_base, tail)
    return S

def run_trial_numba(side, S_params, U_int_params, U_ext_params, s_params, Tmax, dt=0.1/40, noise_amp=0.5):
    t_vec = np.linspace(0, Tmax, int(Tmax / dt) + 1)

    U_int = get_U_fn(**U_int_params)(t_vec)
    U_ext = get_U_ext_fn(**U_ext_params)(t_vec)
    S = get_S_fn(**S_params)(t_vec)
    U = U_int + U_ext  # combinación de estímulos internos y externos
    if side == 'L':
        iL, iC, iR  = S + U, U, U
    elif side == 'C':
        iL, iC, iR = U, S + U, U
    elif side == 'R':
        iL, iC, iR = U, U, S + U
    else:
        iL, iC, iR  = U_int,U_int,U_int # control

    side_code = {'L': 0, 'C': 1, 'R': 2, 'SIL': -1, 'none': -1}[side]
    result =  simulate_path_numba(
        side_code,
        iL, iC, iR,
        s_params['sL'], s_params['sC'], s_params['sR'],
        Tmax=Tmax,
        dt=dt,
        noise_amp=noise_amp
    )
    return {0: 'L', 1: 'C', 2: 'R', -1: 'none'}.get(result, 'none')
