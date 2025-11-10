from numba import njit
import numpy as np

@njit
def drift(X, IL, IC, IR, sL, sC, sR):
    X1, X2 = X[0], X[1]

    # F1 terms
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

    # F2
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
def simulate_path(side_code, IL, IC, IR, sL, sC, sR, Tmax, dt=0.1 / 40, noise_amp=0.5):
    x0 = np.array([0.0, 0.0])
    N = int(Tmax / dt)
    X = np.empty((N + 1, 2))
    X[0] = x0
    th1 = th2 = th3 = 0.0

    for i in range(N):
        dW = np.random.randn(3) * np.sqrt(dt)
        dB1 = (dW[0] - dW[1]) / 2
        dB2 = (dW[0] + dW[1] - 2 * dW[2]) / 6

        # estímulos ya combinados
        drift = drift(X[i], IL[i], IC[i], IR[i], sL, sC, sR)
        X[i + 1] = X[i] + drift * dt + noise_amp * np.array([dB1, dB2])

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

def run_trial(side: str, Tmax: float, U_t, U_ext_t, S_t, dt=0.1 / 40):
    t_vec = np.linspace(0, Tmax, int(Tmax / dt) + 1)

    # Estímulos internos y externos
    U = U_t(t_vec)
    U_ext = U_ext_t(t_vec)
    S = S_t(t_vec)

    # Combina estímulos según la condición (e.g., 'L', 'C', 'R')
    if side == 'L':
        IL = S + U_ext
        sC = S
        sR = S
    elif side == 'C':
        sL = S
        sC = S + U_ext
        sR = S
    elif side == 'R':
        sL = S
        sC = S
        sR = S + U_ext
    else:
        sL = sC = sR = S  # control or 'none'

    # Ejecuta simulación con U y S combinados
    return simulate_path(
        {'L': 0, 'C': 1, 'R': 2, 'none': -1}[side],
        U, U, U,  # IL, IC, IR iguales si solo hay un U
        sL, sC, sR,
        Tmax=Tmax,
        dt=dt
    )
