import numpy as np
from numba import njit, prange

# ========= Helpers estímulo =========

@njit
def _onset_offset_from_codes(stim_code, delay_code, t1, t2, t3, t4):
    # stim: 0 VG,1 SS,2 SM,3 SL,4 SIL
    # delay: 0 DS,1 DM,2 DL
    if stim_code == 0:  # VG
        return 0.0, t4
    elif stim_code == 1:  # SS
        if delay_code == 0:   # DS
            return t2, t3
        elif delay_code == 1: # DM
            return t1, t2
        else:                 # DL
            return 0.0, t1
    elif stim_code == 2:  # SM
        if delay_code == 0:   # DS
            return t1, t3
        else:                 # DM
            return 0.0, t2
    elif stim_code == 3:      # SL
        return 0.0, t3
    else:                     # SIL
        return 0.0, 0.0

@njit
def _S_value(t, amp, d, onset, offset):
    if t < onset:
        return 0.0
    if t <= offset:
        return amp
    tail_end = offset + d
    if t <= tail_end and d > 0.0 and (abs(offset - onset) >= 1e-5):
        return amp * (1.0 - (t - offset) / d)
    return 0.0

@njit
def _U_value(t, amp, base, onset, offset):
    D = offset - onset
    if D <= 0.0:
        return base
    if t < onset or t > offset:
        return base
    return base + amp * (t - onset) / D


# ========= RNG determinista (thread-local) para CRN =========

@njit
def _splitmix64(x):
    """ Mezcla 64 bits → 64 bits (para derivar seeds independientes). """
    z = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z

@njit
def _xorshift64(state):
    """ Paso de xorshift64; devuelve (nuevo_estado, uint64). """
    x = state
    x ^= (x << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(7))
    x ^= (x << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x & np.uint64(0xFFFFFFFFFFFFFFFF), x

@njit
def _rng_u01(state):
    """ Uniforme (0,1) de 53 bits a partir de xorshift64. """
    state, bits = _xorshift64(state)
    # usar 53 bits de mantisa para double:
    u = ((bits >> np.uint64(11)) & np.uint64(0x1FFFFFFFFFFFFF)) * (1.0 / (1 << 53))
    if u <= 1e-16:
        u = 1e-16  # evita log(0) en Box-Muller
    elif u >= 1.0:
        u = 1.0 - 1e-16
    return state, u

@njit
def _rng_norm_pair(state):
    """ Dos normales N(0,1) via Box–Muller; devuelve (state, z0, z1). """
    state, u1 = _rng_u01(state)
    state, u2 = _rng_u01(state)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)
    return state, z0, z1


# ========= Drift =========

@njit
def _drift(x1, x2, IL, IC, IR, sL, sC, sR):
    x1sq = x1 * x1
    x2sq = x2 * x2
    s_sum  = sC + sL
    s_diff = sC - sL

    # --- F1 ---
    F1 = 5.0 * (IL - IC) + 20.0 * x1 * x2
    F1 -= 1.9047619047619 * x1 * (x1sq + 3.0 * x2sq)
    F1 += 5.0 * x1 * s_sum
    F1 += 10.0 * x1 * (0.904761904761905 * (IC + IL) - 0.0952380952380951 * IR + 0.226190476190476 * s_sum - 0.0238095238095238 * sR)
    F1 -= 10.0 * x2 * (IC - IL + 0.25 * s_diff)
    F1 -= 5.0 * (x2 + 0.25) * s_diff

    # --- F2 ---
    F2 = 3.33333333333333 * (IL - IC) * x1 + 3.09523809523809 * (IL + IC) * x2 + 1.66666666666667 * (IL + IC) + 13.0952380952381 * IR * x2 - 3.33333333333333 * IR
    F2 += (-1.9047619047619 * x1sq * x2 + 3.33333333333333 * x1sq - 10.0 * x2sq - 1.9047619047619 * x2 * (x1sq + 3.0 * x2sq))
    F2 += 2.44047619047619 * x2 * s_sum + 9.94047619047619 * x2 * sR + 0.416666666666667 * s_sum - 0.833333333333333 * sR

    return F1, F2

# def _drift(x1, x2, IL, IC, IR, sL, sC, sR):
#     term1_F1 = -5.0 * IC + 5.0 * IL
#     term2_F1 = 20.0 * x1 * x2
#     term3_F1 = -1.9047619047619 * x1 * (x1*x1 + 3.0 * x2*x2)
#     term4_F1 = 5.0 * x1 * (sC + sL)
#     term5_F1 = 10.0 * x1 * (
#         0.904761904761905 * IC +
#         0.904761904761905 * IL -
#         0.0952380952380951 * IR +
#         0.226190476190476 * sC +
#         0.226190476190476 * sL -
#         0.0238095238095238 * sR
#     )
#     term6_F1 = -10.0 * x2 * (IC - IL + 0.25 * sC - 0.25 * sL)
#     term7_F1 = -5.0 * (x2 + 0.25) * (sC - sL)
#     F1 = term1_F1 + term2_F1 + term3_F1 + term4_F1 + term5_F1 + term6_F1 + term7_F1

#     F2 = (
#         -3.33333333333333 * IC * x1 +
#          3.09523809523809 * IC * x2 +
#          1.66666666666667 * IC +
#          3.33333333333333 * IL * x1 +
#          3.09523809523809 * IL * x2 +
#          1.66666666666667 * IL +
#         13.0952380952381 * IR * x2 -
#          3.33333333333333 * IR -
#          1.9047619047619 * x1*x1 * x2 +
#          3.33333333333333 * x1*x1 -
#          10.0 * x2*x2 -
#          1.9047619047619 * x2 * (x1*x1 + 3.0 * x2*x2) +
#          2.44047619047619 * x2 * sC +
#          2.44047619047619 * x2 * sL +
#          9.94047619047619 * x2 * sR +
#          0.416666666666667 * sC +
#          0.416666666666667 * sL -
#          0.833333333333333 * sR
#     )
#     return F1, F2

# ========= Simulación Heun =========

@njit
def _single_path_heun_from_precomputed(S_t, U_t, side_code,
                                       sL, sC, sR, noise_amp,
                                       dt, th1, th2, th3,
                                       rng_state0):
    N = S_t.shape[0]
    x1 = 0.0
    x2 = 0.0
    state = rng_state0

    for i in range(N):
        Sval = S_t[i]
        Uval = U_t[i]
        if side_code == 0:   # L
            IL = Sval + Uval; IC = Uval;       IR = Uval
        elif side_code == 1: # C
            IL = Uval;       IC = Sval + Uval; IR = Uval
        elif side_code == 2: # R
            IL = Uval;       IC = Uval;        IR = Sval + Uval
        else:                # SIL
            IL = Uval;       IC = Uval;        IR = Uval

        # Ruido: necesitamos 3 normales; sacamos 4 y usamos 3
        state, z0, z1 = _rng_norm_pair(state)
        state, z2, z3 = _rng_norm_pair(state)
        dW0 = z0 * np.sqrt(dt)
        dW1 = z1 * np.sqrt(dt)
        dW2 = z2 * np.sqrt(dt)
        dB1 = (dW0 - dW1) / 2.0
        dB2 = (dW0 + dW1 - 2.0 * dW2) / 6.0
        n1 = noise_amp * dB1
        n2 = noise_amp * dB2

        # Heun
        f1a, f2a = _drift(x1, x2, IL, IC, IR, sL, sC, sR)
        x1p = x1 + f1a * dt + n1
        x2p = x2 + f2a * dt + n2
        f1b, f2b = _drift(x1p, x2p, IL, IC, IR, sL, sC, sR)
        x1 = x1 + 0.5 * (f1a + f1b) * dt + n1
        x2 = x2 + 0.5 * (f2a + f2b) * dt + n2

    r1 =  x1 + x2
    r2 = -x1 + x2
    r3 = -2.0 * x2
    if r1 > r2 and r1 > r3 and r1 > th1:
        return 0
    elif r2 > r1 and r2 > r3 and r2 > th2:
        return 1
    elif r3 > r1 and r3 > r2 and r3 > th3:
        return 2
    else:
        return -1

# ========= NLL paralelizado =========

@njit(parallel=True)
def nll_trials_numba(stimd, delayd, side, resp,
                     t1, t2, t3, t4,
                     theta, M, dt, alpha, th1, th2, th3,
                     seeds):
    sL, sC, sR = theta[0], theta[1], theta[2]
    noise_amp  = theta[3]
    S_amp, dS  = theta[4], theta[5]
    U_amp, U_base, U_on = theta[6], theta[7], theta[8]

    Ntr = stimd.shape[0]
    denom = M + 3.0 * alpha
    nll_per_trial = np.empty(Ntr, dtype=np.float64)

    for i in prange(Ntr):
        onset, offset = _onset_offset_from_codes(stimd[i], delayd[i], t1[i], t2[i], t3[i], t4[i])
        N = int(t4[i] / dt)

        S_t = np.empty(N, dtype=np.float64)
        U_t = np.empty(N, dtype=np.float64)
        for k in range(N):
            tt = k * dt
            S_t[k] = _S_value(tt, S_amp, dS, onset, offset)
            U_t[k] = _U_value(tt, U_amp, U_base, U_on, t4[i])

        mL = 0; mC = 0; mR = 0

        # RNG: semilla base por trial (CRN), desmezclada por réplica j
        base = _splitmix64(seeds[i])

        for j in range(M):
            # estado independiente por réplica (determinista y sin pisar otros hilos)
            state = _splitmix64(base ^ np.uint64(j + 1))

            k = _single_path_heun_from_precomputed(
                S_t, U_t, side[i],
                sL, sC, sR, noise_amp,
                dt, th1, th2, th3,
                state
            )
            if k == 0:   
                mL += 1
            elif k == 1: 
                mC += 1
            elif k == 2: 
                mR += 1

        pL = (mL + alpha) / denom
        pC = (mC + alpha) / denom
        pR = (mR + alpha) / denom

        yi = resp[i]
        if yi == 0:
            nll_per_trial[i] = -np.log(pL)
        elif yi == 1:
            nll_per_trial[i] = -np.log(pC)
        else:
            nll_per_trial[i] = -np.log(pR)

    return nll_per_trial.sum()