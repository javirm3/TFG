module SimNLLProfile
# Perfilado de la NLL con Heun y 3 normales por paso usando Random.randn
# θ = [sL, sC, sR, noise_amp, S_amp, dS, U_amp, U_base, U_on]

export nll_trials_profile, onset_offset_from_codes, rng_mode

using Random
using Base.Threads

# ===================== Seeding para CRN =====================
@inline function splitmix64(x::UInt64)
    z = x + 0x9E3779B97F4A7C15
    z ⊻= z >> 30; z *= 0xBF58476D1CE4E5B9
    z ⊻= z >> 27; z *= 0x94D049BB133111EB
    return z ⊻ (z >> 31)
end

@inline make_rng(seed::UInt64) = Xoshiro(seed)

# ===================== Helpers estímulo =====================
@inline function onset_offset_from_codes(stim::Int8, delay::Int8, t1, t2, t3, t4)
    # stim: 0 VG,1 SS,2 SM,3 SL,4 SIL ; delay: 0 DS,1 DM,2 DL
    if stim == 0
        return 0.0, t4
    elseif stim == 1
        return delay == 0 ? (t2,t3) : (delay == 1 ? (t1,t2) : (0.0,t1))
    elseif stim == 2
        return delay == 0 ? (t1,t3) : (0.0,t2)
    elseif stim == 3
        return 0.0, t3
    else
        return 0.0, 0.0
    end
end

@inline function S_value(t, amp, d, onset, offset)
    if t < onset
        return 0.0
    elseif t <= offset
        return amp
    else
        tail_end = offset + d
        return (d > 0.0 && abs(offset - onset) >= 1e-5 && t <= tail_end) ?
               amp * (1.0 - (t - offset) / d) : 0.0
    end
end

@inline function U_value(t, amp, base, onset, offset)
    D = offset - onset
    if D <= 0.0 || t < onset || t > offset
        return base
    else
        return base + amp * (t - onset) / D
    end
end

# ===================== Drift =====================
@inline function drift(x1, x2, IL, IC, IR, sL, sC, sR)
    term1_F1 = -5.0 * IC + 5.0 * IL
    term2_F1 = 20.0 * x1 * x2
    term3_F1 = -1.9047619047619 * x1 * (x1*x1 + 3.0 * x2*x2)
    term4_F1 = 5.0 * x1 * (sC + sL)
    term5_F1 = 10.0 * x1 * (0.904761904761905 * IC + 0.904761904761905 * IL
                           - 0.0952380952380951 * IR + 0.226190476190476 * sC
                           + 0.226190476190476 * sL - 0.0238095238095238 * sR)
    term6_F1 = -10.0 * x2 * (IC - IL + 0.25 * sC - 0.25 * sL)
    term7_F1 = -5.0 * (x2 + 0.25) * (sC - sL)
    F1 = term1_F1 + term2_F1 + term3_F1 + term4_F1 + term5_F1 + term6_F1 + term7_F1

    F2 = (-3.33333333333333 * IC * x1 +
          3.09523809523809  * IC * x2 + 1.66666666666667 * IC +
          3.33333333333333  * IL * x1 +
          3.09523809523809  * IL * x2 + 1.66666666666667 * IL +
          13.0952380952381  * IR * x2 - 3.33333333333333 * IR -
          1.9047619047619   * x1*x1 * x2 + 3.33333333333333 * x1*x1 -
          10.0 * x2*x2 - 1.9047619047619 * x2 * (x1*x1 + 3.0 * x2*x2) +
          2.44047619047619  * x2 * sC + 2.44047619047619 * x2 * sL +
          9.94047619047619  * x2 * sR + 0.416666666666667 * sC +
          0.416666666666667 * sL - 0.833333333333333 * sR)
    return F1, F2
end

# ===================== Relleno S/U =====================
@inline function fill_SU!(S_t::Vector{Float64}, U_t::Vector{Float64}, N::Int,
                          dt::Float64, S_amp::Float64, dS::Float64,
                          onset::Float64, offset::Float64,
                          U_amp::Float64, Ubase::Float64, U_on::Float64, t4i::Float64)
    @inbounds for k in 1:N
        tt = (k - 1) * dt
        S_t[k] = S_value(tt, S_amp, dS, onset, offset)
        U_t[k] = U_value(tt, U_amp, Ubase, U_on, t4i)
    end
    return nothing
end

# ===================== Heun (perfilado, 3 normales/step) =====================
@inline function single_path_heun_profile_rand3!(
    S_t::Vector{Float64}, U_t::Vector{Float64}, side::Int8,
    sL, sC, sR, noise_amp, dt, th1, th2, th3,
    rng::Random.AbstractRNG,
    acc_rng::Base.RefValue{UInt64}, acc_noise::Base.RefValue{UInt64},
    acc_drift::Base.RefValue{UInt64}, acc_update::Base.RefValue{UInt64}
)
    N = length(S_t)
    x1 = 0.0; x2 = 0.0
    sqrt_dt = sqrt(dt)

    @inbounds for i in 1:N
        Sval = S_t[i]; Uval = U_t[i]
        IL::Float64 = Uval; IC::Float64 = Uval; IR::Float64 = Uval
        if side == 0
            IL = Sval + Uval
        elseif side == 1
            IC = Sval + Uval
        elseif side == 2
            IR = Sval + Uval
        end

        t0 = time_ns()
        z0 = randn(rng); z1 = randn(rng); z2 = randn(rng)
        acc_rng[] += (time_ns() - t0)

        t0 = time_ns()
        dW0 = z0 * sqrt_dt
        dW1 = z1 * sqrt_dt
        dW2 = z2 * sqrt_dt
        dB1 = (dW0 - dW1) / 2.0
        dB2 = (dW0 + dW1 - 2.0 * dW2) / 6.0
        n1 = noise_amp * dB1
        n2 = noise_amp * dB2
        acc_noise[] += (time_ns() - t0)

        t0 = time_ns()
        f1a, f2a = drift(x1, x2, IL, IC, IR, sL, sC, sR)
        x1p = x1 + f1a * dt + n1
        x2p = x2 + f2a * dt + n2
        f1b, f2b = drift(x1p, x2p, IL, IC, IR, sL, sC, sR)
        acc_drift[] += (time_ns() - t0)

        t0 = time_ns()
        x1 += 0.5 * (f1a + f1b) * dt + n1
        x2 += 0.5 * (f2a + f2b) * dt + n2
        acc_update[] += (time_ns() - t0)
    end

    r1 =  x1 + x2
    r2 = -x1 + x2
    r3 = -2.0 * x2
    if r1 > r2 && r1 > r3 && r1 > th1
        return 0
    elseif r2 > r1 && r2 > r3 && r2 > th2
        return 1
    elseif r3 > r1 && r3 > r2 && r3 > th3
        return 2
    else
        return -1
    end
end

# ===================== NLL (perfil + multihilo sin librerías externas) =====================
const rng_mode = (; CRN = :crn, Independent = :independent)

"""
nll, stats = nll_trials_profile(...; rng_mode=:crn)

stats::NamedTuple con:
- trials_sum, fill_su, heun_rng, heun_noise, heun_drift, heun_update (en segundos, acumulados)
"""
function nll_trials_profile(stimd::Vector{Int8}, delayd::Vector{Int8}, side::Vector{Int8},
                            resp::Vector{Int8},
                            t1::Vector{Float64}, t2::Vector{Float64},
                            t3::Vector{Float64}, t4::Vector{Float64},
                            theta::Vector{Float64}, M::Int, dt::Float64, alpha::Float64,
                            th1::Float64, th2::Float64, th3::Float64,
                            seeds::Vector{UInt64};
                            rng_mode::Symbol = :crn)

    @inbounds begin
        sL, sC, sR   = theta[1], theta[2], theta[3]
        noise_amp    = theta[4]
        S_amp, dS    = theta[5], theta[6]
        U_amp, Ubase = theta[7], theta[8]
        U_on         = theta[9]
    end

    Ntr = length(stimd)
    denom = M + 3.0 * alpha

    nt = Threads.nthreads()
    # acumuladores por hilo (UInt64 en ns)
    acc_trials   = zeros(UInt64, nt)
    acc_fillsu   = zeros(UInt64, nt)
    acc_rng      = zeros(UInt64, nt)
    acc_noise    = zeros(UInt64, nt)
    acc_drift    = zeros(UInt64, nt)
    acc_update   = zeros(UInt64, nt)

    nll_sum = Threads.Atomic{Float64}(0.0)

    Threads.@threads for i in 1:Ntr
        tid = Threads.threadid()

        t_trial0 = time_ns()

        onset, offset = onset_offset_from_codes(stimd[i], delayd[i], t1[i], t2[i], t3[i], t4[i])
        N = Int(floor(t4[i] / dt))

        S_t = Vector{Float64}(undef, N)
        U_t = Vector{Float64}(undef, N)

        t0 = time_ns()
        fill_SU!(S_t, U_t, N, dt, S_amp, dS, onset, offset, U_amp, Ubase, U_on, t4[i])
        acc_fillsu[tid] += (time_ns() - t0)

        mL = 0; mC = 0; mR = 0

        if rng_mode === :crn
            base = splitmix64(seeds[i])
            @inbounds for j in 1:M
                rng = make_rng(splitmix64(base ⊻ UInt64(j + 1)))

                # contadores locales (Refs) para este path
                r_rng    = Ref{UInt64}(0)
                r_noise  = Ref{UInt64}(0)
                r_drift  = Ref{UInt64}(0)
                r_update = Ref{UInt64}(0)

                k = single_path_heun_profile_rand3!(
                        S_t, U_t, side[i], sL, sC, sR, noise_amp, dt, th1, th2, th3,
                        rng, r_rng, r_noise, r_drift, r_update)

                # acumula en los buckets del hilo
                acc_rng[tid]    += r_rng[]
                acc_noise[tid]  += r_noise[]
                acc_drift[tid]  += r_drift[]
                acc_update[tid] += r_update[]

                if     k == 0; mL += 1
                elseif k == 1; mC += 1
                elseif k == 2; mR += 1
                end
            end
        else
            tbase = splitmix64(UInt64(0xCBF29CE484222325) ⊻ UInt64(tid) ⊻ seeds[i])
            @inbounds for j in 1:M
                rng = make_rng(splitmix64(tbase ⊻ UInt64(j) ⊻ UInt64(j*1664525 + i)))

                r_rng    = Ref{UInt64}(0)
                r_noise  = Ref{UInt64}(0)
                r_drift  = Ref{UInt64}(0)
                r_update = Ref{UInt64}(0)

                k = single_path_heun_profile_rand3!(
                        S_t, U_t, side[i], sL, sC, sR, noise_amp, dt, th1, th2, th3,
                        rng, r_rng, r_noise, r_drift, r_update)

                acc_rng[tid]    += r_rng[]
                acc_noise[tid]  += r_noise[]
                acc_drift[tid]  += r_drift[]
                acc_update[tid] += r_update[]

                if     k == 0; mL += 1
                elseif k == 1; mC += 1
                elseif k == 2; mR += 1
                end
            end
        end

        pL = (mL + alpha) / denom
        pC = (mC + alpha) / denom
        pR = (mR + alpha) / denom
        yi = resp[i]
        li = yi == 0 ? -log(pL) : (yi == 1 ? -log(pC) : -log(pR))
        Threads.atomic_add!(nll_sum, li)

        acc_trials[tid] += (time_ns() - t_trial0)
    end

    # reduce y convertir a segundos
    ns2s(x::UInt64) = x / 1.0e9
    stats = (
        trials_sum = ns2s(sum(acc_trials)),
        fill_su    = ns2s(sum(acc_fillsu)),
        heun_rng   = ns2s(sum(acc_rng)),
        heun_noise = ns2s(sum(acc_noise)),
        heun_drift = ns2s(sum(acc_drift)),
        heun_update= ns2s(sum(acc_update)),
    )

    return nll_sum[], stats
end

end # module