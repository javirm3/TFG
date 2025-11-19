module SimNLL
# θ = [sL, sC, sR, noise_amp, S_amp, dS, U_amp, U_base, U_on]

export nll_trials, onset_offset_from_codes

using Random
using Base.Threads


# ===================== Buffers globales por hilo (tamaño fijo) =====================
const _S_tls  = Ref(Vector{Vector{Float64}}())   # vector de vectores, inicialmente vacío
const _U_tls  = Ref(Vector{Vector{Float64}}())
const _Z0_tls = Ref(Vector{Vector{Float64}}())
const _Z1_tls = Ref(Vector{Vector{Float64}}())
const _Z2_tls = Ref(Vector{Vector{Float64}}())
const _Z3_tls = Ref(Vector{Vector{Float64}}())
const _rng_tls = Ref(Vector{Xoshiro}())
const _buffers_init = Ref(false)

# const _rng_tls = Ref{Vector{Xoshiro}}()
const _tls = Ref(Vector{Float64}())

function _init_buffers_if_needed(Nhint::Int)
    if !_buffers_init[]
        nt = Threads.nthreads()
        _S_tls[]  = [Vector{Float64}(undef, 0) for _ in 1:nt]
        _U_tls[]  = [Vector{Float64}(undef, 0) for _ in 1:nt]
        _Z0_tls[] = [Vector{Float64}(undef, 0) for _ in 1:nt]
        _Z1_tls[] = [Vector{Float64}(undef, 0) for _ in 1:nt]
        _Z2_tls[] = [Vector{Float64}(undef, 0) for _ in 1:nt]
        _Z3_tls[] = [Vector{Float64}(undef, 0) for _ in 1:nt]
        let rd = RandomDevice()
            _rng_tls[] = [Xoshiro(rand(rd, UInt64)) for _ in 1:nt]
        end
        # Sugerencia: reserva inicial para evitar realojos frecuentes
        for v in _S_tls[];  sizehint!(v, Nhint); end
        for v in _U_tls[];  sizehint!(v, Nhint); end
        for v in _Z0_tls[]; sizehint!(v, Nhint); end
        for v in _Z1_tls[]; sizehint!(v, Nhint); end
        for v in _Z2_tls[]; sizehint!(v, Nhint); end
        for v in _Z3_tls[]; sizehint!(v, Nhint); end
        _buffers_init[] = true
    end
end

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

@inline function U_temporal_value(t, amp, base, onset, offset)
    D = offset - onset
    if D <= 0.0 || t < onset || t > offset
        return base
    else
        return base + amp * (t - onset) / D
    end
end

@inline function U_spatial_value(t::Float64, U_amp::Float64, U_base::Float64,
                                 t1::Float64, t2::Float64, t3::Float64, t4::Float64,
                                 w1::Float64, w2::Float64, w3::Float64, w4::Float64)
    # ramp01(x) = clamp(x,0,1)
    r1 = clamp(t * w1,              0.0, 1.0)
    r2 = clamp((t - t1) * w2,       0.0, 1.0)
    r3 = clamp((t - t2) * w3,       0.0, 1.0)
    r4 = clamp((t - t3) * w4,       0.0, 1.0)
    return muladd(0.25 * U_amp, (r1 + r2 + r3 + r4), U_base)
end

@inline function U_ext_value(t, amp, onset, offset)
    return (t < onset) ? 0.0 : amp
end

@inline function phi(x::Float64)
    if x <= 0.0
        return 0.0
    elseif x <= 1.0
        return x * x
    else
        return 2.0 * sqrt(x - 0.75)
    end
end


# ===================== Drift =====================
# @inline function drift(x1, x2, IL, IC, IR, sL, sC, sR)
#     term1_F1 = -5.0 * IC + 5.0 * IL
#     term2_F1 = 20.0 * x1 * x2
#     term3_F1 = -1.9047619047619 * x1 * (x1*x1 + 3.0 * x2*x2)
#     term4_F1 = 5.0 * x1 * (sC + sL)
#     term5_F1 = 10.0 * x1 * (0.904761904761905 * IC + 0.904761904761905 * IL
#                            - 0.0952380952380951 * IR + 0.226190476190476 * sC
#                            + 0.226190476190476 * sL - 0.0238095238095238 * sR)
#     term6_F1 = -10.0 * x2 * (IC - IL + 0.25 * sC - 0.25 * sL)
#     term7_F1 = -5.0 * (x2 + 0.25) * (sC - sL)
#     F1 = term1_F1 + term2_F1 + term3_F1 + term4_F1 + term5_F1 + term6_F1 + term7_F1

#     F2 = (-3.33333333333333 * IC * x1 +
#           3.09523809523809  * IC * x2 + 1.66666666666667 * IC +
#           3.33333333333333  * IL * x1 +
#           3.09523809523809  * IL * x2 + 1.66666666666667 * IL +
#           13.0952380952381  * IR * x2 - 3.33333333333333 * IR -
#           1.9047619047619   * x1*x1 * x2 + 3.33333333333333 * x1*x1 -
#           10.0 * x2*x2 - 1.9047619047619 * x2 * (x1*x1 + 3.0 * x2*x2) +
#           2.44047619047619  * x2 * sC + 2.44047619047619 * x2 * sL +
#           9.94047619047619  * x2 * sR + 0.416666666666667 * sC +
#           0.416666666666667 * sL - 0.833333333333333 * sR)
#     return F1, F2
# end

@inline function drift(x1, x2, IL, IC, IR, sL, sC, sR)
    x1sq = x1 * x1
    x2sq = x2 * x2
    s_sum  = sC + sL
    s_diff = sC - sL

    # --- F1 ---
    F1 = 5.0 * (IL - IC) + 20.0 * x1 * x2
    F1 -= 1.9047619047619 * x1 * (x1sq + 3.0 * x2sq)
    F1 += 5.0 * x1 * s_sum
    F1 += 10.0 * x1 * (0.904761904761905 * (IC + IL) - 0.0952380952380951 * IR +
                       0.226190476190476 * s_sum - 0.0238095238095238 * sR)
    F1 -= 10.0 * x2 * (IC - IL + 0.25 * s_diff)
    F1 -= 5.0 * (x2 + 0.25) * s_diff

    # --- F2 ---
    F2 = 3.33333333333333 * (IL - IC) * x1 +
         3.09523809523809  * (IL + IC) * x2 +
         1.66666666666667  * (IL + IC) +
         13.0952380952381  * IR * x2 - 3.33333333333333 * IR
    F2 += (-1.9047619047619 * x1sq * x2 + 3.33333333333333 * x1sq -
           10.0 * x2sq - 1.9047619047619 * x2 * (x1sq + 3.0 * x2sq))
    F2 += 2.44047619047619 * x2 * s_sum + 9.94047619047619 * x2 * sR +
          0.416666666666667 * s_sum - 0.833333333333333 * sR

    return F1, F2
end

# ===================== Buffers por hilo (crecimiento lazy por threadid) =====================
const _S_cache  = Vector{Vector{Float64}}(undef, 0)
const _U_cache  = Vector{Vector{Float64}}(undef, 0)
const _Z0_cache = Vector{Vector{Float64}}(undef, 0)
const _Z1_cache = Vector{Vector{Float64}}(undef, 0)
const _Z2_cache = Vector{Vector{Float64}}(undef, 0)
const _buf_lock = ReentrantLock()

@inline function _ensure_slot!(tid::Int)
    if tid > length(_S_cache)
        lock(_buf_lock)
        try
            while tid > length(_S_cache)
                push!(_S_cache,  Float64[])
                push!(_U_cache,  Float64[])
                push!(_Z0_cache, Float64[])
                push!(_Z1_cache, Float64[])
                push!(_Z2_cache, Float64[])
            end
        finally
            unlock(_buf_lock)
        end
    end
    return nothing
end

# ===================== Relleno S/U =====================
# using LoopVectorization
@inline function fill_SU!(S_t::Vector{Float64}, U_t::Vector{Float64}, N::Int,
                          dt::Float64, S_amp::Float64, dS::Float64,
                          onset::Float64, offset::Float64,
                          U_amp::Float64, Ubase::Float64,
                          t1i::Float64, t2i::Float64, t3i::Float64, t4i::Float64)
    w1 = inv(t1i - 0.0)
    w2 = inv(t2i - t1i)
    w3 = inv(t3i - t2i)
    w4 = inv(t4i - t3i)

    @inbounds @simd for k in 1:N
        tt = (k - 1) * dt
        S_t[k] = S_value(tt, S_amp, dS, onset, offset)
        U_t[k] = U_spatial_value(tt, U_amp, Ubase, t1i, t2i, t3i, t4i, w1, w2, w3, w4)
    end
    return nothing
end


@inline function _heun_pop4_side!(
    S_t::Vector{Float64}, U_t::Vector{Float64}, N::Int,
    sL::Float64, sC::Float64, sR::Float64,
    dt::Float64, th1::Float64, th2::Float64, th3::Float64,
    Z0::Vector{Float64}, Z1::Vector{Float64}, Z2::Vector{Float64},
    Z3::Vector{Float64}, side::Int8;
    τ::Float64 = 0.1, c::Float64 = 1.0, g::Float64 = 1.0,
    I_I::Float64 = 1/3, noise_amp::Float64 = 0.0)

    # estados
    rL = 0.0; rC = 0.0; rR = 0.0; rI = 0.0
    invτ = 1.0/τ
    √dt = sqrt(dt)

    @inbounds for i in 1:N
        Sval = S_t[i]; Uval = U_t[i]
        # Inputs por lado (idéntico a tus Heun_*):
        if side == 0
            IL = Sval + Uval; IC = Uval;       IR = Uval
        elseif side == 1
            IL = Uval;       IC = Sval + Uval; IR = Uval
        else
            IL = Uval;       IC = Uval;        IR = Sval + Uval
        end

        # Ruido simple e independiente (reutilizando buffers)
        z0 = Z0[i]; z1 = Z1[i]; z2 = Z2[i]; z3 = Z3[i]
        ξL = noise_amp * z0 * √dt
        ξC = noise_amp * z1 * √dt
        ξR = noise_amp * z2 * √dt
        ξI = noise_amp * z3 * √dt

        # --- predictor ---
        fL = invτ*(-rL + phi(sL*rL - c*rI + IL))
        fC = invτ*(-rC + phi(sC*rC - c*rI + IC))
        fR = invτ*(-rR + phi(sR*rR - c*rI + IR))
        fI = invτ*(-rI + phi((g/3.0)*(rL + rC + rR) + I_I))

        rLp = rL + fL*dt + ξL
        rCp = rC + fC*dt + ξC
        rRp = rR + fR*dt + ξR
        rIp = rI + fI*dt + ξI

        # --- corrector ---
        fL2 = invτ*(-rLp + phi(sL*rLp - c*rIp + IL))
        fC2 = invτ*(-rCp + phi(sC*rCp - c*rIp + IC))
        fR2 = invτ*(-rRp + phi(sR*rRp - c*rIp + IR))
        fI2 = invτ*(-rIp + phi((g/3.0)*(rLp + rCp + rRp) + I_I))

        rL = rL + 0.5*(fL + fL2)*dt + ξL
        rC = rC + 0.5*(fC + fC2)*dt + ξC
        rR = rR + 0.5*(fR + fR2)*dt + ξR
        rI = rI + 0.5*(fI + fI2)*dt + ξI
    end

    # lectura de decisión (mismo “readout” que tenías con x1,x2)
    r1 =  rL                    # “Left”
    r2 =  rC                    # “Center”
    r3 =  rR                    # “Right”
    return (r1 > r2 && r1 > r3 && r1 > th1) ? 0 :
           (r2 > r1 && r2 > r3 && r2 > th2) ? 1 :
           (r3 > r1 && r3 > r2 && r3 > th3) ? 2 : -1
end



# ===================== Heun especializados por side (sin ramas por paso) =====================
# Generan ruido desde buffers Z0/Z1/Z2 ya rellenos con randn!
@inline function _heun_sideL!(S_t::Vector{Float64}, U_t::Vector{Float64}, N::Int, sL, sC, sR, s1_coef, s2_coef, dt, th1, th2, th3, Z0::Vector{Float64}, Z1::Vector{Float64}, Z2::Vector{Float64})
    x1 = 0.0; x2 = 0.0

    @inbounds for i in 1:N
        Sval = S_t[i]; Uval = U_t[i]
        IL = Sval + Uval; IC = Uval; IR = Uval

        z0 = Z0[i]; z1 = Z1[i]; z2 = Z2[i]
        n1 = s1_coef * (z0 - z1)
        n2 = s2_coef * (z0 + z1 - 2.0*z2)

        f1a, f2a = drift(x1, x2, IL, IC, IR, sL, sC, sR)
        x1p = x1 + f1a * dt + n1
        x2p = x2 + f2a * dt + n2
        f1b, f2b = drift(x1p, x2p, IL, IC, IR, sL, sC, sR)
        x1 = muladd(0.5*dt, (f1a + f1b), x1 + n1)
        x2 = muladd(0.5*dt, (f2a + f2b), x2 + n2)
    end
    r1 =  x1 + x2; r2 = -x1 + x2; r3 = -2.0 * x2
    return (r1 > r2 && r1 > r3 && r1 > th1) ? 0 :
           (r2 > r1 && r2 > r3 && r2 > th2) ? 1 :
           (r3 > r1 && r3 > r2 && r3 > th3) ? 2 : -1
end

@inline function _heun_sideC!(S_t,U_t,N::Int,sL,sC,sR,s1_coef, s2_coef,dt,th1,th2,th3,Z0,Z1,Z2)
    x1 = 0.0; x2 = 0.0
    @inbounds for i in 1:N
        Sval = S_t[i]; Uval = U_t[i]
        IL = Uval; IC = Sval + Uval; IR = Uval
        z0 = Z0[i]; z1 = Z1[i]; z2 = Z2[i]
        n1 = s1_coef * (z0 - z1)
        n2 = s2_coef * (z0 + z1 - 2.0*z2)
        f1a, f2a = drift(x1, x2, IL, IC, IR, sL, sC, sR)
        x1p = x1 + f1a * dt + n1
        x2p = x2 + f2a * dt + n2
        f1b, f2b = drift(x1p, x2p, IL, IC, IR, sL, sC, sR)
        x1 = muladd(0.5*dt, (f1a + f1b), x1 + n1)
        x2 = muladd(0.5*dt, (f2a + f2b), x2 + n2)
    end
    r1 =  x1 + x2; r2 = -x1 + x2; r3 = -2.0 * x2
    return (r1 > r2 && r1 > r3 && r1 > th1) ? 0 :
           (r2 > r1 && r2 > r3 && r2 > th2) ? 1 :
           (r3 > r1 && r3 > r2 && r3 > th3) ? 2 : -1
end

@inline function _heun_sideR!(S_t,U_t,N::Int,sL,sC,sR,s1_coef, s2_coef,dt,th1,th2,th3,Z0,Z1,Z2)
    x1 = 0.0; x2 = 0.0
    @inbounds for i in 1:N
        Sval = S_t[i]; Uval = U_t[i]
        IL = Uval; IC = Uval; IR = Sval + Uval
        z0 = Z0[i]; z1 = Z1[i]; z2 = Z2[i]
        n1 = s1_coef * (z0 - z1)
        n2 = s2_coef * (z0 + z1 - 2.0*z2)
        f1a, f2a = drift(x1, x2, IL, IC, IR, sL, sC, sR)
        x1p = x1 + f1a * dt + n1
        x2p = x2 + f2a * dt + n2
        f1b, f2b = drift(x1p, x2p, IL, IC, IR, sL, sC, sR)
        x1 = muladd(0.5*dt, (f1a + f1b), x1 + n1)
        x2 = muladd(0.5*dt, (f2a + f2b), x2 + n2)
    end
    r1 =  x1 + x2; r2 = -x1 + x2; r3 = -2.0 * x2
    return (r1 > r2 && r1 > r3 && r1 > th1) ? 0 :
           (r2 > r1 && r2 > r3 && r2 > th2) ? 1 :
           (r3 > r1 && r3 > r2 && r3 > th3) ? 2 : -1
end

# ===================== NLL paralelizado =====================

function nll_trials(stimd::Vector{Int8}, delayd::Vector{Int8}, side::Vector{Int8},
                    resp::Vector{Int8},
                    t1::Vector{Float64}, t2::Vector{Float64},
                    t3::Vector{Float64}, t4::Vector{Float64},
                    theta::Vector{Float64}, M::Int, dt::Float64, alpha::Float64,
                    th1::Float64, th2::Float64, th3::Float64,
                    seeds::Vector{UInt64}; use_pop4::Bool = false)
    @inbounds begin
        sL, sC, sR   = theta[1], theta[2], theta[3]
        noise_amp    = theta[4]
        S_amp, dS    = theta[5], theta[6]
        U_amp, Ubase = theta[7], theta[8]
        # U_on         = theta[9]
        U_ext_amp    = (length(theta) >= 9) ? theta[9] : 0.0
    end
    Nmax = Int(floor(maximum(t4) / dt))
    _init_buffers_if_needed(Nmax)
    Ntr   = length(stimd)
    denom = M + 3.0 * alpha
    if length(_tls[]) < Threads.maxthreadid()
        _tls[] = zeros(Float64, Threads.maxthreadid())
    else
        fill!(_tls[], 0.0)
    end
    tls = _tls[]
    
    # Pre-calcular coeficientes de ruido
    srt = sqrt(dt)
    s1_coef = noise_amp * (srt * 0.5)
    s2_coef = noise_amp * (srt / 6.0)

    Threads.@threads for i in 1:Ntr
        tid = min(Threads.threadid(), length(_S_tls[]))

        onset = 0.0
        offset = 0.0
        stim = stimd[i]
        delay = delayd[i]
        if stim == 0
            onset = 0.0; offset = t4[i]
        elseif stim == 1
            if delay == 0
                onset = t2[i]; offset = t3[i]
            elseif delay == 1
                onset = t1[i]; offset = t2[i]
            else
                onset = 0.0; offset = t1[i]
            end
        elseif stim == 2
            if delay == 0
                onset = t1[i]; offset = t3[i]
            else
                onset = 0.0; offset = t2[i]
            end
        elseif stim == 3
            onset = 0.0; offset = t3[i]
        else
            onset = 0.0; offset = 0.0
        end
        N = Int(floor(t4[i] / dt))

        # Buffers por hilo (reutilizados) - UNA VEZ
        S_t = _S_tls[][tid];  resize!(S_t, N)
        U_t = _U_tls[][tid];  resize!(U_t, N)
        Z0  = _Z0_tls[][tid]; resize!(Z0,  N)
        Z1  = _Z1_tls[][tid]; resize!(Z1,  N)
        Z2  = _Z2_tls[][tid]; resize!(Z2,  N)
        Z3  = _Z3_tls[][tid]; resize!(Z3,  N)

        w1 = inv(t1[i])
        w2 = inv(t2[i] - t1[i])
        w3 = inv(t3[i] - t2[i])
        w4 = inv(t4[i] - t3[i])
        @inbounds @simd for k in 1:N
            tt = (k - 1) * dt
            S_t[k] = S_value(tt, S_amp, dS, onset, offset)
            U_t[k] = U_spatial_value(tt, U_amp, Ubase, t1[i], t2[i], t3[i], t4[i], w1, w2, w3, w4) + U_ext_value(tt, U_ext_amp, onset, t4[i])
        end
        mL = 0; mC = 0; mR = 0

        rng = _rng_tls[][tid]
        @inbounds for j in 1:M
            randn!(rng, Z0)
            randn!(rng, Z1)
            randn!(rng, Z2)
            randn!(rng, Z3)

            k = if use_pop4
                _heun_pop4_side!(S_t, U_t, N, sL, sC, sR, dt, th1, th2, th3,
                                Z0, Z1, Z2, Z3, side[i];
                                τ=0.1, c=1.0, g=1.0, I_I=1/3, noise_amp=noise_amp)
            else
                side[i] == 0 ? _heun_sideL!(S_t,U_t,N,sL,sC,sR,s1_coef,s2_coef,dt,th1,th2,th3,Z0,Z1,Z2) :
                side[i] == 1 ? _heun_sideC!(S_t,U_t,N,sL,sC,sR,s1_coef,s2_coef,dt,th1,th2,th3,Z0,Z1,Z2) :
                            _heun_sideR!(S_t,U_t,N,sL,sC,sR,s1_coef,s2_coef,dt,th1,th2,th3,Z0,Z1,Z2)
            end

            if     k == 0; mL += 1
            elseif k == 1; mC += 1
            elseif k == 2; mR += 1
            end
        end

        pL = (mL + alpha) / denom
        pC = (mC + alpha) / denom
        pR = (mR + alpha) / denom
        yi = resp[i]
        li = yi == 0 ? -log(pL) : (yi == 1 ? -log(pC) : -log(pR))

        tls[tid] += li
    end

    return sum(tls)
end

end # module