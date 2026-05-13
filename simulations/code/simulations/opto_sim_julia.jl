module OptoSimJulia

using Random
using Base.Threads
using Statistics

export simulate_frac_correct_opto, simulate_choice_probs_opto, simulate_choice_probs_trials_opto

@inline function splitmix64(x::UInt64)::UInt64
    x += 0x9e3779b97f4a7c15
    z = x
    z = xor(z, z >> 30) * 0xbf58476d1ce4e5b9
    z = xor(z, z >> 27) * 0x94d049bb133111eb
    return xor(z, z >> 31)
end

@inline function onset_offset_from_codes(
    stim::Int8,
    delay::Int8,
    t1::Float32,
    t2::Float32,
    t3::Float32,
    t4::Float32,
)
    # stim: 0 VG, 1 SS, 2 SM, 3 SL, 4 SIL
    # delay: 0 DS, 1 DM, 2 DL
    if stim == Int8(0)
        return 0f0, t4
    elseif stim == Int8(1)
        return delay == Int8(0) ? (t2, t3) : (delay == Int8(1) ? (t1, t2) : (0f0, t1))
    elseif stim == Int8(2)
        return delay == Int8(0) ? (t1, t3) : (0f0, t2)
    elseif stim == Int8(3)
        return 0f0, t3
    else
        return 0f0, 0f0
    end
end

@inline function S_value(t::Float32, amp::Float32, d::Float32, onset::Float32, offset::Float32)::Float32
    if t < onset
        return 0f0
    elseif t <= offset
        return amp
    end
    tail_end = offset + d
    return (d > 0f0 && t <= tail_end) ? amp * (1f0 - (t - offset) / d) : 0f0
end

@inline function U_temporal_value(
    t::Float32,
    amp::Float32,
    base::Float32,
    onset::Float32,
    offset::Float32,
)::Float32
    return (t >= onset && t <= offset) ? amp + base : base
end

@inline function U_spatial_value(
    t::Float32,
    amp::Float32,
    base::Float32,
    t1::Float32,
    t2::Float32,
    t3::Float32,
    t4::Float32,
)::Float32
    w1 = t1 > 0f0 ? inv(t1) : 0f0
    w2 = t2 > t1 ? inv(t2 - t1) : 0f0
    w3 = t3 > t2 ? inv(t3 - t2) : 0f0
    w4 = t4 > t3 ? inv(t4 - t3) : 0f0
    r1 = clamp(t * w1, 0f0, 1f0)
    r2 = clamp((t - t1) * w2, 0f0, 1f0)
    r3 = clamp((t - t2) * w3, 0f0, 1f0)
    r4 = clamp((t - t3) * w4, 0f0, 1f0)
    return base + 0.25f0 * amp * (r1 + r2 + r3 + r4)
end

@inline function U_ext_value(t::Float32, amp::Float32, onset::Float32, offset::Float32)::Float32
    return (t >= onset && t <= offset) ? amp : 0f0
end

@inline function drift(
    x1::Float32,
    x2::Float32,
    IL::Float32,
    IC::Float32,
    IR::Float32,
    sL::Float32,
    sC::Float32,
    sR::Float32,
)
    x1sq = x1 * x1
    x2sq = x2 * x2
    s_sum = sC + sL
    s_diff = sC - sL

    F1 = 5f0 * (IL - IC) + 20f0 * x1 * x2
    F1 -= 1.9047619047619f0 * x1 * (x1sq + 3f0 * x2sq)
    F1 += 5f0 * x1 * s_sum
    F1 += 10f0 * x1 * (
        0.904761904761905f0 * (IC + IL) -
        0.0952380952380951f0 * IR +
        0.226190476190476f0 * s_sum -
        0.0238095238095238f0 * sR
    )
    F1 -= 10f0 * x2 * (IC - IL + 0.25f0 * s_diff)
    F1 -= 5f0 * (x2 + 0.25f0) * s_diff

    F2 = 3.33333333333333f0 * (IL - IC) * x1 +
         3.09523809523809f0 * (IL + IC) * x2 +
         1.66666666666667f0 * (IL + IC) +
         13.0952380952381f0 * IR * x2 -
         3.33333333333333f0 * IR
    F2 += -1.9047619047619f0 * x1sq * x2 +
          3.33333333333333f0 * x1sq -
          10f0 * x2sq -
          1.9047619047619f0 * x2 * (x1sq + 3f0 * x2sq)
    F2 += 2.44047619047619f0 * x2 * s_sum +
          9.94047619047619f0 * x2 * sR +
          0.416666666666667f0 * s_sum -
          0.833333333333333f0 * sR

    return Float32(F1), Float32(F2)
end

@inline function single_path_heun_opto!(
    S_t::Vector{Float32},
    U_t::Vector{Float32},
    side_code::Int8,
    sL::Float32,
    sC::Float32,
    sR::Float32,
    noise_amp::Float32,
    dt::Float32,
    th1::Float32,
    th2::Float32,
    th3::Float32,
    opto_target::Int8,
    opto_amp::Float32,
    rng::Xoshiro,
)::Int8
    x1 = 0f0
    x2 = 0f0
    sqrt_dt = sqrt(dt)

    @inbounds for i in eachindex(S_t)
        Sval = S_t[i]
        Uval = U_t[i]
        if side_code == Int8(0)
            IL = Sval + Uval
            IC = Uval
            IR = Uval
        elseif side_code == Int8(1)
            IL = Uval
            IC = Sval + Uval
            IR = Uval
        elseif side_code == Int8(2)
            IL = Uval
            IC = Uval
            IR = Sval + Uval
        else
            IL = Uval
            IC = Uval
            IR = Uval
        end

        IC += opto_amp

        dW0 = randn(rng, Float32) * sqrt_dt
        dW1 = randn(rng, Float32) * sqrt_dt
        dW2 = randn(rng, Float32) * sqrt_dt
        n1 = noise_amp * ((dW0 - dW1) * 0.5f0)
        n2 = noise_amp * ((dW0 + dW1 - 2f0 * dW2) / 6f0)

        f1a, f2a = drift(x1, x2, IL, IC, IR, sL, sC, sR)
        x1p = x1 + f1a * dt + n1
        x2p = x2 + f2a * dt + n2
        f1b, f2b = drift(x1p, x2p, IL, IC, IR, sL, sC, sR)
        x1 += 0.5f0 * (f1a + f1b) * dt + n1
        x2 += 0.5f0 * (f2a + f2b) * dt + n2
    end

    r1 = x1 + x2
    r2 = -x1 + x2
    r3 = -2f0 * x2
    if r1 > r2 && r1 > r3 && r1 > th1
        return Int8(0)
    elseif r2 > r1 && r2 > r3 && r2 > th2
        return Int8(1)
    elseif r3 > r1 && r3 > r2 && r3 > th3
        return Int8(2)
    end
    return Int8(-1)
end

function simulate_frac_correct_opto(
    stimd_in,
    delayd_in,
    side_in,
    t1_in,
    t2_in,
    t3_in,
    t4_in,
    theta_in,
    M::Integer,
    dt_in,
    th1_in,
    th2_in,
    th3_in,
    opto_target_in::Integer,
    opto_amp_in,
    use_spatial::Bool,
    seed_in,
)::Float64
    stimd = Vector{Int8}(stimd_in)
    delayd = Vector{Int8}(delayd_in)
    side = Vector{Int8}(side_in)
    t1 = Vector{Float32}(t1_in)
    t2 = Vector{Float32}(t2_in)
    t3 = Vector{Float32}(t3_in)
    t4 = Vector{Float32}(t4_in)
    theta = Vector{Float32}(theta_in)

    sL = get(theta, 1, 0f0)
    sC = get(theta, 2, 0f0)
    sR = get(theta, 3, 0f0)
    noise_amp = get(theta, 4, 0f0)
    S_amp = get(theta, 5, 0f0)
    dS = get(theta, 6, 0f0)
    U_amp = get(theta, 7, 0f0)
    U_base = get(theta, 8, 0f0)
    U_on = get(theta, 9, 0f0)
    U_ext_amp = get(theta, 10, 0f0)

    dt = Float32(dt_in)
    th1 = Float32(th1_in)
    th2 = Float32(th2_in)
    th3 = Float32(th3_in)
    opto_target = Int8(opto_target_in)
    opto_amp = Float32(opto_amp_in)
    seed = UInt64(seed_in)

    Ntr = length(stimd)
    if Ntr == 0 || M <= 0
        return NaN
    end

    correct_per_trial = zeros(Float64, Ntr)

    Threads.@threads for i in 1:Ntr
        n_steps = Int(floor(t4[i] / dt))
        if n_steps <= 0
            correct_per_trial[i] = NaN
            continue
        end

        onset, offset = onset_offset_from_codes(stimd[i], delayd[i], t1[i], t2[i], t3[i], t4[i])
        S_t = Vector{Float32}(undef, n_steps)
        U_t = Vector{Float32}(undef, n_steps)
        @inbounds for k in 1:n_steps
            tt = Float32(k - 1) * dt
            S_t[k] = S_value(tt, S_amp, dS, onset, offset)
            U_t[k] = if use_spatial
                U_spatial_value(tt, U_amp, U_base, t1[i], t2[i], t3[i], t4[i])
            else
                U_temporal_value(tt, U_amp, U_base, U_on, t4[i])
            end
            U_t[k] += U_ext_value(tt, U_ext_amp, onset, t4[i])
        end

        rng = Xoshiro(splitmix64(xor(seed, UInt64(i))))
        n_correct = 0
        @inbounds for _ in 1:M
            choice = single_path_heun_opto!(
                S_t,
                U_t,
                side[i],
                sL,
                sC,
                sR,
                noise_amp,
                dt,
                th1,
                th2,
                th3,
                opto_target,
                opto_amp,
                rng,
            )
            n_correct += choice == side[i] ? 1 : 0
        end
        correct_per_trial[i] = n_correct / M
    end

    valid = filter(isfinite, correct_per_trial)
    return isempty(valid) ? NaN : mean(valid)
end

function simulate_choice_probs_opto(
    stimd_in,
    delayd_in,
    side_in,
    t1_in,
    t2_in,
    t3_in,
    t4_in,
    theta_in,
    M::Integer,
    dt_in,
    th1_in,
    th2_in,
    th3_in,
    opto_target_in::Integer,
    opto_amp_in,
    use_spatial::Bool,
    seed_in,
)
    stimd = Vector{Int8}(stimd_in)
    delayd = Vector{Int8}(delayd_in)
    side = Vector{Int8}(side_in)
    t1 = Vector{Float32}(t1_in)
    t2 = Vector{Float32}(t2_in)
    t3 = Vector{Float32}(t3_in)
    t4 = Vector{Float32}(t4_in)
    theta = Vector{Float32}(theta_in)

    sL = get(theta, 1, 0f0)
    sC = get(theta, 2, 0f0)
    sR = get(theta, 3, 0f0)
    noise_amp = get(theta, 4, 0f0)
    S_amp = get(theta, 5, 0f0)
    dS = get(theta, 6, 0f0)
    U_amp = get(theta, 7, 0f0)
    U_base = get(theta, 8, 0f0)
    U_on = get(theta, 9, 0f0)
    U_ext_amp = get(theta, 10, 0f0)

    dt = Float32(dt_in)
    th1 = Float32(th1_in)
    th2 = Float32(th2_in)
    th3 = Float32(th3_in)
    opto_target = Int8(opto_target_in)
    opto_amp = Float32(opto_amp_in)
    seed = UInt64(seed_in)

    Ntr = length(stimd)
    if Ntr == 0 || M <= 0
        return Float64[NaN, NaN, NaN]
    end

    pL_vec = fill(NaN, Ntr)
    pC_vec = fill(NaN, Ntr)
    pR_vec = fill(NaN, Ntr)

    Threads.@threads for i in 1:Ntr
        n_steps = Int(floor(t4[i] / dt))
        if n_steps <= 0
            continue
        end

        onset, offset = onset_offset_from_codes(stimd[i], delayd[i], t1[i], t2[i], t3[i], t4[i])
        S_t = Vector{Float32}(undef, n_steps)
        U_t = Vector{Float32}(undef, n_steps)
        @inbounds for k in 1:n_steps
            tt = Float32(k - 1) * dt
            S_t[k] = S_value(tt, S_amp, dS, onset, offset)
            U_t[k] = if use_spatial
                U_spatial_value(tt, U_amp, U_base, t1[i], t2[i], t3[i], t4[i])
            else
                U_temporal_value(tt, U_amp, U_base, U_on, t4[i])
            end
            U_t[k] += U_ext_value(tt, U_ext_amp, onset, t4[i])
        end

        rng = Xoshiro(splitmix64(xor(seed, UInt64(i))))
        nL = 0
        nC = 0
        nR = 0
        @inbounds for _ in 1:M
            choice = single_path_heun_opto!(
                S_t,
                U_t,
                side[i],
                sL,
                sC,
                sR,
                noise_amp,
                dt,
                th1,
                th2,
                th3,
                opto_target,
                opto_amp,
                rng,
            )
            if choice == Int8(0)
                nL += 1
            elseif choice == Int8(1)
                nC += 1
            elseif choice == Int8(2)
                nR += 1
            end
        end
        pL_vec[i] = nL / M
        pC_vec[i] = nC / M
        pR_vec[i] = nR / M
    end

    valid = isfinite.(pL_vec) .& isfinite.(pC_vec) .& isfinite.(pR_vec)
    if !any(valid)
        return Float64[NaN, NaN, NaN]
    end
    return Float64[mean(pL_vec[valid]), mean(pC_vec[valid]), mean(pR_vec[valid])]
end

function simulate_choice_probs_trials_opto(
    stimd_in,
    delayd_in,
    side_in,
    t1_in,
    t2_in,
    t3_in,
    t4_in,
    theta_in,
    M::Integer,
    dt_in,
    th1_in,
    th2_in,
    th3_in,
    opto_target_in::Integer,
    opto_amp_in,
    use_spatial::Bool,
    seed_in,
)
    stimd = Vector{Int8}(stimd_in)
    delayd = Vector{Int8}(delayd_in)
    side = Vector{Int8}(side_in)
    t1 = Vector{Float32}(t1_in)
    t2 = Vector{Float32}(t2_in)
    t3 = Vector{Float32}(t3_in)
    t4 = Vector{Float32}(t4_in)
    theta = Vector{Float32}(theta_in)

    sL = get(theta, 1, 0f0)
    sC = get(theta, 2, 0f0)
    sR = get(theta, 3, 0f0)
    noise_amp = get(theta, 4, 0f0)
    S_amp = get(theta, 5, 0f0)
    dS = get(theta, 6, 0f0)
    U_amp = get(theta, 7, 0f0)
    U_base = get(theta, 8, 0f0)
    U_on = get(theta, 9, 0f0)
    U_ext_amp = get(theta, 10, 0f0)

    dt = Float32(dt_in)
    th1 = Float32(th1_in)
    th2 = Float32(th2_in)
    th3 = Float32(th3_in)
    opto_target = Int8(opto_target_in)
    opto_amp = Float32(opto_amp_in)
    seed = UInt64(seed_in)

    Ntr = length(stimd)
    probs = fill(Float64(NaN), Ntr, 3)
    if Ntr == 0 || M <= 0
        return probs
    end

    Threads.@threads for i in 1:Ntr
        n_steps = Int(floor(t4[i] / dt))
        if n_steps <= 0
            continue
        end

        onset, offset = onset_offset_from_codes(stimd[i], delayd[i], t1[i], t2[i], t3[i], t4[i])
        S_t = Vector{Float32}(undef, n_steps)
        U_t = Vector{Float32}(undef, n_steps)
        @inbounds for k in 1:n_steps
            tt = Float32(k - 1) * dt
            S_t[k] = S_value(tt, S_amp, dS, onset, offset)
            U_t[k] = if use_spatial
                U_spatial_value(tt, U_amp, U_base, t1[i], t2[i], t3[i], t4[i])
            else
                U_temporal_value(tt, U_amp, U_base, U_on, t4[i])
            end
            U_t[k] += U_ext_value(tt, U_ext_amp, onset, t4[i])
        end

        rng = Xoshiro(splitmix64(xor(seed, UInt64(i))))
        nL = 0
        nC = 0
        nR = 0
        @inbounds for _ in 1:M
            choice = single_path_heun_opto!(
                S_t,
                U_t,
                side[i],
                sL,
                sC,
                sR,
                noise_amp,
                dt,
                th1,
                th2,
                th3,
                opto_target,
                opto_amp,
                rng,
            )
            if choice == Int8(0)
                nL += 1
            elseif choice == Int8(1)
                nC += 1
            elseif choice == Int8(2)
                nR += 1
            end
        end
        probs[i, 1] = nL / M
        probs[i, 2] = nC / M
        probs[i, 3] = nR / M
    end

    return probs
end

end
