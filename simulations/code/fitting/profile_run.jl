# Ejecuta con:
#   JULIA_NUM_THREADS=14 julia -O3 --check-bounds=no profile_run.jl 10000 200

using Printf
using Random

include("SimNLLProfile.jl")
using .SimNLLProfile

function make_synth(Ntr::Int, dt::Float64; seed=12345)
    rng = MersenneTwister(seed)
    stimd = fill(Int8(1), Ntr)          # SS
    delayd= fill(Int8(1), Ntr)          # DM
    side  = rand(rng, Int8[0,1,2], Ntr) # L/C/R
    resp  = rand(rng, Int8[0,1,2], Ntr)
    t1 = fill(1.0, Ntr); t2 = fill(2.0, Ntr)
    t3 = fill(3.0, Ntr); t4 = fill(5.0, Ntr)
    seeds = rand(rng, UInt64, Ntr)
    return stimd,delayd,side,resp,t1,t2,t3,t4,seeds
end

function main()
    Ntr = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
    M   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
    dt = 0.1/40.0
    alpha = 0.5
    th = 0.5

    @printf("== SimNLLProfileRand ==\nTrials=%d  M=%d  dt=%.4f  threads=%d\n",
            Ntr, M, dt, Threads.nthreads())

    stimd,delayd,side,resp,t1,t2,t3,t4,seeds = make_synth(Ntr, dt)
    theta = [0.5,0.5,0.5, 0.75, 0.1,0.5, 2.0,-1.0,0.2]

    # Warmup (JIT)
    _ = nll_trials_profile(stimd,delayd,side,resp,t1,t2,t3,t4,theta,M,dt,alpha,th,th,th,seeds; rng_mode=:crn)

    t0 = time()
    nll, stats = nll_trials_profile(stimd,delayd,side,resp,t1,t2,t3,t4,theta,M,dt,alpha,th,th,th,seeds; rng_mode=:crn)
    t1 = time()

    @printf("NLL = %.6f\n", nll)
    @printf("\n==== PERF (acumulado) ====\n")
    @printf("Wall total NLL:   %8.3f s\n", (t1 - t0))
    @printf("Trials (sum):     %8.3f s\n", stats.trials_sum)
    @printf("  fill S/U:       %8.3f s\n", stats.fill_su)
    @printf("  Heun RNG:       %8.3f s\n", stats.heun_rng)
    @printf("  Heun Noise:     %8.3f s\n", stats.heun_noise)
    @printf("  Heun Drift:     %8.3f s\n", stats.heun_drift)
    @printf("  Heun Update:    %8.3f s\n", stats.heun_update)
end

main()