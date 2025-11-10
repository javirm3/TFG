#!/usr/bin/env julia
# Ejecuta:  JULIA_NUM_THREADS=28 julia -O3 main.jl [Ntrials] [M]
using BenchmarkTools, Profile, StatProfilerHTML

push!(LOAD_PATH, joinpath(@__DIR__, "./"))
using SimNLL
using Random, Statistics, Printf, Dates

Ntr  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
M    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
dt   = 0.1/40.0
alpha = 0.5
th1 = 0.5; th2 = 0.5; th3 = 0.5

# ==== Datos sintéticos representativos ====
stimd = fill(Int8(1), Ntr)        # SS
delayd = fill(Int8(1), Ntr)       # DM
side  = rand(Int8[0,1,2], Ntr)    # L/C/R
resp  = rand(Int8[0,1,2], Ntr)
t1 = fill(1.0, Ntr); t2 = fill(2.0, Ntr); t3 = fill(3.0, Ntr); t4 = fill(5.0, Ntr)

theta = [0.5, 0.5, 0.5, 0.75, 0.1, 0.5, 2.0, -1.0, 0.2]
SEED = UInt64(12345)
rng = Random.Xoshiro(SEED)
seeds = rand(rng, UInt64, Ntr)

println("== SimNLL main ==")
println("Trials=$Ntr  M=$M  dt=$(dt)  threads=$(Threads.nthreads())  $(Dates.now())")
if Threads.nthreads() == 1
    @warn "Estás corriendo con 1 hilo. Usa:  julia -t 14 -O3 main.jl ..."
end

# --- Warm-up (compila sin medir) ---
Nw = min(Ntr, 2000)   # compila con el mismo tipo; tamaño da igual
_ = nll_trials(stimd[1:Nw], delayd[1:Nw], side[1:Nw], resp[1:Nw],
               t1[1:Nw], t2[1:Nw], t3[1:Nw], t4[1:Nw],
               theta, M, dt, alpha, th1, th2, th3, seeds[1:Nw];
               rng_mode = SimNLL.rng_mode.CRN)

# --- Medición real ---
t0 = time()
nll = nll_trials(stimd, delayd, side, resp, t1,t2,t3,t4,
                 theta, M, dt, alpha, th1, th2, th3, seeds;
                 rng_mode = SimNLL.rng_mode.CRN)
t1_ = time()≤
@printf "NLL = %.6f\n" nll
@printf "Tiempo evaluación: %.3f s\n" (t1_ - t0)

using BenchmarkTools

bench() = nll_trials(stimd,delayd,side,resp,t1,t2,t3,t4,
                     theta,M,dt,alpha,th1,th2,th3,seeds;
                     rng_mode=SimNLL.rng_mode.CRN)

bench()              # warm-up
@btime bench(); 

using StatProfilerHTML

Profile.clear()
@profile begin
    nll_trials(stimd,delayd,side,resp,t1,t2,t3,t4,
               theta,M,dt,alpha,th1,th2,th3,seeds;
               rng_mode=SimNLL.rng_mode.CRN)
end

statprofilehtml()