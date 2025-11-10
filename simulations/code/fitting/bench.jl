#!/usr/bin/env -S julia -t 8
using CSV, DataFrames, Random, Printf, Dates
using Statistics: quantile, mean, std
using PrettyTables, DataFrames, StatsPlots
import StatsPlots: @df, scatter, plot, heatmap, savefig
include(joinpath(@__DIR__, "SimNLL.jl"))
using .SimNLL   # el punto indica “módulo relativo” al Main

# ================= CONFIGURACIÓN =================

const SUBJECT  = "A89"
const DT       = 0.1 / 40.0
const ALPHA    = 0.5
const THETA    = [0.5, 0.5, 0.5, 0.75, 0.1, 0.5, 2.0, -1.0, 0.2]
const THS      = (0.5, 0.5, 0.5)
const RNGMODE  = SimNLL.rng_mode.Independent
const SEED     = 12345
Random.seed!(SEED)

include(joinpath(@__DIR__, "..", "paths.jl"))
using .Paths  # solo si paths.jl define "module Paths"
const CSV_PATH = Paths.DATA_PATH * "/df_filtered.csv"

println("== Benchmark Trials × M para sujeto $SUBJECT ==")
println("Threads: $(Threads.nthreads())  |  $(Dates.now())")

# ================= CARGA Y FILTRADO =================
df_all = CSV.read(CSV_PATH, DataFrame; delim=';')
p95 = quantile(df_all.timepoint_4, 0.95)
df = filter(row -> row.subject == SUBJECT && row.timepoint_4 <= p95 && row.ttype_c != "SIL", df_all)

if nrow(df) == 0
    error("⚠️ No hay datos para $SUBJECT.")
end


df.cond = string.(df.stimd_c, "_", df.ttype_c)

# Muestreo estratificado igual que en Python
conds = unique(df.cond)
samples = DataFrame[]
for c in conds
    cond_df = filter(:cond => ==(c), df)
    n_sample = min(nrow(cond_df), 10_000 ÷ length(conds))
    push!(samples, cond_df[randperm(nrow(cond_df))[1:n_sample], :])
end
# df = vcat(samples...)
df = df
println("Total trials después del muestreo: ", nrow(df))

# ================= CODIFICACIÓN =================
stim_map  = Dict("VG"=>0,"SS"=>1,"SM"=>2,"SL"=>3,"SIL"=>4)
side_map  = Dict("L"=>0,"C"=>1,"R"=>2,"SIL"=>3)
resp_map  = Dict("L"=>0,"C"=>1,"R"=>2)
delay_map = Dict("DS"=>0,"DM"=>1,"DL"=>2)

stimd  = Int8.(get.(Ref(stim_map), df.stimd_c, missing))
side   = Int8.(get.(Ref(side_map), df.x_c, missing))
resp   = Int8.(get.(Ref(resp_map), df.r_c, missing))

delayd = fill(Int8(0), nrow(df))
mask_ss_sm = (df.stimd_c .== "SS") .| (df.stimd_c .== "SM")
delayd[mask_ss_sm] .= Int8.(get.(Ref(delay_map), df.ttype_c[mask_ss_sm], 0))

t1 = Float64.(df.timepoint_1)
t2 = Float64.(df.timepoint_2)
t3 = Float64.(df.timepoint_3)
t4 = Float64.(df.timepoint_4)

seeds = rand(Random.Xoshiro(SEED), UInt64, nrow(df))

# ================= BARRIDO N_TRIALS × M_REPS =================
trial_fracs = [0.05,0.1, 0.25, 0.5]
M_values = [100, 200, 400, 800, 900, 1000]

results = NamedTuple[]

for frac in trial_fracs
    ntr = Int(round(frac * nrow(df)))

    # --- Muestreo estratificado por condición ---
    idx_all = Int[]
    for g in groupby(df, :cond)
        rows = parentindices(g)[1]          # índices en el DataFrame padre
        n_cond = length(rows)
        n_take = clamp(round(Int, frac * n_cond), 1, n_cond)

        sel = rows[randperm(n_cond)[1:n_take]]
        append!(idx_all, sel)
    end

    shuffle!(idx_all)
    idx = idx_all[1:min(ntr, length(idx_all))]

    for M in M_values
        t0 = time()
        nll = SimNLL.nll_trials(
            stimd[idx], delayd[idx], side[idx], resp[idx],
            t1[idx], t2[idx], t3[idx], t4[idx],
            THETA, M, DT, ALPHA, THS[1], THS[2], THS[3],
            seeds[idx]; rng_mode=RNGMODE
        )
        tsec = time() - t0
        push!(results, (ntrials=ntr, M=M, time=tsec, nll=nll))
        println("N=$ntr  M=$M  |  time=$(round(tsec, digits=3))s  NLL=$(round(nll, digits=3))")
    end
end

println("\n== Resumen ordenado por tiempo ==")
df_results = DataFrame(results)
df_results.nll_per_trial = df_results.nll ./ df_results.ntrials

pretty_table(df_results;
    column_labels = ["Trials (N)", "M reps", "Time (s)", "NLL total", "NLL/trial"],
    formatters =[fmt__printf("%5.3f", [3]),fmt__printf("%5.6f", [4]),fmt__printf("%5.4f", [5])],
    alignment = [:r, :r, :r, :r, :r],
)


function rel_change(v::AbstractVector{<:Real})
    n = length(v)
    w = Vector{Union{Missing, Float64}}(undef, n)
    w[1] = missing
    @inbounds for i in 2:n
        a = v[i-1]; b = v[i]
        if isfinite(a) && isfinite(b) && a != 0.0
            w[i] = abs(b - a) / abs(a)
        else
            w[i] = missing
        end
    end
    return w
end


df2 = sort(df_results, [:ntrials, :M])
transform!(groupby(df2, :ntrials),
    :nll_per_trial => rel_change => :stability_M
)

pretty_table(df2;
    column_labels = ["N","M","time", "NLL total","NLL/trial","Δ_rel(M)"],
    formatters = [fmt__printf("%5.3f", [3]),fmt__printf("%5.6f", [4]),fmt__printf("%5.4f", [5])],
)

using Plots

p = @df df_results scatter(
    :M, :nll_per_trial, group=:ntrials,
    xlabel="M (Monte Carlo reps)",
    ylabel="NLL por trial",
    legend=:topright,
    lw=2, ms=6,
    title="Estabilidad de la NLL con N y M"
)
display(p)

p = @df df_results plot(
    :M, :time,
    group=:ntrials,
    lw=2, marker=:circle,
    xlabel="M", ylabel="Tiempo (s)", legend=:topleft,
    title="Escalado temporal"
)
display(p)

CSV.write("bench_trials_M2.csv", df2)
println("\nCSV escrito en bench_trials_M.csv ✅")


df2.efficiency = 1.0 ./ df2.stability_M ./ df2.time

df_eff = dropmissing(df2, :efficiency)

p = @df df_eff scatter(
    :M, :efficiency, group=:ntrials,
    xlabel="M (Monte Carlo reps)",
    ylabel="Eficiencia relativa (1 / (Δ_rel × tiempo))",
    legend=:topleft,
    lw=2, ms=6,
    title="Eficiencia de simulación según N y M"
)
display(p)

# También se puede mostrar como heatmap
dfg = combine(groupby(df_eff, [:ntrials, :M]),
              :efficiency => mean => :eff)

xs = sort!(unique(dfg.ntrials))
ys = sort!(unique(dfg.M))

Z = fill(NaN, length(xs), length(ys))
for r in eachrow(dfg)
    i = searchsortedfirst(xs, r.ntrials)
    j = searchsortedfirst(ys, r.M)
    Z[i, j] = r.eff
end

# Escala de color robusta ignorando NaN
vals = skipmissing(vec(Z))
cl_hi = quantile(collect(vals), 0.95)

# Algunos backends esperan Z' para que (x,y) se mapee a columnas/filas
p = heatmap(xs, ys, Z';
        xlabel = "N (trials)",
        ylabel = "M (MC reps)",
        title = "Mapa de eficiencia (mayor = mejor)",
        colorbar_title = "eff",
        clims = (0, cl_hi))

display(p)