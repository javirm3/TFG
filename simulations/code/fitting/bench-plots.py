import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Carga y saneo de datos --------
df = pd.read_csv("bench_trials_M.csv")
print(df['ntrials'].unique())
# Normaliza nombres que vienen de Julia si hiciera falta
colmap = {
    "N": "ntrials",
    "Time (s)": "time",
    "NLL total": "nll",
    "NLL/trial": "nll_per_trial",
    "Δ_rel(M)": "stability_M",
    "Δ_rel": "stability_M",
}
df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

# Tipos numéricos seguros
for c in ["ntrials", "M", "time", "nll", "nll_per_trial", "stability_M"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Calcula eficiencia si no existe
if "efficiency" not in df.columns:
    if {"stability_M", "time"}.issubset(df.columns):
        df["efficiency"] = 1.0 / (df["stability_M"] * df["time"])
    else:
        df["efficiency"] = np.nan

# Orden estándar
df = df.sort_values(["ntrials", "M"], kind="mergesort").reset_index(drop=True)

# --------- Gráfico: NLL/trial vs M (por N) ----------
if {"M", "nll_per_trial", "ntrials"}.issubset(df.columns):
    plt.figure()
    for N, g in df.groupby("ntrials", sort=True):
        g = g.sort_values("M")
        plt.plot(g["M"].values, g["nll_per_trial"].values, marker="o", label=f"N={N}")
    plt.xlabel("M (Monte Carlo reps)")
    plt.ylabel("NLL por trial")
    plt.title("Estabilidad de la NLL con N y M")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

# --------- Gráfico: Tiempo vs M (por N) ----------
if {"M", "time", "ntrials"}.issubset(df.columns):
    plt.figure()
    for N, g in df.groupby("ntrials", sort=True):
        g = g.sort_values("M")
        plt.plot(g["M"].values, g["time"].values, marker="o", label=f"N={N}")
    plt.xlabel("M (Monte Carlo reps)")
    plt.ylabel("Tiempo (s)")
    plt.title("Escalado temporal")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

# --------- Gráfico: Δ_rel(M) vs M (por N) ----------
if {"M", "stability_M", "ntrials"}.issubset(df.columns):
    plt.figure()
    for N, g in df.groupby("ntrials", sort=True):
        g = g.sort_values("M")
        plt.plot(g["M"].values, g["stability_M"].values, marker="o", label=f"N={N}")
    plt.xlabel("M (Monte Carlo reps)")
    plt.ylabel("Δ_rel(M)")
    plt.title("Variación relativa entre M consecutivos (menor es mejor)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

# --------- Gráfico: Eficiencia vs M (por N) ----------
if {"M", "efficiency", "ntrials"}.issubset(df.columns):
    plt.figure()
    for N, g in df.groupby("ntrials", sort=True):
        g = g.sort_values("M")
        plt.plot(g["M"].values, g["efficiency"].values, marker="o", label=f"N={N}")
    plt.xlabel("M (Monte Carlo reps)")
    plt.ylabel("Eficiencia relativa  (1 / (Δ_rel × tiempo))")
    plt.title("Eficiencia de simulación según N y M")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

# --------- Heatmap de eficiencia media por (N, M) ----------
if {"ntrials", "M", "efficiency"}.issubset(df.columns):
    dfg = (
        df[["ntrials", "M", "efficiency"]]
        .dropna(subset=["efficiency"])
        .groupby(["ntrials", "M"], as_index=False)
        .agg(eff=("efficiency", "mean"))
    )

    if not dfg.empty:
        xs = np.sort(dfg["ntrials"].unique())
        ys = np.sort(dfg["M"].unique())
        Z = np.full((xs.size, ys.size), np.nan, dtype=float)

        # Rellena matriz
        midx = {(int(r.ntrials), int(r.M)): r.eff for _, r in dfg.iterrows()}
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                Z[i, j] = midx.get((int(x), int(y)), np.nan)

        # Límite superior robusto (p95)
        vals = Z[~np.isnan(Z)]
        if vals.size > 0:
            cl_hi = np.quantile(vals, 0.95)
        else:
            cl_hi = 1.0

        plt.figure()
        # Nota: imshow usa [rows, cols] -> [x_index, y_index]
        im = plt.imshow(
            Z.T,  # Transponemos para que el eje X sea N y el Y sea M
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=cl_hi if np.isfinite(cl_hi) and cl_hi > 0 else None,
        )
        plt.colorbar(im, label="Eficiencia media")
        plt.xticks(ticks=np.arange(xs.size), labels=[str(int(v)) for v in xs], rotation=0)
        plt.yticks(ticks=np.arange(ys.size), labels=[str(int(v)) for v in ys])
        plt.xlabel("N (trials)")
        plt.ylabel("M (MC reps)")
        plt.title("Mapa de eficiencia (mayor = mejor)")
        plt.grid(False)
        plt.show()

# --------- (Opcional) Frontera tiempo–calidad por N ----------
# Si quieres ver el compromiso tiempo vs NLL/trial para cada N.
if {"time", "nll_per_trial", "ntrials"}.issubset(df.columns):
    plt.figure()
    for N, g in df.groupby("ntrials", sort=True):
        g = g.sort_values("time")
        plt.plot(g["time"].values, g["nll_per_trial"].values, marker="o", label=f"N={N}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("NLL por trial")
    plt.title("Compromiso tiempo–calidad por N (curvas M)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()
