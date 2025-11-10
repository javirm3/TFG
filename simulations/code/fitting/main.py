#!/usr/bin/env python3
import time
import numpy as np
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import paths 

paths.show_paths()
from helpers.sim_core_numba2 import nll_trials_numba
from numba import get_num_threads, set_num_threads, threading_layer
import os

print("Antes de fijar:")
print("NUMBA_NUM_THREADS env:", os.getenv("NUMBA_NUM_THREADS"))
print("get_num_threads():    ", get_num_threads())
def main(Ntr=10_000, M=200, dt=0.1/40.0, alpha=0.5, th=0.5):
    rng = np.random.default_rng(12345)

    stimd = np.full(Ntr, 1, dtype=np.int8)      # SS
    delayd= np.full(Ntr, 1, dtype=np.int8)      # DMx
    side  = rng.integers(0, 3, size=Ntr, dtype=np.int8)
    resp  = rng.integers(0, 3, size=Ntr, dtype=np.int8)

    t1 = np.full(Ntr, 1.0, dtype=np.float64)
    t2 = np.full(Ntr, 2.0, dtype=np.float64)
    t3 = np.full(Ntr, 3.0, dtype=np.float64)
    t4 = np.full(Ntr, 5.0, dtype=np.float64)

    theta = np.array([0.5,0.5,0.5, 0.75, 0.1,0.5, 2.0,-1.0,0.2], dtype=np.float64)
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=Ntr, dtype=np.uint64)

    # warmup (jit)
    _ = nll_trials_numba(stimd, delayd, side, resp, t1,t2,t3,t4,
                         theta, M, dt, alpha, th, th, th, seeds)

    t0 = time.perf_counter()
    nll = nll_trials_numba(stimd, delayd, side, resp, t1,t2,t3,t4,
                           theta, M, dt, alpha, th, th, th, seeds)
    t1p = time.perf_counter()

    print(f"NLL = {nll:.6f}")
    print(f"Tiempo evaluación: {t1p - t0:.3f} s   (Ntr={Ntr}, M={M}, dt={dt})")

if __name__ == "__main__":
    main()