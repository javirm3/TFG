"""Python bridge for the optogenetic Julia simulator."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_JL = None
_SIM = None


def _init_julia():
    global _JL, _SIM

    if _SIM is not None:
        return _SIM

    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    try:
        from juliacall import Main as jl
    except Exception as exc:  # pragma: no cover - depends on local Julia setup
        raise RuntimeError(
            "Julia backend requires the Python package `juliacall` and a working "
            "Julia installation. Use the Numba backend if Julia is not configured."
        ) from exc

    sim_path = Path(__file__).with_name("opto_sim_julia.jl")
    jl.include(str(sim_path))
    jl.seval("using .OptoSimJulia")
    _JL = jl
    _SIM = jl.OptoSimJulia
    return _SIM


def simulate_frac_correct_julia(
    subject_arrays: dict[str, np.ndarray],
    theta: np.ndarray,
    *,
    reps: int,
    dt: float,
    thresholds: tuple[float, float, float],
    opto_target: int,
    opto_amp: float,
    use_spatial: bool,
    seed: int = 12345,
) -> float:
    """Return mean trial-level fraction correct from Julia CPU simulation."""

    sim = _init_julia()
    theta = np.ascontiguousarray(np.asarray(theta, dtype=np.float32))
    th1, th2, th3 = thresholds
    return float(
        sim.simulate_frac_correct_opto(
            np.ascontiguousarray(subject_arrays["stimd"], dtype=np.int8),
            np.ascontiguousarray(subject_arrays["delayd"], dtype=np.int8),
            np.ascontiguousarray(subject_arrays["side"], dtype=np.int8),
            np.ascontiguousarray(subject_arrays["t1"], dtype=np.float32),
            np.ascontiguousarray(subject_arrays["t2"], dtype=np.float32),
            np.ascontiguousarray(subject_arrays["t3"], dtype=np.float32),
            np.ascontiguousarray(subject_arrays["t4"], dtype=np.float32),
            theta,
            int(reps),
            np.float32(dt),
            np.float32(th1),
            np.float32(th2),
            np.float32(th3),
            int(opto_target),
            np.float32(opto_amp),
            bool(use_spatial),
            np.uint64(seed),
        )
    )


def simulate_choice_probs_julia(
    subject_arrays: dict[str, np.ndarray],
    theta: np.ndarray,
    *,
    reps: int,
    dt: float,
    thresholds: tuple[float, float, float],
    opto_target: int,
    opto_amp: float,
    use_spatial: bool,
    seed: int = 12345,
) -> np.ndarray:
    """Return mean simulated choice probabilities [pL, pC, pR]."""

    sim = _init_julia()
    theta = np.ascontiguousarray(np.asarray(theta, dtype=np.float32))
    th1, th2, th3 = thresholds
    probs = sim.simulate_choice_probs_opto(
        np.ascontiguousarray(subject_arrays["stimd"], dtype=np.int8),
        np.ascontiguousarray(subject_arrays["delayd"], dtype=np.int8),
        np.ascontiguousarray(subject_arrays["side"], dtype=np.int8),
        np.ascontiguousarray(subject_arrays["t1"], dtype=np.float32),
        np.ascontiguousarray(subject_arrays["t2"], dtype=np.float32),
        np.ascontiguousarray(subject_arrays["t3"], dtype=np.float32),
        np.ascontiguousarray(subject_arrays["t4"], dtype=np.float32),
        theta,
        int(reps),
        np.float32(dt),
        np.float32(th1),
        np.float32(th2),
        np.float32(th3),
        int(opto_target),
        np.float32(opto_amp),
        bool(use_spatial),
        np.uint64(seed),
    )
    return np.asarray(probs, dtype=np.float64)


def simulate_choice_probs_trials_julia(
    subject_arrays: dict[str, np.ndarray],
    theta: np.ndarray,
    *,
    reps: int,
    dt: float,
    thresholds: tuple[float, float, float],
    opto_target: int,
    opto_amp: float,
    use_spatial: bool,
    seed: int = 12345,
) -> np.ndarray:
    """Return trial-wise simulated choice probabilities with columns [pL, pC, pR]."""

    sim = _init_julia()
    theta = np.ascontiguousarray(np.asarray(theta, dtype=np.float32))
    th1, th2, th3 = thresholds
    probs = sim.simulate_choice_probs_trials_opto(
        np.ascontiguousarray(subject_arrays["stimd"], dtype=np.int8),
        np.ascontiguousarray(subject_arrays["delayd"], dtype=np.int8),
        np.ascontiguousarray(subject_arrays["side"], dtype=np.int8),
        np.ascontiguousarray(subject_arrays["t1"], dtype=np.float32),
        np.ascontiguousarray(subject_arrays["t2"], dtype=np.float32),
        np.ascontiguousarray(subject_arrays["t3"], dtype=np.float32),
        np.ascontiguousarray(subject_arrays["t4"], dtype=np.float32),
        theta,
        int(reps),
        np.float32(dt),
        np.float32(th1),
        np.float32(th2),
        np.float32(th3),
        int(opto_target),
        np.float32(opto_amp),
        bool(use_spatial),
        np.uint64(seed),
    )
    return np.asarray(probs, dtype=np.float64)


def backend_info() -> dict[str, str]:
    sim_path = Path(__file__).with_name("opto_sim_julia.jl")
    return {"simulator": str(sim_path), "bridge": str(Path(__file__))}
