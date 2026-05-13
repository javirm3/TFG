# fit_mnle_bads_100_restarts_with_trials.py

import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import torch
from pybads import BADS


STORAGE_PATH = "../../../../storage/javi/data"
CSV_PATH = f"{STORAGE_PATH}/df_filtered_7B.csv"
MNLE_PKL = "mnle_trained_network.pkl"   # (estimator, trainer)

OUT_DIR = "./mnle_bads_fits"
SPLIT_DIR = os.path.join(OUT_DIR, "cv_splits")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

N_RESTARTS = 100
SEED_BASE = 12345
CHUNK = int(os.environ.get("MNLE_CHUNK", "20000"))

SUBJECTS_ENV = os.getenv("SUBJECTS", "").strip()

MAX_TRIALS = int(os.getenv("MAX_TRIALS", "0"))  # 0 = no limitar
BALANCE_BY_COND = int(os.getenv("BALANCE_BY_COND", "0")) == 1

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")

def validate_and_encode(df,
                        stim_col='stimd_c', delay_col='ttype_c',
                        side_col='x_c', resp_col='r_c'):
    stim_map  = {'VG':0,'SS':1,'SM':2,'SL':3,'SIL':4}
    side_map  = {'L':0,'C':1,'R':2,'SIL':3}
    resp_map  = {'L':0,'C':1,'R':2}
    delay_map = {'DS':0,'DM':1,'DL':2}

    for col in (stim_col, delay_col, side_col, resp_col):
        if col in df.columns:
            df[col] = df[col].astype('string').str.strip().str.upper()

    stim_series  = df[stim_col]
    delay_series = df[delay_col]
    side_series  = df[side_col]
    resp_series  = df[resp_col]

    stimd = stim_series.map(stim_map).astype('Int64')
    side  = side_series.map(side_map).astype('Int64')
    resp  = resp_series.map(resp_map).astype('Int64')

    delayd = np.zeros(len(df), dtype=np.int64)
    mask_delay_needed = stim_series.isin(['SS','SM'])
    delayd[mask_delay_needed.values] = (
        delay_series[mask_delay_needed].map(delay_map)
        .astype('Int64').to_numpy(dtype=np.int64)
    )

    tp_cols = ['timepoint_1','timepoint_2','timepoint_3','timepoint_4']
    if df[tp_cols].isna().any().any():
        raise ValueError("NaNs en timepoints.")
    if (df['timepoint_4'] <= 0).any():
        raise ValueError("Hay trials con timepoint_4 <= 0.")

    return (
        stimd.to_numpy(dtype=np.int8),
        delayd.astype(np.int8),
        side.to_numpy(dtype=np.int8),
        resp.to_numpy(dtype=np.int8),
        df['timepoint_1'].to_numpy(dtype=np.float32),
        df['timepoint_2'].to_numpy(dtype=np.float32),
        df['timepoint_3'].to_numpy(dtype=np.float32),
        df['timepoint_4'].to_numpy(dtype=np.float32),
        df
    )

def build_cond_np(stimd, delayd, side, t1, t2, t3, t4) -> np.ndarray:
    # (N,7) float32
    return np.column_stack([
        stimd.astype(np.float32),
        delayd.astype(np.float32),
        side.astype(np.float32),
        t1.astype(np.float32),
        t2.astype(np.float32),
        t3.astype(np.float32),
        t4.astype(np.float32),
    ]).astype(np.float32, copy=False)


@torch.no_grad()
def mnle_nll_for_theta(estimator, theta_free_np, cond_np, rt_obs, choice_obs, chunk=20000):
    device = next(estimator.parameters()).device
    N = cond_np.shape[0]

    theta_free_t = torch.tensor(theta_free_np, dtype=torch.float32, device=device).view(1, -1)
    cond_t = torch.tensor(cond_np, dtype=torch.float32, device=device)
    x = torch.cat([theta_free_t.repeat(N, 1), cond_t], dim=1)

    y = torch.stack([
        torch.tensor(rt_obs, dtype=torch.float32, device=device),
        torch.tensor(choice_obs, dtype=torch.float32, device=device),
    ], dim=1)

    nll = 0.0
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        lp = estimator.log_prob(y[start:end].unsqueeze(0), condition=x[start:end]).squeeze(0)
        nll += float((-lp).sum().detach().cpu().numpy())
    return nll


def select_trials_for_subject(df_all, subject, rng=None):
    """
    Devuelve df_fit (subset) + train_ids (trial_id).
    - Si BALANCE_BY_COND=0: uses all the trials
    - If BALANCE_BY_COND=1: balanced sample by stimd_c and ttype_c
    """
    df_s = df_all[df_all["subject"] == subject].copy()
    if df_s.empty:
        return None, None

    if rng is None:
        rng = np.random.default_rng(0)

    if BALANCE_BY_COND:
        df_s["cond"] = df_s["stimd_c"].astype(str) + "_" + df_s["ttype_c"].astype(str)
        conds = df_s["cond"].unique().tolist()
        target = MAX_TRIALS if MAX_TRIALS > 0 else 10000
        per_cond = max(1, target // max(1, len(conds)))

        parts = []
        for c in conds:
            d = df_s[df_s["cond"] == c]
            n = min(len(d), per_cond)

            parts.append(d.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
        df_fit = pd.concat(parts, ignore_index=True)
        df_fit = df_fit.drop(columns=["cond"])
    else:
        df_fit = df_s
        if MAX_TRIALS > 0 and len(df_fit) > MAX_TRIALS:
            df_fit = df_fit.sample(n=MAX_TRIALS, random_state=int(rng.integers(0, 2**31 - 1)))

    train_ids = df_fit["trial_id"].to_numpy(dtype=np.int64)
    return df_fit, train_ids


def fit_subject(estimator, df_all, subject, bounds, n_restarts=100, seed_base=12345, chunk=20000):
    rng = np.random.default_rng(seed_base + (abs(hash(subject)) % 10_000))

    df_fit, train_ids = select_trials_for_subject(df_all, subject, rng=rng)
    if df_fit is None or df_fit.empty:
        print(f"⚠️ No data for subject {subject}")
        return None

    train_path = os.path.join(SPLIT_DIR, f"{subject}_train_ids.npy")
    np.save(train_path, train_ids)


    stimd, delayd, side, resp, t1, t2, t3, t4, _ = validate_and_encode(df_fit)
    cond_np_all = build_cond_np(stimd, delayd, side, t1, t2, t3, t4)    

    lb, ub, plb, pub = (bounds["lb"].astype(float), bounds["ub"].astype(float),
                        bounds["plb"].astype(float), bounds["pub"].astype(float))

    rows = []
    for r in range(n_restarts):

        def obj(theta_free):
            th = np.asarray(theta_free, dtype=np.float32)
            return mnle_nll_for_theta(estimator, th, cond_np_all, t4, resp, chunk=chunk)

        t0 = time.perf_counter()
        bads = BADS(
            fun=obj,
            lower_bounds=lb, upper_bounds=ub,
            plausible_lower_bounds=plb, plausible_upper_bounds=pub,
            options={"display": "off", "uncertainty_handling": False},
        )
        res = bads.optimize()
        t1_ = time.perf_counter()

        rows.append({
            "subject": subject,
            "restart_id": int(r),
            "n_trials": int(len(cond_np_all)),
            "train_ids_path": train_path,   # referencia al fichero con trial_id
            "x0": res.x0.tolist(),
            "x_hat": np.asarray(res.x).tolist(),
            "fval": float(res.fval),
            "func_count": int(getattr(res, "func_count", -1)),
            "iterations": int(getattr(res, "iterations", -1)),
            "total_time_opt": float(getattr(res, "total_time", np.nan)),
            "wall_time_sec": float(t1_ - t0),
        })

        if (r + 1) % 10 == 0 or r == 0:
            best = min(rows, key=lambda d: d["fval"])
            print(f"[{subject}] {r+1}/{n_restarts} | fval={rows[-1]['fval']:.4g} | best={best['fval']:.4g}")

    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================
def main():
    df_all = pd.read_csv(CSV_PATH, sep=";")
    df_all = df_all[df_all["r_c"].notna()].copy()
    df_all = df_all.reset_index(drop=True)

    df_all["trial_id"] = np.arange(len(df_all), dtype=np.int64)

    if SUBJECTS_ENV:
        subjects = [s.strip() for s in SUBJECTS_ENV.split(",") if s.strip()]
    else:
        subjects = sorted(df_all["subject"].unique().tolist())

    print("Subjects:", subjects)
    print("BALANCE_BY_COND:", BALANCE_BY_COND, "| MAX_TRIALS:", MAX_TRIALS)

    with open(MNLE_PKL, "rb") as f:
        estimator, _trainer = pickle.load(f)
    estimator.to(DEVICE)
    estimator.eval()
    print("Loaded MNLE estimator.")

    bounds = {
        "lb":  np.array([-3.0, -3.0,  0.0, -3.0,  0.0], dtype=float),
        "ub":  np.array([ 3.0,  3.0,  3.0,  3.0, 10.0], dtype=float),
        "plb": np.array([-1.0, -1.0,  0.05, -1.0,  0.1], dtype=float),
        "pub": np.array([ 1.0,  1.0,  2.0,  1.0,  5.0], dtype=float),
    }

    all_dfs = []
    for s in subjects:
        print(f"\n=== Fitting subject {s} | restarts={N_RESTARTS} ===")
        df_res = fit_subject(
            estimator=estimator,
            df_all=df_all,
            subject=s,
            bounds=bounds,
            n_restarts=N_RESTARTS,
            seed_base=SEED_BASE,
            chunk=CHUNK
        )
        if df_res is None:
            continue

        out_csv_s = os.path.join(OUT_DIR, f"fits_{s}_restarts_{N_RESTARTS}.csv")
        df_res.to_csv(out_csv_s, index=False)
        print("[SAVED]", out_csv_s)

        all_dfs.append(df_res)

    if not all_dfs:
        print("No fits produced.")
        return

    df_all_res = pd.concat(all_dfs, ignore_index=True)

    out_csv = os.path.join(OUT_DIR, f"fits_ALL_restarts_{N_RESTARTS}.csv")
    df_all_res.to_csv(out_csv, index=False)
    print("[SAVED]", out_csv)

    out_pkl = os.path.join(OUT_DIR, f"fits_ALL_restarts_{N_RESTARTS}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(df_all_res, f)
    print("[SAVED]", out_pkl)

    best = df_all_res.sort_values("fval").groupby("subject", as_index=False).head(1)
    out_best = os.path.join(OUT_DIR, f"best_per_subject_restarts_{N_RESTARTS}.csv")
    best.to_csv(out_best, index=False)
    print("[SAVED]", out_best)


if __name__ == "__main__":
    main()