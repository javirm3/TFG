import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import pickle
    import sys
    import warnings
    from pathlib import Path

    # Avoid noisy optional pandas accelerators that are broken in this env.
    sys.modules.setdefault("numexpr", None)
    sys.modules.setdefault("bottleneck", None)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return Path, mo, np, pd, plt


@app.cell
def _(Path):
    HERE = Path(__file__).resolve()
    CODE_DIR = HERE.parents[1]
    SIM_DIR = HERE.parents[2]
    DATA_PATH = SIM_DIR / "datasets" / "df_filtered.csv"
    PARAMS_DIR = SIM_DIR / "params"
    RESULT_CACHE_DIR = SIM_DIR / "simulations" / "opto_trial_results"

    DT = 0.1 / 40.0
    THRESHOLDS = (0.5, 0.5, 0.5)
    BALANCE_MAX_TRIALS = 10000
    BALANCE_COND_COLUMNS = ("stimd_c", "ttype_c")
    BALANCE_SEED = 42
    REQUIRED_COLUMNS = [
        "subject",
        "stimd_c",
        "ttype_c",
        "x_c",
        "r_c",
        "timepoint_1",
        "timepoint_2",
        "timepoint_3",
        "timepoint_4",
    ]
    return (
        BALANCE_COND_COLUMNS,
        BALANCE_MAX_TRIALS,
        BALANCE_SEED,
        CODE_DIR,
        DATA_PATH,
        DT,
        PARAMS_DIR,
        REQUIRED_COLUMNS,
        RESULT_CACHE_DIR,
        THRESHOLDS,
    )


@app.cell
def _():
    import numpy as _np
    from numba import njit, prange, set_num_threads

    @njit
    def _onset_offset_from_codes(stim_code, delay_code, t1, t2, t3, t4):
        if stim_code == 0:
            return 0.0, t4
        if stim_code == 1:
            if delay_code == 0:
                return t2, t3
            if delay_code == 1:
                return t1, t2
            return 0.0, t1
        if stim_code == 2:
            if delay_code == 0:
                return t1, t3
            return 0.0, t2
        if stim_code == 3:
            return 0.0, t3
        return 0.0, 0.0

    @njit
    def _S_value(t, amp, d, onset, offset):
        if t < onset:
            return 0.0
        if t <= offset:
            return amp
        tail_end = offset + d
        if t <= tail_end and d > 0.0:
            return amp * (1.0 - (t - offset) / d)
        return 0.0

    @njit
    def _U_value(t, amp, base, onset, offset):
        if (t >= onset) and (t <= offset):
            return amp + base
        return base

    @njit
    def _drift(x1, x2, IL, IC, IR, sL, sC, sR):
        term1_F1 = -5.0 * IC + 5.0 * IL
        term2_F1 = 20.0 * x1 * x2
        term3_F1 = -1.9047619047619 * x1 * (x1 * x1 + 3.0 * x2 * x2)
        term4_F1 = 5.0 * x1 * (sC + sL)
        term5_F1 = 10.0 * x1 * (
            0.904761904761905 * IC
            + 0.904761904761905 * IL
            - 0.0952380952380951 * IR
            + 0.226190476190476 * sC
            + 0.226190476190476 * sL
            - 0.0238095238095238 * sR
        )
        term6_F1 = -10.0 * x2 * (IC - IL + 0.25 * sC - 0.25 * sL)
        term7_F1 = -5.0 * (x2 + 0.25) * (sC - sL)
        F1 = (
            term1_F1
            + term2_F1
            + term3_F1
            + term4_F1
            + term5_F1
            + term6_F1
            + term7_F1
        )

        F2 = (
            -3.33333333333333 * IC * x1
            + 3.09523809523809 * IC * x2
            + 1.66666666666667 * IC
            + 3.33333333333333 * IL * x1
            + 3.09523809523809 * IL * x2
            + 1.66666666666667 * IL
            + 13.0952380952381 * IR * x2
            - 3.33333333333333 * IR
            - 1.9047619047619 * x1 * x1 * x2
            + 3.33333333333333 * x1 * x1
            - 10.0 * x2 * x2
            - 1.9047619047619 * x2 * (x1 * x1 + 3.0 * x2 * x2)
            + 2.44047619047619 * x2 * sC
            + 2.44047619047619 * x2 * sL
            + 9.94047619047619 * x2 * sR
            + 0.416666666666667 * sC
            + 0.416666666666667 * sL
            - 0.833333333333333 * sR
        )
        return F1, F2

    @njit
    def _single_path_heun_opto(
        S_t,
        U_t,
        side_code,
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
    ):
        x1 = 0.0
        x2 = 0.0

        for i in range(S_t.shape[0]):
            Sval = S_t[i]
            Uval = U_t[i]
            if side_code == 0:
                IL = Sval + Uval
                IC = Uval
                IR = Uval
            elif side_code == 1:
                IL = Uval
                IC = Sval + Uval
                IR = Uval
            elif side_code == 2:
                IL = Uval
                IC = Uval
                IR = Sval + Uval
            else:
                IL = Uval
                IC = Uval
                IR = Uval

            IC += opto_amp

            dW0 = _np.random.randn() * _np.sqrt(dt)
            dW1 = _np.random.randn() * _np.sqrt(dt)
            dW2 = _np.random.randn() * _np.sqrt(dt)
            dB1 = (dW0 - dW1) / 2.0
            dB2 = (dW0 + dW1 - 2.0 * dW2) / 6.0
            n1 = noise_amp * dB1
            n2 = noise_amp * dB2

            f1a, f2a = _drift(x1, x2, IL, IC, IR, sL, sC, sR)
            x1p = x1 + f1a * dt + n1
            x2p = x2 + f2a * dt + n2
            f1b, f2b = _drift(x1p, x2p, IL, IC, IR, sL, sC, sR)
            x1 = x1 + 0.5 * (f1a + f1b) * dt + n1
            x2 = x2 + 0.5 * (f2a + f2b) * dt + n2

        r1 = x1 + x2
        r2 = -x1 + x2
        r3 = -2.0 * x2
        if r1 > r2 and r1 > r3 and r1 > th1:
            return 0
        if r2 > r1 and r2 > r3 and r2 > th2:
            return 1
        if r3 > r1 and r3 > r2 and r3 > th3:
            return 2
        return -1

    @njit(parallel=True)
    def simulate_frac_correct_opto(
        stimd,
        delayd,
        side,
        t1,
        t2,
        t3,
        t4,
        theta,
        M,
        dt,
        th1,
        th2,
        th3,
        opto_target,
        opto_amp,
    ):
        sL = theta[0]
        sC = theta[1]
        sR = theta[2]
        noise_amp = theta[3]
        S_amp = theta[4]
        dS = theta[5]
        U_amp = theta[6]
        U_base = theta[7]
        U_on = theta[8]

        Ntr = stimd.shape[0]
        correct_per_trial = _np.zeros(Ntr, dtype=_np.float64)

        for i in prange(Ntr):
            onset, offset = _onset_offset_from_codes(
                stimd[i], delayd[i], t1[i], t2[i], t3[i], t4[i]
            )
            n_steps = int(t4[i] / dt)
            if n_steps <= 0:
                correct_per_trial[i] = 0.0
                continue

            S_t = _np.empty(n_steps, dtype=_np.float64)
            U_t = _np.empty(n_steps, dtype=_np.float64)
            for k in range(n_steps):
                tt = k * dt
                S_t[k] = _S_value(tt, S_amp, dS, onset, offset)
                U_t[k] = _U_value(tt, U_amp, U_base, U_on, t4[i])

            n_correct = 0
            for _ in range(M):
                choice = _single_path_heun_opto(
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
                )
                if choice == side[i]:
                    n_correct += 1

            correct_per_trial[i] = n_correct / M

        return correct_per_trial.mean()

    def make_simulator():
        return simulate_frac_correct_opto, set_num_threads

    return (make_simulator,)


@app.cell
def _(
    BALANCE_COND_COLUMNS,
    BALANCE_MAX_TRIALS,
    BALANCE_SEED,
    CODE_DIR,
    np,
    pd,
):
    STIM_MAP = {"VG": 0, "SS": 1, "SM": 2, "SL": 3, "SIL": 4}
    DELAY_MAP = {"DS": 0, "DM": 1, "DL": 2, "VG": 0}
    DELAY_LABELS = {0: "DS", 1: "DM", 2: "DL"}
    SIDE_MAP = {"L": 0, "C": 1, "R": 2}
    SIDE_LABELS = {0: "L", 1: "C", 2: "R"}
    TARGET_LABELS = {"center": "Center", "sides": "Sides"}
    POLARITY_LABELS = {"excitation": "Excitation", "inhibition": "Inhibition"}
    MODEL_SUBDIRS = {
        "Spatial reduced 3": "spatial_reduced3",
        "Temporal": "temporalU",
    }
    THETA_COLUMNS = [
        "sL",
        "sC",
        "sR",
        "noise_amp",
        "S_amplitude",
        "S_d",
        "U_int_amplitude",
        "U_int_baseline",
        "U_int_onset",
        "U_ext_amplitude",
    ]

    def _infer_csv_sep(path):
        with open(path, "r", encoding="utf-8") as handle:
            first_line = handle.readline()
        return ";" if first_line.count(";") > first_line.count(",") else ","

    def balance_trials_by_subject_conditions(
        df,
        *,
        max_trials_per_subject=BALANCE_MAX_TRIALS,
        cond_cols=BALANCE_COND_COLUMNS,
        seed=BALANCE_SEED,
    ):
        if max_trials_per_subject is None or max_trials_per_subject <= 0:
            return df.copy()

        parts = []
        for subject_index, (subject, subject_df) in enumerate(
            df.sort_values(["subject", *cond_cols]).groupby("subject", sort=True)
        ):
            condition_keys = subject_df[list(cond_cols)].astype(str).agg("_".join, axis=1)
            n_conditions = int(condition_keys.nunique())
            if n_conditions == 0:
                parts.append(subject_df)
                continue

            per_condition = max(1, int(max_trials_per_subject) // n_conditions)
            subject_parts = []
            subject_df = subject_df.assign(_balance_condition=condition_keys.to_numpy())
            rng = np.random.default_rng(int(seed) + subject_index)
            for _, condition_df in subject_df.groupby("_balance_condition", sort=True):
                n_sample = min(len(condition_df), per_condition)
                random_state = int(rng.integers(0, 2**31 - 1))
                subject_parts.append(
                    condition_df.sample(n=n_sample, random_state=random_state)
                )

            balanced_subject = pd.concat(subject_parts, ignore_index=False)
            balanced_subject = balanced_subject.drop(columns="_balance_condition")
            parts.append(balanced_subject)

        if not parts:
            return df.iloc[0:0].copy()
        return pd.concat(parts, ignore_index=False).sort_index().reset_index(drop=True)

    def load_trials(path, required_columns):
        sep = _infer_csv_sep(path)
        df = pd.read_csv(path, sep=sep, usecols=required_columns)
        for col in ("stimd_c", "ttype_c", "x_c", "r_c", "subject"):
            df[col] = df[col].astype("string").str.strip()

        df = df[
            df["stimd_c"].isin(STIM_MAP)
            & df["ttype_c"].isin(DELAY_MAP)
            & df["x_c"].isin(SIDE_MAP)
            & df["r_c"].isin(SIDE_MAP)
            & (df["timepoint_4"] <= 5.0)
        ].copy()

        filtered_n_trials = int(len(df))
        df = balance_trials_by_subject_conditions(df)
        df.attrs["filtered_n_trials"] = filtered_n_trials
        df.attrs["balanced_n_trials"] = int(len(df))
        df.attrs["balance_max_trials_per_subject"] = int(BALANCE_MAX_TRIALS)
        df.attrs["balance_cond_columns"] = list(BALANCE_COND_COLUMNS)
        df.attrs["balance_seed"] = int(BALANCE_SEED)

        df["stim_code"] = df["stimd_c"].map(STIM_MAP).astype(np.int8)
        df["delay_code"] = df["ttype_c"].map(DELAY_MAP).astype(np.int8)
        df["side_code"] = df["x_c"].map(SIDE_MAP).astype(np.int8)
        return df

    def load_model_params(df, model_subdir):
        import contextlib
        import io
        import sys

        if str(CODE_DIR) not in sys.path:
            sys.path.insert(0, str(CODE_DIR))
        from fitting import process_params

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            params_df = process_params.process_params(df, subdirs=[model_subdir])

        params_df = params_df[params_df["model"].astype(str) == model_subdir].copy()
        if params_df.empty:
            return params_df, params_df, stdout.getvalue()

        params_df["subject"] = params_df["subject"].astype(str).str.strip()
        params_df["nll/trial"] = pd.to_numeric(params_df["nll/trial"], errors="coerce")
        best_params_df = (
            params_df.sort_values(["subject", "nll/trial"], na_position="last")
            .groupby("subject", as_index=False)
            .first()
        )
        return params_df, best_params_df, stdout.getvalue()

    def theta_from_params_row(row):
        values = []
        for column in THETA_COLUMNS:
            value = row.get(column, 0.0)
            if pd.isna(value):
                value = 0.0
            values.append(float(value))
        return np.ascontiguousarray(np.asarray(values, dtype=np.float64))

    def arrays_for_subject(df, subject):
        sub = df[df["subject"].astype(str) == subject].copy()
        return {
            "stimd": sub["stim_code"].to_numpy(dtype=np.int8),
            "delayd": sub["delay_code"].to_numpy(dtype=np.int8),
            "side": sub["side_code"].to_numpy(dtype=np.int8),
            "t1": sub["timepoint_1"].to_numpy(dtype=np.float64),
            "t2": sub["timepoint_2"].to_numpy(dtype=np.float64),
            "t3": sub["timepoint_3"].to_numpy(dtype=np.float64),
            "t4": sub["timepoint_4"].to_numpy(dtype=np.float64),
            "n_trials": int(len(sub)),
        }

    def subset_arrays_by_trial_group(subject_arrays, trial_group):
        if trial_group == "center":
            mask = subject_arrays["side"] == 1
        elif trial_group == "sides":
            mask = subject_arrays["side"] != 1
        else:
            mask = np.ones(subject_arrays["side"].shape[0], dtype=bool)

        return {
            "stimd": subject_arrays["stimd"][mask],
            "delayd": subject_arrays["delayd"][mask],
            "side": subject_arrays["side"][mask],
            "t1": subject_arrays["t1"][mask],
            "t2": subject_arrays["t2"][mask],
            "t3": subject_arrays["t3"][mask],
            "t4": subject_arrays["t4"][mask],
            "n_trials": int(mask.sum()),
        }

    def subset_arrays_by_delay_side(subject_arrays, delay_code=None, side_code=None):
        mask = np.ones(subject_arrays["side"].shape[0], dtype=bool)
        if delay_code is not None:
            mask &= subject_arrays["delayd"] == int(delay_code)
        if side_code is not None:
            mask &= subject_arrays["side"] == int(side_code)

        return {
            "stimd": subject_arrays["stimd"][mask],
            "delayd": subject_arrays["delayd"][mask],
            "side": subject_arrays["side"][mask],
            "t1": subject_arrays["t1"][mask],
            "t2": subject_arrays["t2"][mask],
            "t3": subject_arrays["t3"][mask],
            "t4": subject_arrays["t4"][mask],
            "n_trials": int(mask.sum()),
        }

    return (
        DELAY_LABELS,
        MODEL_SUBDIRS,
        POLARITY_LABELS,
        SIDE_LABELS,
        TARGET_LABELS,
        arrays_for_subject,
        load_model_params,
        load_trials,
        subset_arrays_by_trial_group,
        theta_from_params_row,
    )


@app.cell
def _():
    PLOT_COLORS = {
        "opto_off": "tab:grey",
        "excitation": "tab:red",
        "inhibition": "tab:blue",
        "paired_lines": "#A8A8A8",
        "subject_curve": "#1A1A1A",
        "zero_lines": "#8A8A8A",
        "heatmap_cmap": "viridis",
        "triangle_side_colors": {
            "L": "tab:red",
            "C": "tab:green",
            "R": "tab:blue",
        },
    }
    return (PLOT_COLORS,)


@app.cell
def _(np):
    def _resolve_boxplot_colors(colors, n, name):
        if isinstance(colors, str):
            return [colors] * n
        colors = list(colors)
        if len(colors) != n:
            raise ValueError(f"{name} must have length {n}, got {len(colors)}.")
        return colors

    def custom_boxplot(
        ax,
        values,
        *,
        positions,
        widths,
        median_colors,
        box_facecolor="white",
        box_edgecolor="#666666",
        box_alpha=1.0,
        box_linewidth=1.1,
        whisker_color="#666666",
        whisker_linewidth=1.0,
        median_linewidth=3.0,
        line_values=None,
        line_color="#B0B0B0",
        line_alpha=0.15,
        line_linewidth=1.25,
        line_zorder=2.0,
        showfliers=False,
        showcaps=False,
        zorder=0,
        **kwargs,
    ):
        positions = list(np.atleast_1d(np.asarray(positions, dtype=float)))
        resolved_median_colors = _resolve_boxplot_colors(
            median_colors,
            len(positions),
            "median_colors",
        )

        if len(positions) == 1:
            grouped_values = [np.asarray(values, dtype=float)]
        else:
            grouped_values = [np.asarray(vals, dtype=float) for vals in values]
            if len(grouped_values) != len(positions):
                raise ValueError(
                    f"values must have length {len(positions)}, got {len(grouped_values)}."
                )

        valid_triplets = [
            (vals, pos, color)
            for vals, pos, color in zip(
                grouped_values, positions, resolved_median_colors, strict=False
            )
            if vals.size > 0
        ]

        if valid_triplets:
            valid_values, valid_positions, valid_median_colors = zip(
                *valid_triplets, strict=False
            )
            box = ax.boxplot(
                list(valid_values),
                positions=list(valid_positions),
                widths=widths,
                patch_artist=True,
                showfliers=showfliers,
                showcaps=showcaps,
                zorder=zorder,
                **kwargs,
            )

            for patch in box["boxes"]:
                patch.set(
                    facecolor=box_facecolor,
                    edgecolor=box_edgecolor,
                    alpha=box_alpha,
                    linewidth=box_linewidth,
                )

            for elem in ("whiskers", "caps"):
                for artist in box[elem]:
                    artist.set(color=whisker_color, linewidth=whisker_linewidth)

            for median, color in zip(box["medians"], valid_median_colors, strict=False):
                median.set(color=color, linewidth=median_linewidth)
        else:
            box = {"boxes": [], "whiskers": [], "caps": [], "medians": []}

        if line_values is not None:
            line_values = np.asarray(line_values, dtype=float)
            if line_values.ndim == 1:
                line_values = line_values[None, :]
            line_positions = np.asarray(positions, dtype=float)
            if line_values.shape[1] != len(line_positions):
                raise ValueError(
                    "line_values must have one column per box position. "
                    f"Expected {len(line_positions)}, got {line_values.shape[1]}."
                )

            for ys in line_values:
                valid_idx = np.flatnonzero(np.isfinite(ys))
                if valid_idx.size < 2:
                    continue
                split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
                for segment in np.split(valid_idx, split_points):
                    if segment.size < 2:
                        continue
                    ax.plot(
                        line_positions[segment],
                        ys[segment],
                        color=line_color,
                        alpha=line_alpha,
                        lw=line_linewidth,
                        zorder=line_zorder,
                    )

        return box

    return (custom_boxplot,)


@app.cell
def _(mo):
    magnitude_slider = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.02,
        value=0.08,
        show_value=True,
        label="Opto current magnitude",
    )
    model_selector = mo.ui.radio(
        options={"Spatial reduced 3": "spatial_reduced3", "Temporal": "temporalU"},
        value="Spatial reduced 3",
        inline=True,
        label="Parameter model",
    )
    target_selector = mo.ui.radio(
        options={"Both": "both", "Center": "center", "Sides": "sides"},
        value="Both",
        inline=True,
        label="Trial groups",
    )
    backend_selector = mo.ui.radio(
        options={"Numba": "numba", "Julia": "julia"},
        value="Numba",
        inline=True,
        label="Simulator backend",
    )
    reps_slider = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=200,
        show_value=True,
        label="Monte Carlo reps",
    )
    thread_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=8,
        show_value=True,
        label="Numba threads",
    )
    run_button = mo.ui.run_button(label="Run simulation")
    load_button = mo.ui.run_button(label="Load saved result")

    controls = mo.vstack(
        [
            model_selector,
            backend_selector,
            mo.hstack([magnitude_slider, reps_slider]),
            mo.hstack([target_selector, thread_slider, run_button, load_button]),
        ]
    )
    controls
    return (
        backend_selector,
        load_button,
        magnitude_slider,
        model_selector,
        reps_slider,
        run_button,
        target_selector,
        thread_slider,
    )


@app.cell
def _(DATA_PATH, MODEL_SUBDIRS, PARAMS_DIR, RESULT_CACHE_DIR, mo):
    available_models = {
        label: subdir
        for label, subdir in MODEL_SUBDIRS.items()
        if (PARAMS_DIR / subdir).exists()
    }
    mo.md(
        f"""
        **Data:** `{DATA_PATH}`

        **Saved result cache:** `{RESULT_CACHE_DIR}`

        **Parameter models:** {", ".join(f"{label} (`{subdir}`)" for label, subdir in available_models.items())}

        For repeated fits in a model folder, the notebook uses the lowest `nll/trial` row per subject.
        """
    )
    return


@app.cell
def _(
    BALANCE_COND_COLUMNS,
    BALANCE_MAX_TRIALS,
    BALANCE_SEED,
    DT,
    RESULT_CACHE_DIR,
    THRESHOLDS,
    pd,
):
    import hashlib
    import json
    from datetime import datetime

    def result_config(
        model_subdir, target_selector, magnitude, reps, numba_threads, backend
    ):
        return {
            "schema_version": 4,
            "model": str(model_subdir),
            "backend": str(backend),
            "target_selector": str(target_selector),
            "opto_population": "center",
            "magnitude": round(float(magnitude), 6),
            "M": int(reps),
            "numba_threads": int(numba_threads),
            "dt": float(DT),
            "thresholds": [float(v) for v in THRESHOLDS],
            "balance_max_trials_per_subject": int(BALANCE_MAX_TRIALS),
            "balance_cond_columns": list(BALANCE_COND_COLUMNS),
            "balance_seed": int(BALANCE_SEED),
        }

    def result_key(config):
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def result_paths(config):
        key = result_key(config)
        return (
            RESULT_CACHE_DIR / f"opto_trials_{key}.csv",
            RESULT_CACHE_DIR / f"opto_trials_{key}.json",
        )

    def save_result_df(summary_df, config):
        RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        csv_path, json_path = result_paths(config)
        summary_df.to_csv(csv_path, index=False)
        metadata = {
            "config": config,
            "csv": csv_path.name,
            "rows": int(len(summary_df)),
            "n_param_rows_total": int(summary_df.attrs.get("n_param_rows_total", 0)),
            "n_subjects": int(summary_df.attrs.get("n_subjects", 0)),
            "n_data_subjects": int(summary_df.attrs.get("n_data_subjects", 0)),
            "data_subjects": list(summary_df.attrs.get("data_subjects", [])),
            "n_trials_filtered": int(summary_df.attrs.get("n_trials_filtered", 0)),
            "n_trials_balanced": int(summary_df.attrs.get("n_trials_balanced", 0)),
            "balance_max_trials_per_subject": int(
                summary_df.attrs.get("balance_max_trials_per_subject", 0)
            ),
            "balance_cond_columns": list(
                summary_df.attrs.get("balance_cond_columns", [])
            ),
            "balance_seed": int(summary_df.attrs.get("balance_seed", 0)),
            "progress_total_steps": int(summary_df.attrs.get("progress_total_steps", 0)),
            "progress_steps_per_subject": int(
                summary_df.attrs.get("progress_steps_per_subject", 0)
            ),
            "subjects": sorted(summary_df["subject"].astype(str).unique().tolist())
            if "subject" in summary_df
            else [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return csv_path, json_path

    def load_result_df(config):
        csv_path, json_path = result_paths(config)
        if not csv_path.exists() or not json_path.exists():
            return None, {
                "loaded": False,
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            }
        summary_df = pd.read_csv(csv_path)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        summary_df.attrs["loaded_from"] = str(csv_path)
        summary_df.attrs["config"] = metadata.get("config", config)
        summary_df.attrs["model"] = metadata.get("config", {}).get("model", config["model"])
        summary_df.attrs["n_param_rows_total"] = int(metadata.get("n_param_rows_total", 0))
        summary_df.attrs["n_subjects"] = int(metadata.get("n_subjects", 0))
        summary_df.attrs["n_data_subjects"] = int(metadata.get("n_data_subjects", 0))
        summary_df.attrs["data_subjects"] = list(metadata.get("data_subjects", []))
        summary_df.attrs["n_trials_filtered"] = int(
            metadata.get("n_trials_filtered", 0)
        )
        summary_df.attrs["n_trials_balanced"] = int(
            metadata.get("n_trials_balanced", 0)
        )
        summary_df.attrs["balance_max_trials_per_subject"] = int(
            metadata.get("balance_max_trials_per_subject", 0)
        )
        summary_df.attrs["balance_cond_columns"] = list(
            metadata.get("balance_cond_columns", [])
        )
        summary_df.attrs["balance_seed"] = int(metadata.get("balance_seed", 0))
        summary_df.attrs["progress_total_steps"] = int(
            metadata.get("progress_total_steps", 0)
        )
        summary_df.attrs["progress_steps_per_subject"] = int(
            metadata.get("progress_steps_per_subject", 0)
        )
        metadata["loaded"] = True
        metadata["csv_path"] = str(csv_path)
        metadata["json_path"] = str(json_path)
        return summary_df, metadata

    return load_result_df, result_config, result_paths, save_result_df


@app.cell
def _(
    CODE_DIR,
    DATA_PATH,
    DT,
    POLARITY_LABELS,
    REQUIRED_COLUMNS,
    TARGET_LABELS,
    THRESHOLDS,
    arrays_for_subject,
    backend_selector,
    load_button,
    load_model_params,
    load_result_df,
    load_trials,
    magnitude_slider,
    make_simulator,
    mo,
    model_selector,
    np,
    pd,
    reps_slider,
    result_config,
    result_paths,
    run_button,
    save_result_df,
    subset_arrays_by_trial_group,
    target_selector,
    theta_from_params_row,
    thread_slider,
):
    def current_config():
        return result_config(
            model_selector.value,
            target_selector.value,
            magnitude_slider.value,
            reps_slider.value,
            thread_slider.value,
            backend_selector.value,
        )

    def empty_summary_df():
        return pd.DataFrame(
            columns=[
                "subject",
                "model",
                "backend",
                "target",
                "target_label",
                "polarity",
                "polarity_label",
                "opto",
                "opto_amp",
                "magnitude",
                "M",
                "n_trials",
                "n_param_rows",
                "param_nll_per_trial",
                "frac_correct",
            ]
        )

    def build_run_status(summary, config, source, metadata=None):
        csv_path, json_path = result_paths(config)
        metadata = metadata or {}
        if summary.empty:
            invalid_count = 0
            subjects = metadata.get("subjects", [])
        else:
            invalid_rows = summary[
                ~np.isfinite(summary["frac_correct"])
                | (summary["frac_correct"] < 0.0)
                | (summary["frac_correct"] > 1.0)
            ]
            invalid_count = int(len(invalid_rows))
            subjects = sorted(summary["subject"].astype(str).unique().tolist())

        data_subjects = list(
            summary.attrs.get("data_subjects", metadata.get("data_subjects", []))
        )
        simulated_subjects = list(
            summary.attrs.get("simulated_subjects", metadata.get("subjects", subjects))
        )
        return {
            "data_path": str(DATA_PATH),
            "model": config["model"],
            "backend": config["backend"],
            "opto_population": config["opto_population"],
            "target_selector": config["target_selector"],
            "magnitude": config["magnitude"],
            "M": config["M"],
            "numba_threads": config["numba_threads"],
            "source": source,
            "loaded": bool(metadata.get("loaded", False)),
            "cache_csv": metadata.get("csv_path", str(csv_path)),
            "cache_json": metadata.get("json_path", str(json_path)),
            "cache_exists": csv_path.exists() and json_path.exists(),
            "n_param_rows_total": int(summary.attrs.get("n_param_rows_total", 0)),
            "n_data_subjects": int(
                summary.attrs.get("n_data_subjects", len(data_subjects))
            ),
            "data_subjects": data_subjects,
            "n_trials_filtered": int(summary.attrs.get("n_trials_filtered", 0)),
            "n_trials_balanced": int(summary.attrs.get("n_trials_balanced", 0)),
            "balance_max_trials_per_subject": int(
                summary.attrs.get(
                    "balance_max_trials_per_subject",
                    config.get("balance_max_trials_per_subject", 0),
                )
            ),
            "balance_cond_columns": list(
                summary.attrs.get(
                    "balance_cond_columns", config.get("balance_cond_columns", [])
                )
            ),
            "balance_seed": int(
                summary.attrs.get("balance_seed", config.get("balance_seed", 0))
            ),
            "n_subjects": int(summary.attrs.get("n_subjects", len(simulated_subjects))),
            "subjects": simulated_subjects,
            "progress_total_steps": int(summary.attrs.get("progress_total_steps", 0)),
            "progress_steps_per_subject": int(
                summary.attrs.get("progress_steps_per_subject", 0)
            ),
            "rows": int(len(summary)),
            "invalid_rows": invalid_count,
        }

    def load_julia_backend():
        import importlib
        import sys

        simulator_dir = CODE_DIR / "simulations"
        if str(simulator_dir) not in sys.path:
            sys.path.insert(0, str(simulator_dir))
        return importlib.import_module("opto_julia_backend")

    def run_opto_sweep():
        config = current_config()
        backend = config["backend"]
        if backend == "numba":
            simulate_frac_correct_opto, set_num_threads = make_simulator()
            set_num_threads(int(thread_slider.value))
            julia_backend = None
        else:
            simulate_frac_correct_opto = None
            julia_backend = load_julia_backend()

        magnitude = float(config["magnitude"])
        reps = int(config["M"])
        model_subdir = config["model"]
        use_spatial = model_subdir.startswith("spatial")
        selected_group = config["target_selector"]
        trial_groups = ["center", "sides"] if selected_group == "both" else [selected_group]

        def simulate_backend(subject_arrays, theta, opto_amp, seed):
            if backend == "julia":
                return julia_backend.simulate_frac_correct_julia(
                    subject_arrays,
                    theta,
                    reps=reps,
                    dt=DT,
                    thresholds=THRESHOLDS,
                    opto_target=0,
                    opto_amp=opto_amp,
                    use_spatial=use_spatial,
                    seed=seed,
                )

            return float(
                simulate_frac_correct_opto(
                    subject_arrays["stimd"],
                    subject_arrays["delayd"],
                    subject_arrays["side"],
                    subject_arrays["t1"],
                    subject_arrays["t2"],
                    subject_arrays["t3"],
                    subject_arrays["t4"],
                    theta,
                    reps,
                    DT,
                    THRESHOLDS[0],
                    THRESHOLDS[1],
                    THRESHOLDS[2],
                    0,
                    opto_amp,
                )
            )

        trials_df = load_trials(DATA_PATH, REQUIRED_COLUMNS)
        params_df, best_params_df, params_log = load_model_params(trials_df, model_subdir)
        data_subjects = sorted(trials_df["subject"].dropna().astype(str).unique())
        best_params_df = best_params_df[
            best_params_df["subject"].astype(str).isin(set(data_subjects))
        ].copy()
        subjects = sorted(best_params_df["subject"].astype(str).unique())

        rows = []
        param_rows = list(best_params_df.sort_values("subject").iterrows())
        steps_per_subject = 3 * len(trial_groups)
        total_steps = len(param_rows) * steps_per_subject
        with mo.status.progress_bar(
            total=total_steps,
            title="Running opto simulations",
            subtitle=f"Model: {model_subdir}; backend: {backend}",
            completion_title="Opto simulations complete",
            completion_subtitle=f"{len(subjects)} subjects simulated",
        ) as bar:
            for subject_index, (_, param_row) in enumerate(param_rows):
                subject = str(param_row["subject"])
                subject_arrays = arrays_for_subject(trials_df, subject)
                subject_steps = 3 * len(trial_groups)
                if subject_arrays["n_trials"] == 0:
                    bar.update(
                        increment=subject_steps,
                        subtitle=f"Skipped {subject}: no trials",
                    )
                    continue

                theta = theta_from_params_row(param_row)
                condition_index = 1
                for target in trial_groups:
                    group_arrays = subset_arrays_by_trial_group(subject_arrays, target)
                    if group_arrays["n_trials"] == 0:
                        bar.update(
                            increment=3,
                            subtitle=f"Skipped {subject}: no {target} trials",
                        )
                        continue

                    baseline = simulate_backend(
                        group_arrays,
                        theta,
                        0.0,
                        12345 + 1000 * subject_index + condition_index,
                    )
                    condition_index += 1
                    bar.update(subtitle=f"{subject}: {target} baseline")

                    for polarity, signed_amp in (
                        ("excitation", magnitude),
                        ("inhibition", -magnitude),
                    ):
                        opto_on = simulate_backend(
                            group_arrays,
                            theta,
                            signed_amp,
                            12345 + 1000 * subject_index + condition_index,
                        )
                        condition_index += 1
                        bar.update(subtitle=f"{subject}: {polarity} {target}")
                        for opto_label, frac_correct, opto_amp in (
                            ("Opto off", baseline, 0.0),
                            ("Opto on", opto_on, signed_amp),
                        ):
                            rows.append(
                                {
                                    "subject": subject,
                                    "model": model_subdir,
                                    "backend": backend,
                                    "target": target,
                                    "target_label": TARGET_LABELS[target],
                                    "polarity": polarity,
                                    "polarity_label": POLARITY_LABELS[polarity],
                                    "opto": opto_label,
                                    "opto_amp": opto_amp,
                                    "magnitude": magnitude,
                                    "M": reps,
                                    "n_trials": group_arrays["n_trials"],
                                    "n_param_rows": int(
                                        (params_df["subject"].astype(str) == subject).sum()
                                    ),
                                    "param_nll_per_trial": float(param_row["nll/trial"]),
                                    "frac_correct": frac_correct,
                                }
                            )

        summary = pd.DataFrame(rows)
        summary.attrs["params_log"] = params_log
        summary.attrs["model"] = model_subdir
        summary.attrs["backend"] = backend
        summary.attrs["n_param_rows_total"] = int(len(params_df))
        summary.attrs["n_subjects"] = int(len(subjects))
        summary.attrs["simulated_subjects"] = subjects
        summary.attrs["n_data_subjects"] = int(len(data_subjects))
        summary.attrs["data_subjects"] = data_subjects
        summary.attrs["n_trials_filtered"] = int(
            trials_df.attrs.get("filtered_n_trials", len(trials_df))
        )
        summary.attrs["n_trials_balanced"] = int(
            trials_df.attrs.get("balanced_n_trials", len(trials_df))
        )
        summary.attrs["balance_max_trials_per_subject"] = int(
            trials_df.attrs.get("balance_max_trials_per_subject", 0)
        )
        summary.attrs["balance_cond_columns"] = list(
            trials_df.attrs.get("balance_cond_columns", [])
        )
        summary.attrs["balance_seed"] = int(trials_df.attrs.get("balance_seed", 0))
        summary.attrs["progress_total_steps"] = int(total_steps)
        summary.attrs["progress_steps_per_subject"] = int(steps_per_subject)
        summary.attrs["config"] = config
        return summary

    if run_button.value:
        config = current_config()
        summary_df = run_opto_sweep()
        csv_path, json_path = save_result_df(summary_df, config)
        run_status = build_run_status(
            summary_df,
            config,
            "ran and saved",
            {"csv_path": str(csv_path), "json_path": str(json_path)},
        )
    elif load_button.value:
        config = current_config()
        loaded_df, metadata = load_result_df(config)
        if loaded_df is None:
            summary_df = empty_summary_df()
            run_status = build_run_status(
                summary_df,
                config,
                "load requested; no saved result for current config",
                metadata,
            )
        else:
            summary_df = loaded_df
            run_status = build_run_status(summary_df, config, "loaded", metadata)
    else:
        config = current_config()
        summary_df = empty_summary_df()
        run_status = build_run_status(summary_df, config, "idle")
    return run_status, summary_df


@app.cell
def _(PLOT_COLORS, custom_boxplot, mo, np, plt, summary_df):
    def _panel_boxplot(ax, data, polarity):
        panel = data[data["polarity"] == polarity].copy()
        target_order = ["center", "sides"]
        opto_order = ["Opto off", "Opto on"]
        positions_by_key = {
            ("center", "Opto off"): 0.82,
            ("center", "Opto on"): 1.18,
            ("sides", "Opto off"): 1.82,
            ("sides", "Opto on"): 2.18,
        }
        present_keys = [
            (target, opto)
            for target in target_order
            for opto in opto_order
            if ((panel["target"] == target) & (panel["opto"] == opto)).any()
        ]
        positions = [positions_by_key[key] for key in present_keys]
        values = [
            panel[(panel["target"] == target) & (panel["opto"] == opto)][
                "frac_correct"
            ].to_numpy(dtype=float)
            for target, opto in present_keys
        ]

        on_color = (
            PLOT_COLORS["excitation"]
            if polarity == "excitation"
            else PLOT_COLORS["inhibition"]
        )
        median_colors = [
            PLOT_COLORS["opto_off"] if opto == "Opto off" else on_color
            for _, opto in present_keys
        ]

        subjects_for_lines = sorted(panel["subject"].unique())
        line_rows = []
        for subject in subjects_for_lines:
            row = []
            subject_panel = panel[panel["subject"] == subject]
            for target, opto in present_keys:
                vals = subject_panel[
                    (subject_panel["target"] == target) & (subject_panel["opto"] == opto)
                ]["frac_correct"].to_numpy(dtype=float)
                row.append(vals[0] if vals.size else np.nan)
            line_rows.append(row)

        custom_boxplot(
            ax,
            values,
            positions=positions,
            widths=0.28,
            median_colors=median_colors,
            line_values=np.asarray(line_rows, dtype=float) if line_rows else None,
            line_color=PLOT_COLORS["paired_lines"],
            line_alpha=0.25,
            line_linewidth=1.0,
            zorder=1,
        )

        for target, opto in present_keys:
            vals = panel[(panel["target"] == target) & (panel["opto"] == opto)][
                "frac_correct"
            ].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            x = positions_by_key[(target, opto)]
            color = PLOT_COLORS["opto_off"] if opto == "Opto off" else on_color
            ax.scatter(
                np.full(vals.size, x),
                vals,
                s=22,
                color=color,
                alpha=0.65,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

        ax.set_title("Excitation" if polarity == "excitation" else "Inhibition")
        xticks = []
        xticklabels = []
        if any(key[0] == "center" for key in present_keys):
            xticks.append(1.0)
            xticklabels.append("Center")
        if any(key[0] == "sides" for key in present_keys):
            xticks.append(2.0)
            xticklabels.append("Sides")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Frac. correct")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

        handles = [
            plt.Line2D([0], [0], color=PLOT_COLORS["opto_off"], lw=3, label="Opto off"),
            plt.Line2D([0], [0], color=on_color, lw=3, label="Opto on"),
        ]
        ax.legend(handles=handles, frameon=False, loc="lower right")

    if summary_df.empty:
        opto_plot = mo.md("Run the simulation to create the optogenetic boxplots.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
        _panel_boxplot(axes[0], summary_df, "excitation")
        _panel_boxplot(axes[1], summary_df, "inhibition")
        fig.tight_layout()
        opto_plot = fig

    opto_plot
    return


@app.cell
def _(mo, run_status, summary_df):
    status_md = mo.md(
        f"""
        **Result status:** {run_status['source']}

        **Config:** model `{run_status['model']}`, backend `{run_status['backend']}`,
        opto population `{run_status['opto_population']}`, trial groups `{run_status['target_selector']}`,
        magnitude `{run_status['magnitude']}`,
        M `{run_status['M']}`, numba threads `{run_status['numba_threads']}`

        **Cache CSV:** `{run_status['cache_csv']}`

        **Simulation dataframe subjects ({run_status['n_data_subjects']}):**
        {", ".join(run_status['data_subjects']) if run_status['data_subjects'] else "not loaded yet"}

        **Trial subsample:** {run_status['n_trials_balanced']} balanced trials from
        {run_status['n_trials_filtered']} filtered trials, capped at
        {run_status['balance_max_trials_per_subject']} per subject over
        `{", ".join(run_status['balance_cond_columns'])}`.

        **Subjects simulated/plotted ({run_status['n_subjects']}):**
        {", ".join(run_status['subjects']) if run_status['subjects'] else "none"}

        **Progress steps:** {run_status['progress_total_steps']} total =
        {run_status['n_subjects']} subjects x {run_status['progress_steps_per_subject']} steps per subject
        (baseline opto-off + excitation-on/inhibition-on for each selected trial group).

        **Rows:** {run_status['rows']} | **parameter rows:** {run_status['n_param_rows_total']} |
        **invalid frac_correct rows:** {run_status['invalid_rows']}
        """
    )
    if summary_df.empty:
        table_view = status_md
    else:
        table_view = mo.vstack(
            [
                status_md,
                mo.ui.table(summary_df, page_size=20),
            ]
        )
    table_view
    return


@app.cell
def _(mo):
    sweep_max_magnitude_slider = mo.ui.slider(
        start=0.02,
        stop=2.0,
        step=0.02,
        value=0.5,
        show_value=True,
        label="Sweep max |current|",
    )
    sweep_points_slider = mo.ui.slider(
        start=3,
        stop=41,
        step=2,
        value=21,
        show_value=True,
        label="Sweep points",
    )
    sweep_run_button = mo.ui.run_button(label="Run magnitude sweep")
    sweep_load_button = mo.ui.run_button(label="Load saved sweep")
    sweep_controls = mo.vstack(
        [
            mo.md("**Magnitude sweep**"),
            mo.hstack(
                [
                    sweep_max_magnitude_slider,
                    sweep_points_slider,
                    sweep_run_button,
                    sweep_load_button,
                ]
            ),
        ]
    )
    sweep_controls
    return (
        sweep_load_button,
        sweep_max_magnitude_slider,
        sweep_points_slider,
        sweep_run_button,
    )


@app.cell
def _(
    BALANCE_COND_COLUMNS,
    BALANCE_MAX_TRIALS,
    BALANCE_SEED,
    CODE_DIR,
    DATA_PATH,
    DELAY_LABELS,
    DT,
    REQUIRED_COLUMNS,
    RESULT_CACHE_DIR,
    SIDE_LABELS,
    THRESHOLDS,
    arrays_for_subject,
    backend_selector,
    load_model_params,
    load_trials,
    mo,
    model_selector,
    np,
    pd,
    reps_slider,
    sweep_load_button,
    sweep_max_magnitude_slider,
    sweep_points_slider,
    sweep_run_button,
    theta_from_params_row,
    thread_slider,
):
    def empty_sweep_df():
        return pd.DataFrame(
            columns=[
                "subject",
                "model",
                "backend",
                "delay_code",
                "ttype_c",
                "side_code",
                "side_label",
                "trial_group",
                "opto_amp",
                "magnitude_abs",
                "pL",
                "pC",
                "pR",
                "p_correct",
                "M",
            ]
        )

    def sweep_config():
        return {
            "schema_version": 3,
            "kind": "opto_magnitude_sweep",
            "model": str(model_selector.value),
            "backend": str(backend_selector.value),
            "opto_population": "center",
            "max_magnitude": round(float(sweep_max_magnitude_slider.value), 6),
            "n_points": int(sweep_points_slider.value),
            "M": int(reps_slider.value),
            "numba_threads": int(thread_slider.value),
            "dt": float(DT),
            "thresholds": [float(v) for v in THRESHOLDS],
            "balance_max_trials_per_subject": int(BALANCE_MAX_TRIALS),
            "balance_cond_columns": list(BALANCE_COND_COLUMNS),
            "balance_seed": int(BALANCE_SEED),
        }

    def sweep_paths(config):
        import hashlib
        import json

        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return (
            RESULT_CACHE_DIR / f"opto_magnitude_sweep_{key}.csv",
            RESULT_CACHE_DIR / f"opto_magnitude_sweep_{key}.json",
        )

    def save_sweep_df(sweep_df, config):
        import json
        from datetime import datetime

        RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        csv_path, json_path = sweep_paths(config)
        sweep_df.to_csv(csv_path, index=False)
        metadata = {
            "config": config,
            "csv": csv_path.name,
            "rows": int(len(sweep_df)),
            "subjects": int(sweep_df.attrs.get("n_subjects", 0)),
            "filtered_trials": int(sweep_df.attrs.get("n_trials_filtered", 0)),
            "balanced_trials": int(sweep_df.attrs.get("n_trials_balanced", 0)),
            "magnitudes": list(sweep_df.attrs.get("magnitudes", [])),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return csv_path, json_path

    def load_sweep_df(config):
        import json

        csv_path, json_path = sweep_paths(config)
        if not csv_path.exists() or not json_path.exists():
            return None, {
                "loaded": False,
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            }

        sweep_df = pd.read_csv(csv_path)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        sweep_df.attrs["loaded_from"] = str(csv_path)
        sweep_df.attrs["config"] = metadata.get("config", config)
        sweep_df.attrs["n_subjects"] = int(metadata.get("subjects", 0))
        sweep_df.attrs["n_trials_filtered"] = int(metadata.get("filtered_trials", 0))
        sweep_df.attrs["n_trials_balanced"] = int(metadata.get("balanced_trials", 0))
        sweep_df.attrs["magnitudes"] = list(metadata.get("magnitudes", []))
        metadata["loaded"] = True
        metadata["csv_path"] = str(csv_path)
        metadata["json_path"] = str(json_path)
        return sweep_df, metadata

    def run_magnitude_sweep():
        backend = backend_selector.value
        if backend != "julia":
            raise RuntimeError("The magnitude sweep is now trial-wise and requires the Julia backend.")

        import importlib
        import sys

        import polars as pl

        simulator_dir = CODE_DIR / "simulations"
        if str(simulator_dir) not in sys.path:
            sys.path.insert(0, str(simulator_dir))
        julia_backend = importlib.import_module("opto_julia_backend")

        reps = int(reps_slider.value)
        model_subdir = model_selector.value
        use_spatial = model_subdir.startswith("spatial")
        max_magnitude = float(sweep_max_magnitude_slider.value)
        n_points = int(sweep_points_slider.value)
        signed_magnitudes = np.linspace(-max_magnitude, max_magnitude, n_points)

        zero_idx = int(np.argmin(np.abs(signed_magnitudes)))
        signed_magnitudes[zero_idx] = 0.0

        def simulate_choice_probs_trials(subject_arrays, theta, opto_amp, seed):
            return julia_backend.simulate_choice_probs_trials_julia(
                subject_arrays,
                theta,
                reps=reps,
                dt=DT,
                thresholds=THRESHOLDS,
                opto_target=0,
                opto_amp=opto_amp,
                use_spatial=use_spatial,
                seed=seed,
            )

        trials_df = load_trials(DATA_PATH, REQUIRED_COLUMNS)
        params_df, best_params_df, params_log = load_model_params(trials_df, model_subdir)
        data_subjects = sorted(trials_df["subject"].dropna().astype(str).unique())
        best_params_df = best_params_df[
            best_params_df["subject"].astype(str).isin(set(data_subjects))
        ].copy()
        param_rows = list(best_params_df.sort_values("subject").iterrows())

        frames = []
        total_steps = len(param_rows) * len(signed_magnitudes)
        with mo.status.progress_bar(
            total=total_steps,
            title="Running magnitude sweep",
            subtitle=f"Model: {model_subdir}; backend: {backend}",
            completion_title="Magnitude sweep complete",
            completion_subtitle=f"{len(param_rows)} subjects simulated",
        ) as bar:
            for subject_index, (_, param_row) in enumerate(param_rows):
                subject = str(param_row["subject"])
                subject_arrays = arrays_for_subject(trials_df, subject)
                theta = theta_from_params_row(param_row)
                n_trials = int(subject_arrays["n_trials"])
                if n_trials == 0:
                    bar.update(
                        increment=len(signed_magnitudes),
                        subtitle=f"Skipped {subject}: no trials",
                    )
                    continue

                delay_codes = subject_arrays["delayd"].astype(np.int16, copy=False)
                side_codes = subject_arrays["side"].astype(np.int16, copy=False)
                ttype_labels = np.asarray(
                    [DELAY_LABELS.get(int(code), str(int(code))) for code in delay_codes],
                    dtype=object,
                )
                side_labels = np.asarray(
                    [SIDE_LABELS.get(int(code), str(int(code))) for code in side_codes],
                    dtype=object,
                )
                trial_groups = np.where(side_codes == 1, "center", "sides")

                for amp_index, signed_amp in enumerate(signed_magnitudes):
                    seed_base = 900_000 + 10_000 * subject_index + 10 * amp_index
                    probs = simulate_choice_probs_trials(
                        subject_arrays, theta, float(signed_amp), seed_base
                    )
                    correct_idx = np.clip(side_codes.astype(np.int64), 0, 2)
                    p_correct = probs[np.arange(n_trials), correct_idx]
                    frames.append(
                        pl.DataFrame(
                            {
                                "subject": [subject] * n_trials,
                                "model": [model_subdir] * n_trials,
                                "backend": [backend] * n_trials,
                                "delay_code": delay_codes,
                                "ttype_c": ttype_labels,
                                "side_code": side_codes,
                                "side_label": side_labels,
                                "trial_group": trial_groups,
                                "opto_amp": np.full(n_trials, float(signed_amp)),
                                "magnitude_abs": np.full(n_trials, abs(float(signed_amp))),
                                "pL": probs[:, 0],
                                "pC": probs[:, 1],
                                "pR": probs[:, 2],
                                "p_correct": p_correct,
                                "M": np.full(n_trials, reps, dtype=np.int32),
                            }
                        )
                    )
                    bar.update(subtitle=f"{subject}: amp {signed_amp:.3g}")

        if frames:
            sweep_pl = pl.concat(frames, how="vertical")
            sweep = pd.DataFrame(sweep_pl.to_dict(as_series=False))
        else:
            sweep = empty_sweep_df()
        sweep.attrs["params_log"] = params_log
        sweep.attrs["model"] = model_subdir
        sweep.attrs["backend"] = backend
        sweep.attrs["n_subjects"] = int(sweep["subject"].nunique()) if not sweep.empty else 0
        sweep.attrs["n_trials_filtered"] = int(
            trials_df.attrs.get("filtered_n_trials", len(trials_df))
        )
        sweep.attrs["n_trials_balanced"] = int(
            trials_df.attrs.get("balanced_n_trials", len(trials_df))
        )
        sweep.attrs["magnitudes"] = [float(v) for v in signed_magnitudes]
        return sweep

    _sweep_config = sweep_config()
    if sweep_run_button.value:
        _loaded_sweep_df, _sweep_metadata = load_sweep_df(_sweep_config)
        if _loaded_sweep_df is not None:
            sweep_df = _loaded_sweep_df
            sweep_status = {
                "source": "loaded cached sweep",
                "rows": int(len(sweep_df)),
                "subjects": int(sweep_df.attrs.get("n_subjects", 0)),
                "filtered_trials": int(sweep_df.attrs.get("n_trials_filtered", 0)),
                "balanced_trials": int(sweep_df.attrs.get("n_trials_balanced", 0)),
                "cache_csv": _sweep_metadata["csv_path"],
            }
        else:
            sweep_df = run_magnitude_sweep()
            _sweep_csv_path, _sweep_json_path = save_sweep_df(sweep_df, _sweep_config)
            sweep_status = {
                "source": "ran and saved",
                "rows": int(len(sweep_df)),
                "subjects": int(sweep_df.attrs.get("n_subjects", 0)),
                "filtered_trials": int(sweep_df.attrs.get("n_trials_filtered", 0)),
                "balanced_trials": int(sweep_df.attrs.get("n_trials_balanced", 0)),
                "cache_csv": str(_sweep_csv_path),
            }
    elif sweep_load_button.value:
        _loaded_sweep_df, _sweep_metadata = load_sweep_df(_sweep_config)
        if _loaded_sweep_df is not None:
            sweep_df = _loaded_sweep_df
            sweep_status = {
                "source": "loaded cached sweep",
                "rows": int(len(sweep_df)),
                "subjects": int(sweep_df.attrs.get("n_subjects", 0)),
                "filtered_trials": int(sweep_df.attrs.get("n_trials_filtered", 0)),
                "balanced_trials": int(sweep_df.attrs.get("n_trials_balanced", 0)),
                "cache_csv": _sweep_metadata["csv_path"],
            }
        else:
            sweep_df = empty_sweep_df()
            sweep_status = {
                "source": "load requested; no saved sweep for current config",
                "rows": 0,
                "subjects": 0,
                "filtered_trials": 0,
                "balanced_trials": 0,
                "cache_csv": _sweep_metadata["csv_path"],
            }
    else:
        _loaded_sweep_df, _sweep_metadata = load_sweep_df(_sweep_config)
        if _loaded_sweep_df is not None:
            sweep_df = _loaded_sweep_df
            sweep_status = {
                "source": "loaded cached sweep",
                "rows": int(len(sweep_df)),
                "subjects": int(sweep_df.attrs.get("n_subjects", 0)),
                "filtered_trials": int(sweep_df.attrs.get("n_trials_filtered", 0)),
                "balanced_trials": int(sweep_df.attrs.get("n_trials_balanced", 0)),
                "cache_csv": _sweep_metadata["csv_path"],
            }
        else:
            sweep_df = empty_sweep_df()
            sweep_status = {
                "source": "idle",
                "rows": 0,
                "subjects": 0,
                "filtered_trials": 0,
                "balanced_trials": 0,
                "cache_csv": _sweep_metadata["csv_path"],
            }
    return sweep_df, sweep_status


@app.cell
def _(DELAY_LABELS, mo, np, sweep_df):
    hard_ttype_selector = mo.ui.dropdown(
        options={label: code for code, label in DELAY_LABELS.items()},
        value=2,
        label="Harder ttype_c curve",
    )
    if sweep_df.empty or "opto_amp" not in sweep_df.columns:
        _triangle_amp_values = np.array([0.0])
    else:
        _triangle_amp_values = np.sort(
            sweep_df["opto_amp"].dropna().astype(float).unique()
        )
        if _triangle_amp_values.size == 0:
            _triangle_amp_values = np.array([0.0])
    _zero_idx = int(np.argmin(np.abs(_triangle_amp_values)))
    _triangle_default = float(_triangle_amp_values[_zero_idx])
    triangle_amp_selector = mo.ui.dropdown(
        options={
            f"{float(value):+.3g}": float(value)
            for value in _triangle_amp_values
        },
        value=_triangle_default,
        label="Triangle IC current",
    )
    triangle_video_button = mo.ui.run_button(label="Save triangle IC video")
    plot_controls = mo.vstack(
        [
            mo.md("**Sweep plots**"),
            mo.hstack(
                [
                    hard_ttype_selector,
                    triangle_amp_selector,
                    triangle_video_button,
                ]
            ),
        ]
    )
    plot_controls
    return hard_ttype_selector, triangle_amp_selector, triangle_video_button


@app.cell
def _(
    PLOT_COLORS,
    hard_ttype_selector,
    mo,
    np,
    pd,
    plt,
    sweep_df,
    sweep_status,
    triangle_amp_selector,
):
    import polars as pl

    def _curve_data(raw_pl, delay_code=None):
        data = raw_pl
        if delay_code is not None:
            data = data.filter(pl.col("delay_code") == int(delay_code))
        grouped = (
            data.group_by(["subject", "opto_amp"])
            .agg(
                pl.col("p_correct")
                .filter(pl.col("trial_group") == "center")
                .mean()
                .alias("frac_center"),
                pl.col("p_correct")
                .filter(pl.col("trial_group") == "sides")
                .mean()
                .alias("frac_sides"),
            )
            .sort(["subject", "opto_amp"])
        )
        curve = pd.DataFrame(grouped.to_dict(as_series=False))
        if curve.empty:
            return curve

        baseline = curve[curve["opto_amp"] == 0.0][
            ["subject", "frac_center", "frac_sides"]
        ].rename(
            columns={
                "frac_center": "baseline_center",
                "frac_sides": "baseline_sides",
            }
        )
        curve = curve.merge(baseline, on="subject", how="left")
        curve["delta_center"] = curve["frac_center"] - curve["baseline_center"]
        curve["delta_sides"] = curve["frac_sides"] - curve["baseline_sides"]
        return curve

    def _plot_subject_segments(ax, data, x_col, y_col):
        for _, subject_df in data.groupby("subject"):
            subject_df = subject_df.sort_values("opto_amp")
            for mask, color in (
                (subject_df["opto_amp"] <= 0.0, PLOT_COLORS["inhibition"]),
                (subject_df["opto_amp"] >= 0.0, PLOT_COLORS["excitation"]),
            ):
                segment = subject_df[mask]
                if len(segment) < 2:
                    continue
                ax.plot(
                    segment[x_col],
                    segment[y_col],
                    color=color,
                    alpha=0.14,
                    lw=1.7,
                    zorder=1,
                )

    def _plot_mean_segments(ax, data, x_col, y_col, *, linestyle, label_suffix):
        for mask, color, label in (
            (data["opto_amp"] <= 0.0, PLOT_COLORS["inhibition"], "Inhibition"),
            (data["opto_amp"] >= 0.0, PLOT_COLORS["excitation"], "Excitation"),
        ):
            segment = data[mask].sort_values("opto_amp")
            if len(segment) < 2:
                continue
            ax.plot(
                segment[x_col],
                segment[y_col],
                color=color,
                lw=3.0,
                linestyle=linestyle,
                label=f"{label} {label_suffix}",
                zorder=4,
            )

    def _style_delta_ax(ax, xlabel, ylabel):
        ax.axhline(0.0, color=PLOT_COLORS["zero_lines"], lw=1.2, zorder=0)
        ax.axvline(0.0, color=PLOT_COLORS["zero_lines"], lw=1.2, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_aspect(1)
        ax.legend(frameon=False, loc="best", fontsize=8)

    def _add_amp_scatter(fig, ax, data, x_col, y_col):
        cmap = plt.get_cmap("RdBu_r")
        norm = plt.Normalize(
            vmin=float(data["opto_amp"].min()),
            vmax=float(data["opto_amp"].max()),
        )
        points = ax.scatter(
            data[x_col],
            data[y_col],
            c=data["opto_amp"],
            cmap=cmap,
            norm=norm,
            s=30,
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )
        cbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("IC opto current")

    def _mean_curve(data, value_cols):
        return data.groupby("opto_amp", as_index=False)[value_cols].mean().sort_values("opto_amp")

    def _heat_data(data, mask):
        panel = data[mask].copy()
        panel["amp_abs"] = panel["opto_amp"].abs().round(6)
        return (
            panel.groupby("amp_abs", as_index=False)[["frac_center", "frac_sides"]]
            .mean()
            .sort_values("amp_abs")
        )

    def _set_sparse_amp_ticks(ax, amps):
        if len(amps) == 0:
            return
        tick_step = max(1, int(np.ceil(len(amps) / 7)))
        ticks = list(range(0, len(amps), tick_step))
        if ticks[-1] != len(amps) - 1:
            ticks.append(len(amps) - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{amps[i]:.2g}" for i in ticks], rotation=45, ha="right")

    def _ternary_xy(pL, pC, pR):
        total = pL + pC + pR
        if total <= 0 or not np.isfinite(total):
            return np.nan, np.nan
        pL, pC, pR = pL / total, pC / total, pR / total
        return pR + 0.5 * pC, (np.sqrt(3.0) / 2.0) * pC

    def _draw_triangle_axes(ax):
        h = np.sqrt(3.0) / 2.0
        ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="#163B4A", lw=1.4)
        ax.text(-0.04, -0.04, "L", ha="right", va="top", fontsize=12, weight="bold")
        ax.text(1.04, -0.04, "R", ha="left", va="top", fontsize=12, weight="bold")
        ax.text(0.5, h + 0.04, "C", ha="center", va="bottom", fontsize=12, weight="bold")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, h + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")

    if sweep_df.empty:
        sweep_plot = mo.md("Run the magnitude sweep to create the model sweep plots.")
    else:
        raw_pl = pl.from_pandas(sweep_df)
        selected_delay = int(hard_ttype_selector.value)
        all_curve = _curve_data(raw_pl)
        hard_curve = _curve_data(raw_pl, selected_delay)
        hard_labels = (
            raw_pl.filter(pl.col("delay_code") == selected_delay)
            .select("ttype_c")
            .unique()
            .to_series()
            .to_list()
        )
        hard_label = f"ttype_c={hard_labels[0]}" if hard_labels else f"ttype_c={selected_delay}"

        mean_all = _mean_curve(
            all_curve,
            ["delta_center", "delta_sides"],
        )
        mean_hard = (
            _mean_curve(
                hard_curve,
                ["delta_center", "delta_sides"],
            )
            if not hard_curve.empty
            else None
        )

        acc_fig, acc_ax = plt.subplots(figsize=(5.4, 4.9))
        _plot_subject_segments(acc_ax, all_curve, "delta_center", "delta_sides")
        _plot_mean_segments(
            acc_ax, mean_all, "delta_center", "delta_sides", linestyle="-", label_suffix="all"
        )
        _add_amp_scatter(acc_fig, acc_ax, mean_all, "delta_center", "delta_sides")
        _style_delta_ax(acc_ax, r"$\Delta$Acc center", r"$\Delta$Acc side")
        acc_ax.set_title("All trials")
        acc_ax.set_xlim(-1, 1)
        acc_ax.set_ylim(-1, 1)
        acc_fig.tight_layout()

        hard_fig, hard_ax = plt.subplots(figsize=(5.4, 4.9))
        if mean_hard is not None:
            _plot_subject_segments(hard_ax, hard_curve, "delta_center", "delta_sides")
            _plot_mean_segments(
                hard_ax,
                mean_hard,
                "delta_center",
                "delta_sides",
                linestyle="-",
                label_suffix=hard_label,
            )
            _add_amp_scatter(hard_fig, hard_ax, mean_hard, "delta_center", "delta_sides")
        _style_delta_ax(hard_ax, r"$\Delta$Acc center", r"$\Delta$Acc side")
        hard_ax.set_title(hard_label)
        hard_ax.set_xlim(-1, 1)
        hard_ax.set_ylim(-1, 1)
        hard_fig.tight_layout()

        mean_acc_df = _mean_curve(all_curve, ["frac_center", "frac_sides"])
        heat_panels = [
            (
                "Inhibition",
                _heat_data(mean_acc_df, mean_acc_df["opto_amp"] <= 0.0),
                PLOT_COLORS["inhibition"],
            ),
            (
                "Excitation",
                _heat_data(mean_acc_df, mean_acc_df["opto_amp"] >= 0.0),
                PLOT_COLORS["excitation"],
            ),
        ]
        heat_fig, heat_axes = plt.subplots(
            1,
            2,
            figsize=(8.0, 2.9),
            sharey=True,
            constrained_layout=True,
        )
        heat_im = None
        for heat_ax, (title, heat_data, title_color) in zip(
            heat_axes, heat_panels, strict=False
        ):
            heat_values = heat_data[["frac_center", "frac_sides"]].to_numpy(dtype=float).T
            heat_im = heat_ax.imshow(
                heat_values,
                cmap=PLOT_COLORS["heatmap_cmap"],
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
            )
            amps = heat_data["amp_abs"].to_numpy(dtype=float)
            _set_sparse_amp_ticks(heat_ax, amps)
            heat_ax.set_yticks([0, 1])
            heat_ax.set_yticklabels(["Center", "Sides"])
            heat_ax.set_title(title, color=title_color)
            heat_ax.set_xlabel("|IC current|")
        heat_axes[0].set_ylabel("Trial group")
        heat_fig.colorbar(
            heat_im,
            ax=heat_axes,
            fraction=0.046,
            pad=0.04,
            label="Model accuracy",
        )

        _available_triangle_amps = [
            float(value)
            for value in raw_pl.select("opto_amp")
            .unique()
            .sort("opto_amp")
            .to_series()
            .to_list()
        ]
        _requested_triangle_amp = float(triangle_amp_selector.value)
        _selected_triangle_amp = min(
            _available_triangle_amps,
            key=lambda value: abs(value - _requested_triangle_amp),
        )
        tri_raw = raw_pl.filter(pl.col("opto_amp") == _selected_triangle_amp)
        _triangle_probs_pl = (
            tri_raw.group_by(["ttype_c", "side_label"])
            .agg(
                pl.col("pL").mean().alias("pL"),
                pl.col("pC").mean().alias("pC"),
                pl.col("pR").mean().alias("pR"),
            )
        )
        _triangle_probs_plot_df = pd.DataFrame(_triangle_probs_pl.to_dict(as_series=False))
        if _triangle_probs_plot_df.empty:
            triangle_plot = mo.md(
                "Triangle probabilities are available only after a Julia sweep run."
            )
        else:
            tri_mean = (
                _triangle_probs_plot_df.groupby(
                    ["ttype_c", "side_label"], as_index=False
                )[["pL", "pC", "pR"]]
                .mean()
                .sort_values(["ttype_c", "side_label"])
            )
            ttype_values = sorted(tri_mean["ttype_c"].unique())
            if len(ttype_values) == 1:
                ttype_alpha = {ttype_values[0]: 1.0}
            else:
                ttype_alpha = {
                    ttype: 0.35 + 0.65 * idx / (len(ttype_values) - 1)
                    for idx, ttype in enumerate(ttype_values)
                }
            side_colors = PLOT_COLORS["triangle_side_colors"]
            triangle_fig, triangle_ax = plt.subplots(figsize=(4.8, 4.2))
            _draw_triangle_axes(triangle_ax)
            for _, row in tri_mean.iterrows():
                x, y = _ternary_xy(row["pL"], row["pC"], row["pR"])
                triangle_ax.scatter(
                    x,
                    y,
                    s=70,
                    color=side_colors.get(row["side_label"], "tab:gray"),
                    alpha=ttype_alpha[row["ttype_c"]],
                    marker="o",
                    edgecolor="#163B4A",
                    linewidth=0.8,
                    zorder=3,
                )
            triangle_ax.set_title(f"IC current {_selected_triangle_amp:+.3g}")

            side_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor="#163B4A",
                    label=f"side {side}",
                )
                for side, color in side_colors.items()
            ]
            ttype_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    color="#163B4A",
                    markerfacecolor="#163B4A",
                    alpha=ttype_alpha[ttype],
                    label=str(ttype),
                )
                for ttype in ttype_values
            ]
            triangle_ax.legend(
                handles=side_handles + ttype_handles,
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
            )
            triangle_fig.tight_layout()
            triangle_plot = triangle_fig

        sweep_plot = mo.vstack(
            [
                mo.md(
                    f"**Sweep status:** {sweep_status['source']} | "
                    f"{sweep_status['subjects']} subjects | "
                    f"{sweep_status['balanced_trials']} balanced trials"
                ),
                acc_fig,
                hard_fig,
                heat_fig,
                triangle_plot,
                mo.ui.table(sweep_df, page_size=20),
            ]
        )

    sweep_plot
    return


@app.cell
def _(
    PLOT_COLORS,
    RESULT_CACHE_DIR,
    mo,
    np,
    pd,
    plt,
    sweep_df,
    triangle_video_button,
):
    import imageio.v2 as imageio
    import polars as _pl

    def _ternary_xy(pL, pC, pR):
        total = pL + pC + pR
        if total <= 0 or not np.isfinite(total):
            return np.nan, np.nan
        pL, pC, pR = pL / total, pC / total, pR / total
        return pR + 0.5 * pC, (np.sqrt(3.0) / 2.0) * pC

    def _draw_triangle_axes(ax):
        h = np.sqrt(3.0) / 2.0
        ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="#163B4A", lw=1.4)
        ax.text(-0.04, -0.04, "L", ha="right", va="top", fontsize=12, weight="bold")
        ax.text(1.04, -0.04, "R", ha="left", va="top", fontsize=12, weight="bold")
        ax.text(0.5, h + 0.04, "C", ha="center", va="bottom", fontsize=12, weight="bold")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, h + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")

    def _triangle_mean(raw_pl, opto_amp):
        triangle_pl = (
            raw_pl.filter(_pl.col("opto_amp") == float(opto_amp))
            .group_by(["ttype_c", "side_label"])
            .agg(
                _pl.col("pL").mean().alias("pL"),
                _pl.col("pC").mean().alias("pC"),
                _pl.col("pR").mean().alias("pR"),
            )
            .sort(["ttype_c", "side_label"])
        )
        return pd.DataFrame(triangle_pl.to_dict(as_series=False))

    def _plot_triangle_frame(ax, triangle_df, opto_amp, ttype_alpha):
        _draw_triangle_axes(ax)
        side_colors = PLOT_COLORS["triangle_side_colors"]
        for _, row in triangle_df.iterrows():
            x, y = _ternary_xy(row["pL"], row["pC"], row["pR"])
            ax.scatter(
                x,
                y,
                s=76,
                color=side_colors.get(row["side_label"], "tab:gray"),
                alpha=ttype_alpha[row["ttype_c"]],
                marker="o",
                edgecolor="#163B4A",
                linewidth=0.8,
                zorder=3,
            )
        ax.set_title(f"IC current {float(opto_amp):+.3g}")

    def _write_triangle_video():
        raw_pl = _pl.from_pandas(sweep_df)
        amps = [
            float(value)
            for value in raw_pl.select("opto_amp")
            .unique()
            .sort("opto_amp")
            .to_series()
            .to_list()
        ]
        ttype_values = (
            raw_pl.select("ttype_c").unique().sort("ttype_c").to_series().to_list()
        )
        if len(ttype_values) == 1:
            ttype_alpha = {ttype_values[0]: 1.0}
        else:
            ttype_alpha = {
                ttype: 0.35 + 0.65 * idx / (len(ttype_values) - 1)
                for idx, ttype in enumerate(ttype_values)
            }
        frames = []
        for opto_amp in amps:
            triangle_df = _triangle_mean(raw_pl, opto_amp)
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            _plot_triangle_frame(ax, triangle_df, opto_amp, ttype_alpha)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            plt.close(fig)
        RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        video_path = RESULT_CACHE_DIR / "opto_triangle_ic_sweep.gif"
        imageio.mimsave(video_path, frames, fps=4, loop=0)
        return video_path

    if sweep_df.empty:
        triangle_video_status = mo.md("Run or load a sweep before saving the triangle video.")
    elif triangle_video_button.value:
        _triangle_video_path = _write_triangle_video()
        triangle_video_status = mo.md(
            f"Saved triangle IC sweep video: `{_triangle_video_path}`"
        )
    else:
        triangle_video_status = mo.md(
            "Click **Save triangle IC video** to write an animation over all simulated IC currents."
        )

    triangle_video_status
    return


if __name__ == "__main__":
    app.run()
