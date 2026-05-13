# /// script
# dependencies = [
#     "anywidget",
#     "imageio",
#     "jax",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "traitlets",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import warnings
    from pathlib import Path

    sys.modules.setdefault("numexpr", None)
    sys.modules.setdefault("bottleneck", None)

    import anywidget
    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import traitlets

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return Path, anywidget, imageio, jax, jnp, mo, np, pd, plt, traitlets


@app.cell
def _(Path):
    HERE = Path(__file__).resolve()
    CODE_DIR = HERE.parents[1]
    SIM_DIR = HERE.parents[2]
    DATA_PATH = SIM_DIR / "datasets" / "df_filtered.csv"
    RESULT_DIR = SIM_DIR / "simulations" / "synthetic_selective_inhibition_results"
    MOLAB_URL = (
        "https://molab.marimo.io/github/javirm3/TFG/blob/main/"
        "simulations/code/simulations/synthetic_selective_inhibition_opto.py"
    )

    DT = 0.01
    N_STEPS = 360
    CHOICE_LABELS = ("L", "C", "R")
    SIDE_COLORS = {"L": "tab:red", "C": "tab:green", "R": "tab:blue"}
    STIM_BY_SIDE = {"L": "SL", "C": "SM", "R": "SS"}
    SIDE_BY_STIM = {"SL": "L", "SM": "C", "SS": "R", "VG": "C"}
    return (
        CHOICE_LABELS,
        DATA_PATH,
        DT,
        MOLAB_URL,
        N_STEPS,
        RESULT_DIR,
        SIDE_COLORS,
        STIM_BY_SIDE,
    )


@app.cell
def _(DATA_PATH, STIM_BY_SIDE, pd):
    FALLBACK_TIMING = {
        ("VG", "VG"): (0.5937, 0.9900, 1.3532, 3.1564),
        ("DS", "SL"): (0.6075, 0.9978, 1.3466, 3.2325),
        ("DS", "SM"): (0.6628, 1.0782, 1.4463, 3.3350),
        ("DS", "SS"): (0.6614, 1.0799, 1.4622, 3.4225),
        ("DM", "SM"): (0.6253, 1.0238, 1.3838, 3.3435),
        ("DM", "SS"): (0.6615, 1.0783, 1.4539, 3.4008),
        ("DL", "SS"): (0.6315, 1.0443, 1.4425, 3.3978),
    }

    def _infer_csv_sep(path):
        with open(path, "r", encoding="utf-8") as handle:
            first_line = handle.readline()
        return ";" if first_line.count(";") > first_line.count(",") else ","

    def load_real_timing_means(path):
        if not path.exists():
            return pd.DataFrame(
                [
                    {
                        "ttype_c": key[0],
                        "stimd_c": key[1],
                        "timepoint_1": value[0],
                        "timepoint_2": value[1],
                        "timepoint_3": value[2],
                        "timepoint_4": value[3],
                    }
                    for key, value in FALLBACK_TIMING.items()
                ]
            )
        sep = _infer_csv_sep(path)
        cols = [
            "ttype_c",
            "stimd_c",
            "timepoint_1",
            "timepoint_2",
            "timepoint_3",
            "timepoint_4",
        ]
        df = pd.read_csv(path, sep=sep, usecols=cols)
        for col in ("ttype_c", "stimd_c"):
            df[col] = df[col].astype("string").str.strip()
        means = (
            df.groupby(["ttype_c", "stimd_c"], as_index=False)[
                ["timepoint_1", "timepoint_2", "timepoint_3", "timepoint_4"]
            ]
            .mean()
            .sort_values(["ttype_c", "stimd_c"])
        )
        wanted = set(FALLBACK_TIMING)
        means = means[
            means.apply(lambda row: (row["ttype_c"], row["stimd_c"]) in wanted, axis=1)
        ].copy()
        present = set(zip(means["ttype_c"], means["stimd_c"], strict=False))
        missing_rows = []
        for key, value in FALLBACK_TIMING.items():
            if key not in present:
                missing_rows.append(
                    {
                        "ttype_c": key[0],
                        "stimd_c": key[1],
                        "timepoint_1": value[0],
                        "timepoint_2": value[1],
                        "timepoint_3": value[2],
                        "timepoint_4": value[3],
                    }
                )
        if missing_rows:
            means = pd.concat([means, pd.DataFrame(missing_rows)], ignore_index=True)
        return means.sort_values(["ttype_c", "stimd_c"]).reset_index(drop=True)

    timing_means_df = load_real_timing_means(DATA_PATH)

    def timing_lookup_from_df(df):
        lookup = {}
        for _, row in df.iterrows():
            lookup[(str(row["ttype_c"]), str(row["stimd_c"]))] = tuple(
                float(row[f"timepoint_{idx}"]) for idx in range(1, 5)
            )
        for key, value in FALLBACK_TIMING.items():
            lookup.setdefault(key, value)
        return lookup

    TIMING_LOOKUP = timing_lookup_from_df(timing_means_df)
    TTYPE_CODE = {"VG": 0, "DS": 1, "DM": 2, "DL": 3}
    STIM_CODE = {"VG": 0, "SS": 1, "SM": 2, "SL": 3}
    SIDE_CODE = {"L": 0, "C": 1, "R": 2}
    condition_keys = [
        key
        for ttype in ("VG", "DS", "DM", "DL")
        for stimd in ("VG", "SL", "SM", "SS")
        if (key := (ttype, stimd)) in TIMING_LOOKUP
    ]
    SIMULATION_CONDITIONS = tuple(
        (
            TTYPE_CODE[ttype],
            ttype,
            STIM_CODE[stimd],
            stimd,
            SIDE_CODE[side],
            side,
            *TIMING_LOOKUP[(ttype, stimd)],
        )
        for ttype, stimd in condition_keys
        for side in ("L", "C", "R")
    )

    def timing_for_trial(ttype, side_label):
        stim = "VG" if ttype == "VG" else STIM_BY_SIDE[side_label]
        return stim, TIMING_LOOKUP[(ttype, stim)]

    return SIMULATION_CONDITIONS, timing_means_df


@app.cell
def _(MOLAB_URL, mo, timing_means_df):
    mo.vstack(
        [
            mo.md(
                f"""
                # Synthetic selective-inhibition optogenetic sweep

                [![Open in molab](https://marimo.io/molab-shield.svg)]({MOLAB_URL})

                Synthetic L/C/R trials use the real-data mean timing table below, but no subject
                data or fitted parameters are used in the simulation.

                The simulated rates follow the original three-choice form with separate
                inhibitory populations: no E-E connections and no I-I connections. Each
                excitatory population has self-amplification, each inhibitory population
                inhibits its matching excitatory population and the two alternatives, and
                excitatory populations drive the two non-matching inhibitory populations.
                """
            ),
            mo.ui.table(timing_means_df.round(4), page_size=8),
        ]
    )
    return


@app.function
def default_params():
    return {
        "sL": 1.00,
        "sC": 1.00,
        "sR": 1.00,
        "w_EL_IC": 1.00,
        "w_EL_IR": 1.00,
        "w_EC_IL": 1.00,
        "w_EC_IR": 1.00,
        "w_ER_IL": 1.00,
        "w_ER_IC": 1.00,
        "w_IL_EL": 1.00,
        "w_IL_EC": 1.00,
        "w_IL_ER": 1.00,
        "w_IC_EL": 1.00,
        "w_IC_EC": 1.00,
        "w_IC_ER": 1.00,
        "w_IR_EL": 1.00,
        "w_IR_EC": 1.00,
        "w_IR_ER": 1.00,
        "i0_IL": 1.0 / 3.0,
        "i0_IC": 1.0 / 3.0,
        "i0_IR": 1.0 / 3.0,
        "tau_e": 0.13,
        "tau_i": 0.055,
        "decision_threshold": 0.22,
        "noise_amp": 1.00,
        "opto_target": 1,
        "opto_mode": 1,
    }


@app.cell
def _(architecture_widget, mo):
    symbolic_equations = mo.md(
        r"""
        ## Transfer function and rate equations

        Here \(I_L(t),I_C(t),I_R(t)\) are the simulated stimulus plus urgency inputs to the excitatory populations. The \(b_{I_L},b_{I_C},b_{I_R}\) terms are tonic inhibitory-population biases, not stimulus inputs.

        $$\phi(x)=\begin{cases}0,&x\le 0\\x^2,&0<x\le 1\\2\sqrt{x-0.75},&x>1\end{cases}$$

        $$\tau \dot r_L = -r_L + \phi\!\left(s_L r_L - w_{I_L E_L}r_{I_L} - w_{I_C E_L}r_{I_C} - w_{I_R E_L}r_{I_R} + I_L(t)\right)$$
        $$\tau \dot r_C = -r_C + \phi\!\left(s_C r_C - w_{I_L E_C}r_{I_L} - w_{I_C E_C}r_{I_C} - w_{I_R E_C}r_{I_R} + I_C(t)\right)$$
        $$\tau \dot r_R = -r_R + \phi\!\left(s_R r_R - w_{I_L E_R}r_{I_L} - w_{I_C E_R}r_{I_C} - w_{I_R E_R}r_{I_R} + I_R(t)\right)$$
        $$\tau \dot r_{I_L} = -r_{I_L} + \phi\!\left(\frac{w_{E_C I_L}}{3}r_C + \frac{w_{E_R I_L}}{3}r_R + b_{I_L}\right)$$
        $$\tau \dot r_{I_C} = -r_{I_C} + \phi\!\left(\frac{w_{E_L I_C}}{3}r_L + \frac{w_{E_R I_C}}{3}r_R + b_{I_C} + I_{\mathrm{opto}}(t)\right)$$
        $$\tau \dot r_{I_R} = -r_{I_R} + \phi\!\left(\frac{w_{E_L I_R}}{3}r_L + \frac{w_{E_C I_R}}{3}r_C + b_{I_R}\right)$$"""
    )

    def _fmt(value):
        return f"{float(value):.3g}"

    params = architecture_widget.value["params"]

    substituted_equations = mo.md(
        rf"""
        ## Current values substituted

        $$\phi(x)=\begin{{cases}}0,&x\le 0\\x^2,&0<x\le 1\\2\sqrt{{x-0.75}},&x>1\end{{cases}}$$
        $${_fmt(params["tau_e"])}\,\dot r_L = -r_L + \phi\!\left({_fmt(params["sL"])}r_L - {_fmt(params["w_IL_EL"])}r_{{I_L}} - {_fmt(params["w_IC_EL"])}r_{{I_C}} - {_fmt(params["w_IR_EL"])}r_{{I_R}} + I_L(t)\right)$$
        $${_fmt(params["tau_e"])}\,\dot r_C = -r_C + \phi\!\left({_fmt(params["sC"])}r_C - {_fmt(params["w_IL_EC"])}r_{{I_L}} - {_fmt(params["w_IC_EC"])}r_{{I_C}} - {_fmt(params["w_IR_EC"])}r_{{I_R}} + I_C(t)\right)$$
        $${_fmt(params["tau_e"])}\,\dot r_R = -r_R + \phi\!\left({_fmt(params["sR"])}r_R - {_fmt(params["w_IL_ER"])}r_{{I_L}} - {_fmt(params["w_IC_ER"])}r_{{I_C}} - {_fmt(params["w_IR_ER"])}r_{{I_R}} + I_R(t)\right)$$
        $${_fmt(params["tau_i"])}\,\dot r_{{I_L}} = -r_{{I_L}} + \phi\!\left({_fmt(params["w_EC_IL"] / 3.0)}r_C + {_fmt(params["w_ER_IL"] / 3.0)}r_R + {_fmt(params["i0_IL"])}\right)$$
        $${_fmt(params["tau_i"])}\,\dot r_{{I_C}} = -r_{{I_C}} + \phi\!\left({_fmt(params["w_EL_IC"] / 3.0)}r_L + {_fmt(params["w_ER_IC"] / 3.0)}r_R + {_fmt(params["i0_IC"])} + I_{{\mathrm{{opto}}}}(t)\right)$$
        $${_fmt(params["tau_i"])}\,\dot r_{{I_R}} = -r_{{I_R}} + \phi\!\left({_fmt(params["w_EL_IR"] / 3.0)}r_L + {_fmt(params["w_EC_IR"] / 3.0)}r_C + {_fmt(params["i0_IR"])}\right)$$
        """
    )

    mo.hstack([mo.vstack([symbolic_equations]), mo.vstack([substituted_equations])], align="end")
    return


@app.cell
def _(anywidget, mo, traitlets):
    class ArchitectureWidget(anywidget.AnyWidget):
        _esm = """
        function clamp(value, lo, hi) {
          return Math.max(lo, Math.min(hi, value));
        }

        function render({ model, el }) {
          el.classList.add("si-arch-widget");
          const params = () => ({...model.get("params")});
          const setParam = (key, value) => {
            const next = params();
            next[key] = value;
            model.set("params", next);
            model.save_changes();
            draw();
          };
          const inhibitoryToExcitatoryKeys = [
            "w_IL_EL", "w_IL_EC", "w_IL_ER",
            "w_IC_EL", "w_IC_EC", "w_IC_ER",
            "w_IR_EL", "w_IR_EC", "w_IR_ER",
          ];
          const excitatoryToInhibitoryKeys = [
            "w_EL_IC", "w_EL_IR",
            "w_EC_IL", "w_EC_IR",
            "w_ER_IL", "w_ER_IC",
          ];
          const selfInhibitoryKeys = ["w_IL_EL", "w_IC_EC", "w_IR_ER"];
          const crossInhibitoryKeys = inhibitoryToExcitatoryKeys.filter(
            (key) => !selfInhibitoryKeys.includes(key)
          );
          const setKeys = (next, keys, value) => {
            for (const key of keys) {
              next[key] = value;
            }
          };
          const applyPreset = (preset) => {
            const next = params();
            if (preset === "no_inhibition") {
              setKeys(next, inhibitoryToExcitatoryKeys, 0.0);
            } else if (preset === "all_one") {
              setKeys(next, excitatoryToInhibitoryKeys, 1.0);
              setKeys(next, inhibitoryToExcitatoryKeys, 1.0);
            } else if (preset === "excite_one_inhib_half") {
              setKeys(next, excitatoryToInhibitoryKeys, 1.0);
              setKeys(next, inhibitoryToExcitatoryKeys, 0.5);
            } else if (preset === "excite_one_cross_half_self_quarter") {
              setKeys(next, excitatoryToInhibitoryKeys, 1.0);
              setKeys(next, crossInhibitoryKeys, 0.5);
              setKeys(next, selfInhibitoryKeys, 0.25);
            } else {
              return;
            }
            model.set("params", next);
            model.save_changes();
            draw();
          };

          const intrinsicControls = [
            ["noise_amp", "noise", 0, 2.0, 0.01],
          ];

          const populationControls = [
            ["sL", "s_L", 0, 2, 0.02, [78, 306]],
            ["sC", "s_C", 0, 2, 0.02, [240, 40]],
            ["sR", "s_R", 0, 2, 0.02, [402, 306]],
            ["i0_IL", "b_IL", -1, 1, 0.02, [176, 226]],
            ["i0_IC", "b_IC", -1, 1, 0.02, [240, 224]],
            ["i0_IR", "b_IR", -1, 1, 0.02, [304, 226]],
          ];

          const edgeControls = [
            ["w_EL_IC", "E_L->I_C", 0, 2.5, 0.02],
            ["w_EL_IR", "E_L->I_R", 0, 2.5, 0.02],
            ["w_EC_IL", "E_C->I_L", 0, 2.5, 0.02],
            ["w_EC_IR", "E_C->I_R", 0, 2.5, 0.02],
            ["w_ER_IL", "E_R->I_L", 0, 2.5, 0.02],
            ["w_ER_IC", "E_R->I_C", 0, 2.5, 0.02],
            ["w_IL_EL", "I_L->E_L", 0, 2.5, 0.02],
            ["w_IL_EC", "I_L->E_C", 0, 2.5, 0.02],
            ["w_IL_ER", "I_L->E_R", 0, 2.5, 0.02],
            ["w_IC_EL", "I_C->E_L", 0, 2.5, 0.02],
            ["w_IC_EC", "I_C->E_C", 0, 2.5, 0.02],
            ["w_IC_ER", "I_C->E_R", 0, 2.5, 0.02],
            ["w_IR_EL", "I_R->E_L", 0, 2.5, 0.02],
            ["w_IR_EC", "I_R->E_C", 0, 2.5, 0.02],
            ["w_IR_ER", "I_R->E_R", 0, 2.5, 0.02],
          ];
          const edgeControlByKey = new Map(edgeControls.map((control) => [control[0], control]));
          const groupControls = [
            ["all_inhibition", "all I->E", inhibitoryToExcitatoryKeys, 0, 2.5, 0.02],
            ["cross_inhibition", "cross I->E", crossInhibitoryKeys, 0, 2.5, 0.02],
            ["self_inhibition", "self I->E", selfInhibitoryKeys, 0, 2.5, 0.02],
          ];
          const optoModes = [
            [0, "Opto E_C"],
            [1, "Opto I_C"],
            [2, "Opto E_C + I_C"],
          ];

          const nodePos = {
            E_L: [78, 252], E_C: [240, 62], E_R: [402, 252],
            I_L: [198, 186], I_C: [240, 154], I_R: [282, 186],
          };
          const edges = [
            ["E_L", "I_C", "w_EL_IC", "excit", -58], ["E_L", "I_R", "w_EL_IR", "excit", 36],
            ["E_C", "I_L", "w_EC_IL", "excit", 78], ["E_C", "I_R", "w_EC_IR", "excit", -78],
            ["E_R", "I_L", "w_ER_IL", "excit", -36], ["E_R", "I_C", "w_ER_IC", "excit", 58],
            ["I_L", "E_L", "w_IL_EL", "inhib", 0], ["I_L", "E_C", "w_IL_EC", "inhib", -52], ["I_L", "E_R", "w_IL_ER", "inhib", 62],
            ["I_C", "E_L", "w_IC_EL", "inhib", -50], ["I_C", "E_C", "w_IC_EC", "inhib", 0], ["I_C", "E_R", "w_IC_ER", "inhib", 50],
            ["I_R", "E_L", "w_IR_EL", "inhib", -62], ["I_R", "E_C", "w_IR_EC", "inhib", 52], ["I_R", "E_R", "w_IR_ER", "inhib", 0],
          ];
          const edgeOffsets = {
            w_EL_IC: {endAngleDeg: 150},
            w_EC_IL: {endAngleDeg: 150},
            w_EC_IR: {endAngleDeg: 40},
            w_ER_IC: {endAngleDeg: 40},
            w_IL_EC: {sx: -18, sy: 0},
            w_IR_EC: {sx: 18, sy: 0},
          };

          function tangle(key, label, lo, hi, step) {
            const p = params();
            const wrap = document.createElement("div");
            wrap.className = "tangle-row";
            const name = document.createElement("span");
            name.className = "tangle-label";
            name.textContent = label;
            const value = document.createElement("span");
            value.className = "tangle-value";
            value.textContent = Number(p[key]).toFixed(step < 0.02 ? 2 : 2);
            value.title = "Drag left/right to change";
            let startX = 0;
            let startValue = Number(p[key]);
            value.addEventListener("mousedown", (event) => {
              event.preventDefault();
              startX = event.clientX;
              startValue = Number(params()[key]);
              value.classList.add("dragging");
              const move = (moveEvent) => {
                const delta = Math.round((moveEvent.clientX - startX) / 5) * step;
                setParam(key, clamp(startValue + delta, lo, hi));
              };
              const up = () => {
                value.classList.remove("dragging");
                document.removeEventListener("mousemove", move);
                document.removeEventListener("mouseup", up);
              };
              document.addEventListener("mousemove", move);
              document.addEventListener("mouseup", up);
            });
            if (label) {
              wrap.appendChild(name);
            }
            wrap.appendChild(value);
            return wrap;
          }

          function groupTangle(label, keys, lo, hi, step) {
            const p = params();
            const wrap = document.createElement("div");
            wrap.className = "tangle-row group-tangle";
            const name = document.createElement("span");
            name.className = "tangle-label";
            name.textContent = label;
            const value = document.createElement("span");
            value.className = "tangle-value";
            const groupValue = keys.reduce((total, key) => total + Number(p[key]), 0) / keys.length;
            value.textContent = groupValue.toFixed(2);
            value.title = "Drag left/right to set this group";
            let startX = 0;
            let startValue = groupValue;
            value.addEventListener("mousedown", (event) => {
              event.preventDefault();
              startX = event.clientX;
              startValue = keys.reduce((total, key) => total + Number(params()[key]), 0) / keys.length;
              value.classList.add("dragging");
              const move = (moveEvent) => {
                const delta = Math.round((moveEvent.clientX - startX) / 5) * step;
                const nextValue = clamp(startValue + delta, lo, hi);
                const next = params();
                setKeys(next, keys, nextValue);
                model.set("params", next);
                model.save_changes();
                draw();
              };
              const up = () => {
                value.classList.remove("dragging");
                document.removeEventListener("mousemove", move);
                document.removeEventListener("mouseup", up);
              };
              document.addEventListener("mousemove", move);
              document.addEventListener("mouseup", up);
            });
            wrap.appendChild(name);
            wrap.appendChild(value);
            return wrap;
          }

          function svgTangle(key, label, lo, hi, step, x, y) {
            const box = document.createElementNS("http://www.w3.org/2000/svg", "foreignObject");
            box.setAttribute("x", x - 25);
            box.setAttribute("y", y - 11);
            box.setAttribute("width", "50");
            box.setAttribute("height", "24");
            const div = tangle(key, "", lo, hi, step);
            div.className = "floating-value";
            div.title = label;
            box.appendChild(div);
            return box;
          }

          function connectionPoints(a, b, key) {
            const [ax, ay] = nodePos[a];
            const [bx, by] = nodePos[b];
            const dx = bx - ax;
            const dy = by - ay;
            const len = Math.max(Math.hypot(dx, dy), 1);
            const nodeRadius = 31;
            const offsets = edgeOffsets[key] || {};
            let x2 = bx - (dx / len) * nodeRadius + (offsets.ex || 0);
            let y2 = by - (dy / len) * nodeRadius + (offsets.ey || 0);
            if (offsets.endAngleDeg !== undefined) {
              const theta = offsets.endAngleDeg * Math.PI / 180.0;
              x2 = bx + Math.cos(theta) * nodeRadius;
              y2 = by - Math.sin(theta) * nodeRadius;
            }
            return [
              ax + (dx / len) * nodeRadius + (offsets.sx || 0),
              ay + (dy / len) * nodeRadius + (offsets.sy || 0),
              x2,
              y2,
            ];
          }

          function endpointDot(x, y, edgeType) {
            const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            dot.setAttribute("cx", x);
            dot.setAttribute("cy", y);
            dot.setAttribute("r", "3.7");
            dot.setAttribute("class", `edge-dot ${edgeType}`);
            return dot;
          }

          function pathMidpoint(x1, y1, x2, y2, bend) {
            if (!bend) {
              return [0.5 * (x1 + x2), 0.5 * (y1 + y2)];
            }
            const mx = 0.5 * (x1 + x2);
            const my = 0.5 * (y1 + y2);
            const dx = x2 - x1;
            const dy = y2 - y1;
            const len = Math.max(Math.hypot(dx, dy), 1);
            const cx = mx - (dy / len) * bend;
            const cy = my + (dx / len) * bend;
            return [
              0.25 * x1 + 0.5 * cx + 0.25 * x2,
              0.25 * y1 + 0.5 * cy + 0.25 * y2,
            ];
          }

          function draw() {
            const p = params();
            el.innerHTML = "";
            const shell = document.createElement("div");
            shell.className = "arch-shell";
            const top = document.createElement("div");
            top.className = "arch-top";

            const left = document.createElement("div");
            left.className = "arch-canvas-wrap";
            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.setAttribute("viewBox", "0 0 480 350");
            svg.classList.add("arch-svg");
            svg.innerHTML = `
              <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#2f80ed"></path>
                </marker>
                <marker id="inhib-dot" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5.8" markerHeight="5.8" orient="auto">
                  <circle cx="5" cy="5" r="3.2" fill="#c0392b"></circle>
                </marker>
              </defs>`;
            const loops = [
              ["M 56 237 C 20 213 32 173 82 224"],
              ["M 218 44 C 222 10 258 10 262 44"],
              ["M 424 237 C 460 213 448 173 398 224"],
            ];
            for (const [d] of loops) {
              const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
              path.setAttribute("d", d);
              path.setAttribute("class", "edge self");
              path.setAttribute("marker-end", "url(#arrow)");
              svg.appendChild(path);
            }
            const edgeLabelPoints = [];
            for (const [a, b, key, edgeType, bend] of edges) {
              const [x1, y1, x2, y2] = connectionPoints(a, b, key);
              const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
              const strokeWidth = 0.8 + 2.0 * Math.min(1.4, Number(p[key])) / 1.4;
              if (bend) {
                const mx = 0.5 * (x1 + x2);
                const my = 0.5 * (y1 + y2);
                const dx = x2 - x1;
                const dy = y2 - y1;
                const len = Math.max(Math.hypot(dx, dy), 1);
                const cx = mx - (dy / len) * bend;
                const cy = my + (dx / len) * bend;
                path.setAttribute("d", `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`);
              } else {
                path.setAttribute("d", `M ${x1} ${y1} L ${x2} ${y2}`);
              }
              path.setAttribute("stroke-width", String(strokeWidth));
              path.setAttribute("class", `edge ${edgeType}`);
              path.setAttribute("marker-end", edgeType === "inhib" ? "url(#inhib-dot)" : "url(#arrow)");
              svg.appendChild(path);
              edgeLabelPoints.push([key, ...pathMidpoint(x1, y1, x2, y2, bend)]);
            }
            for (const [name, xy] of Object.entries(nodePos)) {
              const [x, y] = xy;
              const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
              circle.setAttribute("cx", x);
              circle.setAttribute("cy", y);
              circle.setAttribute("r", "28");
              const optoMode = Number(p.opto_mode ?? 1);
              const isOpto =
                (name === "E_C" && (optoMode === 0 || optoMode === 2)) ||
                (name === "I_C" && (optoMode === 1 || optoMode === 2));
              circle.setAttribute(
                "class",
                `${name.startsWith("E") ? "node exc" : "node inh"}${isOpto ? " opto" : ""}`,
              );
              svg.appendChild(circle);
              const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
              text.setAttribute("x", x);
              text.setAttribute("y", y + 5);
              text.setAttribute("text-anchor", "middle");
              text.setAttribute("class", "node-label");
              text.textContent = name.replace("_", "");
              svg.appendChild(text);
            }
            for (const [key, x, y] of edgeLabelPoints) {
              const [_, label, lo, hi, step] = edgeControlByKey.get(key);
              svg.appendChild(svgTangle(key, label, lo, hi, step, x, y));
            }
            for (const [key, label, lo, hi, step, xy] of populationControls) {
              svg.appendChild(svgTangle(key, label, lo, hi, step, xy[0], xy[1]));
            }
            left.appendChild(svg);

            const buttons = document.createElement("div");
            buttons.className = "arch-buttons";
            const optoSelect = document.createElement("select");
            optoSelect.className = "arch-select";
            optoSelect.title = "Where optogenetic current is injected";
            for (const [value, label] of optoModes) {
              const option = document.createElement("option");
              option.value = String(value);
              option.textContent = label;
              optoSelect.appendChild(option);
            }
            optoSelect.value = String(Number(p.opto_mode ?? 1));
            optoSelect.addEventListener("change", () => {
              setParam("opto_mode", Number(optoSelect.value));
            });
            const preset = document.createElement("select");
            preset.className = "arch-select";
            preset.title = "Apply parameter preset";
            for (const [value, label] of [
              ["custom", "Preset"],
              ["no_inhibition", "No inhibition"],
              ["all_one", "Excitation 1, inhibition 1"],
              ["excite_one_inhib_half", "Excitation 1, inhibition 0.5"],
              ["excite_one_cross_half_self_quarter", "Excitation 1, cross inhibition 0.5, self 0.25"],
            ]) {
              const option = document.createElement("option");
              option.value = value;
              option.textContent = label;
              preset.appendChild(option);
            }
            preset.addEventListener("change", () => {
              applyPreset(preset.value);
              preset.value = "custom";
            });
            const save = document.createElement("button");
            save.textContent = "Save JSON";
            save.addEventListener("click", () => {
              const blob = new Blob([JSON.stringify(params(), null, 2)], {type: "application/json"});
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = "selective_inhibition_params.json";
              a.click();
              URL.revokeObjectURL(url);
            });
            const load = document.createElement("button");
            load.textContent = "Load JSON";
            const input = document.createElement("input");
            input.type = "file";
            input.accept = "application/json";
            input.style.display = "none";
            input.addEventListener("change", async () => {
              const file = input.files[0];
              if (!file) return;
              const text = await file.text();
              const loaded = JSON.parse(text);
              model.set("params", {...params(), ...loaded});
              model.save_changes();
              draw();
            });
            load.addEventListener("click", () => input.click());
            buttons.appendChild(optoSelect);
            buttons.appendChild(preset);
            buttons.appendChild(save);
            buttons.appendChild(load);
            buttons.appendChild(input);
            left.appendChild(buttons);

            const right = document.createElement("div");
            right.className = "tangle-grid";
            const groups = [
              ["Inputs", intrinsicControls],
              ["Populations", populationControls],
            ];
            for (const [title, controls] of groups) {
              const heading = document.createElement("div");
              heading.className = "tangle-heading";
              heading.textContent = title;
              right.appendChild(heading);
              for (const [key, label, lo, hi, step] of controls) {
                right.appendChild(tangle(key, label, lo, hi, step));
              }
            }
            const groupHeading = document.createElement("div");
            groupHeading.className = "tangle-heading";
            groupHeading.textContent = "Grouped weights";
            right.appendChild(groupHeading);
            for (const [_, label, keys, lo, hi, step] of groupControls) {
              right.appendChild(groupTangle(label, keys, lo, hi, step));
            }
            top.appendChild(left);
            top.appendChild(right);
            shell.appendChild(top);

            const connectionPanel = document.createElement("div");
            connectionPanel.className = "connection-panel";
            const connectionHeading = document.createElement("div");
            connectionHeading.className = "tangle-heading";
            connectionHeading.textContent = "Connections";
            connectionPanel.appendChild(connectionHeading);
            const connectionGrid = document.createElement("div");
            connectionGrid.className = "connection-grid";
            for (const [key, label, lo, hi, step] of edgeControls) {
              connectionGrid.appendChild(tangle(key, label, lo, hi, step));
            }
            connectionPanel.appendChild(connectionGrid);
            shell.appendChild(connectionPanel);
            el.appendChild(shell);
          }

          model.on("change:params", draw);
          draw();
        }
        export default { render };
        """
        _css = """
        .si-arch-widget { display: block; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }
        .arch-shell { display: grid; grid-template-columns: 1fr; gap: 12px; align-items: start; }
        .arch-top { display: grid; grid-template-columns: minmax(320px, 1fr) 220px; gap: 14px; align-items: start; }
        .arch-canvas-wrap { border: 1px solid #d0d7de; border-radius: 8px; padding: 10px; background: #ffffff; }
        .arch-svg { width: 100%; min-height: 260px; display: block; }
        .edge { fill: none; opacity: 0.68; stroke-linecap: round; }
        .edge.excit { stroke: #2f80ed; }
        .edge.inhib { stroke: #c0392b; }
        .edge.self { fill: none; stroke: #2f80ed; stroke-width: 2.4; opacity: 0.7; }
        .edge-dot { stroke: #ffffff; stroke-width: 1.2; pointer-events: none; }
        .edge-dot.excit { fill: #2f80ed; }
        .edge-dot.inhib { fill: #c0392b; }
        .node { stroke-width: 2; }
        .node.exc { fill: #eaf3ff; stroke: #2f80ed; }
        .node.inh { fill: #fff0ee; stroke: #c0392b; }
        .node.opto { stroke: #f59e0b; stroke-width: 4; filter: drop-shadow(0 0 5px rgba(245,158,11,0.75)); }
        .node-label { font-size: 14px; font-weight: 700; fill: #1f2933; }
        .tangle-grid { display: grid; grid-template-columns: 1fr; gap: 7px; }
        .connection-panel { border: 1px solid #d0d7de; border-radius: 8px; padding: 8px; background: #ffffff; }
        .connection-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px 8px; }
        .tangle-row { display: flex; justify-content: space-between; gap: 10px; padding: 6px 8px; border: 1px solid #d0d7de; border-radius: 6px; background: #f6f8fa; }
        .edge-tangle { display: flex; justify-content: space-between; gap: 5px; padding: 4px 6px; border: 1px solid #b8c0cc; border-radius: 6px; background: rgba(255,255,255,0.94); box-shadow: 0 1px 3px rgba(15,23,42,0.12); font-size: 11px; }
        .tangle-heading { margin-top: 4px; color: #111827; font-size: 12px; font-weight: 750; text-transform: uppercase; letter-spacing: 0.04em; }
        .floating-value { display: inline-flex; justify-content: center; min-width: 20px; padding: 0; border: 0; background: transparent; box-shadow: none; font-size: 9px; pointer-events: all; text-shadow: 0 1px 2px #ffffff, 0 -1px 2px #ffffff; }
        .floating-value .tangle-value { font-size: 9px; color: #0b63ce; text-decoration: underline; }
        .edge-tangle .tangle-label { font-size: 9px; max-width: 72px; overflow: hidden; white-space: nowrap; }
        .edge-tangle .tangle-value { font-size: 11px; }
        .tangle-label { color: #374151; font-size: 12px; }
        .tangle-value { color: #0b63ce; text-decoration: underline; cursor: ew-resize; font-variant-numeric: tabular-nums; font-weight: 700; }
        .tangle-value.dragging { cursor: grabbing; color: #8a3ffc; }
        .arch-buttons { display: flex; gap: 8px; margin-top: 8px; }
        .arch-buttons button, .arch-select { border: 1px solid #b8c0cc; border-radius: 6px; background: #f6f8fa; padding: 6px 10px; cursor: pointer; }
        @media (prefers-color-scheme: dark) {
          .arch-canvas-wrap, .connection-panel { background: #111827; border-color: #374151; }
          .tangle-row { background: #18212f; border-color: #374151; }
          .edge-tangle { background: rgba(24,33,47,0.94); border-color: #4b5563; }
          .floating-value { background: transparent; text-shadow: 0 1px 2px #111827, 0 -1px 2px #111827; }
          .edge-dot { stroke: #111827; }
          .tangle-heading, .tangle-label, .node-label { color: #d1d5db; fill: #d1d5db; }
          .arch-buttons button, .arch-select { background: #18212f; color: #e5e7eb; border-color: #4b5563; }
        }
        @media (max-width: 900px) { .connection-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 760px) { .arch-top { grid-template-columns: 1fr; } }
        """
        params = traitlets.Dict(default_value={}).tag(sync=True)

    architecture_widget = mo.ui.anywidget(
        ArchitectureWidget(params=default_params())
    )
    architecture_widget
    return (architecture_widget,)


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    n_trials_slider = mo.ui.slider(
        start=300,
        stop=6000,
        step=300,
        value=900,
        label="number of trials",
        show_value=True,
    )
    seed_slider = mo.ui.slider(
        start=0, stop=9999, step=1, value=17, label="random seed", show_value=True
    )
    opto_min_slider = mo.ui.slider(
        start=-1.5, stop=0.0, step=0.05, value=-0.6, label="opto min", show_value=True
    )
    opto_max_slider = mo.ui.slider(
        start=0.0, stop=1.5, step=0.05, value=0.6, label="opto max", show_value=True
    )
    opto_steps_slider = mo.ui.slider(
        start=3,
        stop=31,
        step=2,
        value=7,
        label="opto steps",
        show_value=True,
    )
    hard_ttype_selector = mo.ui.dropdown(
        options={"VG": "VG", "DS": "DS", "DM": "DM", "DL": "DL"},
        value="DS",
        label="harder condition",
    )
    S_amp_slider = mo.ui.slider(
        start=0.0, stop=2.0, step=0.02, value=0.15, label="S_amp", show_value=True
    )
    U_amp_slider = mo.ui.slider(
        start=0.0, stop=4.0, step=0.02, value=2.0, label="U_amp", show_value=True
    )
    U_baseline_slider = mo.ui.slider(
        start=-3.0,
        stop=1.0,
        step=0.02,
        value=-1.0,
        label="U_baseline",
        show_value=True,
    )
    S_d_slider = mo.ui.slider(
        start=0.0, stop=1.5, step=0.02, value=0.32, label="S_d", show_value=True
    )
    save_gif_button = mo.ui.run_button(label="Save triangle GIF")
    controls = mo.vstack(
        [
            mo.md("## Simulation controls"),
            mo.hstack([n_trials_slider, seed_slider, hard_ttype_selector]),
            mo.hstack([opto_min_slider, opto_max_slider, opto_steps_slider]),
            mo.hstack([S_amp_slider, U_amp_slider, U_baseline_slider, S_d_slider]),
            save_gif_button,
        ]
    )
    controls
    return (
        S_amp_slider,
        S_d_slider,
        U_amp_slider,
        U_baseline_slider,
        hard_ttype_selector,
        n_trials_slider,
        opto_max_slider,
        opto_min_slider,
        opto_steps_slider,
        save_gif_button,
        seed_slider,
    )


@app.cell
def _(SIMULATION_CONDITIONS, np, pd):
    def make_synthetic_trials(n_trials, seed):
        rng = np.random.default_rng(int(seed))
        base = np.asarray(SIMULATION_CONDITIONS, dtype=object)
        reps = int(np.ceil(int(n_trials) / len(base)))
        grid = np.tile(base, (reps, 1))[: int(n_trials)]
        rng.shuffle(grid)

        ttype = grid[:, 0].astype(np.int32)
        ttype_labels = grid[:, 1].astype(object)
        stim = grid[:, 2].astype(np.int32)
        stim_labels = grid[:, 3].astype(object)
        side = grid[:, 4].astype(np.int32)
        side_labels = grid[:, 5].astype(object)
        times = np.zeros((len(grid), 4), dtype=np.float32)
        timing_by_condition = {
            (str(ttype_label), str(stimd_label)): timing
            for _, ttype_label, _, stimd_label, _, _, *timing in SIMULATION_CONDITIONS
        }
        for idx, (ttype_label, stimd_label) in enumerate(zip(ttype_labels, stim_labels, strict=False)):
            timing = timing_by_condition[(str(ttype_label), str(stimd_label))]
            times[idx, :] = np.asarray(timing, dtype=np.float32)

        return {
            "side": side,
            "ttype": ttype,
            "stim": stim,
            "side_label": side_labels,
            "ttype_c": ttype_labels,
            "stimd_c": stim_labels,
            "t1": times[:, 0],
            "t2": times[:, 1],
            "t3": times[:, 2],
            "t4": times[:, 3],
        }

    def trials_to_frame(trials):
        return pd.DataFrame(
            {
                "side": trials["side"],
                "stim": trials["stim"],
                "side_label": trials["side_label"],
                "ttype": trials["ttype"],
                "ttype_c": trials["ttype_c"],
                "stimd_c": trials["stimd_c"],
                "t1": trials["t1"],
                "t2": trials["t2"],
                "t3": trials["t3"],
                "t4": trials["t4"],
            }
        )

    return make_synthetic_trials, trials_to_frame


@app.cell
def _(jax, jnp):
    def _phi(x):
        return jnp.where(
            x <= 0.0,
            0.0,
            jnp.where(x <= 1.0, x * x, 2.0 * jnp.sqrt(jnp.maximum(x - 0.75, 0.0))),
        )

    def _stimulus_window(ttype, stim, side, t, t1, t2, t3, t4, s_amp, s_d):
        # Codes match the original Julia helper:
        # stim: 0 VG, 1 SS, 2 SM, 3 SL; delay: 0 DS, 1 DM, 2 DL.
        delay = jnp.maximum(ttype - 1, 0)

        ss_on = jnp.where(delay == 0, t2, jnp.where(delay == 1, t1, 0.0))
        ss_off = jnp.where(delay == 0, t3, jnp.where(delay == 1, t2, t1))
        sm_on = jnp.where(delay == 0, t1, 0.0)
        sm_off = jnp.where(delay == 0, t3, t2)
        sl_on = 0.0
        sl_off = t3

        onset = jnp.where(
            stim == 0,
            0.0,
            jnp.where(stim == 1, ss_on, jnp.where(stim == 2, sm_on, sl_on)),
        )
        offset = jnp.where(
            stim == 0,
            t4,
            jnp.where(stim == 1, ss_off, jnp.where(stim == 2, sm_off, sl_off)),
        )
        plateau = (t >= onset) & (t <= offset)
        tail = (t > offset) & (t <= offset + s_d) & (s_d > 0.0)
        tail_value = s_amp * (1.0 - (t - offset) / jnp.maximum(s_d, 1e-6))
        s_val = jnp.where(plateau, s_amp, jnp.where(tail, tail_value, 0.0))
        return jax.nn.one_hot(side, 3, dtype=jnp.float32) * s_val

    def _urgency_value(t, t1, t2, t3, t4, u_amp, u_baseline):
        w1 = 1.0 / jnp.maximum(t1, 1e-6)
        w2 = 1.0 / jnp.maximum(t2 - t1, 1e-6)
        w3 = 1.0 / jnp.maximum(t3 - t2, 1e-6)
        w4 = 1.0 / jnp.maximum(t4 - t3, 1e-6)
        r1 = jnp.clip(t * w1, 0.0, 1.0)
        r2 = jnp.clip((t - t1) * w2, 0.0, 1.0)
        r3 = jnp.clip((t - t2) * w3, 0.0, 1.0)
        r4 = jnp.clip((t - t3) * w4, 0.0, 1.0)
        return u_baseline + 0.25 * u_amp * (r1 + r2 + r3 + r4)

    def _single_trial(
        opto_amp,
        side,
        stim,
        ttype,
        t1,
        t2,
        t3,
        t4,
        noise,
        params,
        w_ei,
        w_ie,
        t_grid,
        dt,
        s_amp,
        u_amp,
        u_baseline,
        s_d,
    ):
        e0 = jnp.zeros(3, dtype=jnp.float32)
        i0 = jnp.zeros(3, dtype=jnp.float32)
        state0 = (e0, i0)
        opto_base_vec = jax.nn.one_hot(
            params["opto_target"].astype(jnp.int32), 3, dtype=jnp.float32
        ) * opto_amp
        opto_mode = params["opto_mode"].astype(jnp.int32)
        opto_e_vec = opto_base_vec * jnp.where(
            (opto_mode == 0) | (opto_mode == 2), 1.0, 0.0
        )
        opto_i_vec = opto_base_vec * jnp.where(
            (opto_mode == 1) | (opto_mode == 2), 1.0, 0.0
        )
        sqrt_dt = jnp.sqrt(dt)

        def step(carry, inputs):
            e, inh = carry
            t, eps = inputs
            stim_input = _stimulus_window(ttype, stim, side, t, t1, t2, t3, t4, s_amp, s_d)
            urgency = _urgency_value(t, t1, t2, t3, t4, u_amp, u_baseline)
            ext = stim_input + urgency
            s_vec = jnp.asarray(
                [params["sL"], params["sC"], params["sR"]], dtype=jnp.float32
            )
            i0_i = jnp.asarray(
                [params["i0_IL"], params["i0_IC"], params["i0_IR"]],
                dtype=jnp.float32,
            )
            xi_e = params["noise_amp"] * eps[:3] * sqrt_dt
            xi_i = params["noise_amp"] * eps[3:] * sqrt_dt

            x_e = s_vec * e - (w_ie @ inh) + ext + opto_e_vec
            x_i = (w_ei @ e) / 3.0 + i0_i + opto_i_vec
            f_e = (-e + _phi(x_e)) / params["tau_e"]
            f_i = (-inh + _phi(x_i)) / params["tau_i"]
            e_pred = e + f_e * dt + xi_e
            i_pred = inh + f_i * dt + xi_i

            x_e2 = s_vec * e_pred - (w_ie @ i_pred) + ext + opto_e_vec
            x_i2 = (w_ei @ e_pred) / 3.0 + i0_i + opto_i_vec
            f_e2 = (-e_pred + _phi(x_e2)) / params["tau_e"]
            f_i2 = (-i_pred + _phi(x_i2)) / params["tau_i"]

            e_next = jnp.clip(e + 0.5 * (f_e + f_e2) * dt + xi_e, 0.0, 3.0)
            i_next = jnp.clip(inh + 0.5 * (f_i + f_i2) * dt + xi_i, 0.0, 3.0)
            return (e_next, i_next), e_next

        final, e_hist = jax.lax.scan(step, state0, (t_grid, noise))
        e_final, _ = final
        final_choice = jnp.argmax(e_final).astype(jnp.int32)
        valid = jnp.array(1.0, dtype=jnp.float32)
        return final_choice, valid, t4, e_final, e_hist[-1]

    def make_jax_simulator(dt, n_steps):
        t_grid = jnp.arange(n_steps, dtype=jnp.float32) * jnp.float32(dt)

        @jax.jit
        def simulate_sweep(
            opto_amps,
            side,
            stim,
            ttype,
            t1,
            t2,
            t3,
            t4,
            noise,
            params,
            s_amp,
            u_amp,
            u_baseline,
            s_d,
        ):
            w_ei = jnp.asarray(
                [
                    [0.0, params["w_EC_IL"], params["w_ER_IL"]],
                    [params["w_EL_IC"], 0.0, params["w_ER_IC"]],
                    [params["w_EL_IR"], params["w_EC_IR"], 0.0],
                ],
                dtype=jnp.float32,
            )
            w_ie = jnp.asarray(
                [
                    [params["w_IL_EL"], params["w_IC_EL"], params["w_IR_EL"]],
                    [params["w_IL_EC"], params["w_IC_EC"], params["w_IR_EC"]],
                    [params["w_IL_ER"], params["w_IC_ER"], params["w_IR_ER"]],
                ],
                dtype=jnp.float32,
            )

            def sim_amp(opto_amp):
                return jax.vmap(
                    lambda sd, st, tt, a, b, c, d, eps: _single_trial(
                        opto_amp,
                        sd,
                        st,
                        tt,
                        a,
                        b,
                        c,
                        d,
                        eps,
                        params,
                        w_ei,
                        w_ie,
                        t_grid,
                        jnp.float32(dt),
                        s_amp,
                        u_amp,
                        u_baseline,
                        s_d,
                    ),
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
                )(side, stim, ttype, t1, t2, t3, t4, noise)

            return jax.vmap(sim_amp)(opto_amps)

        return simulate_sweep

    return (make_jax_simulator,)


@app.cell
def _(
    DT,
    N_STEPS,
    S_amp_slider,
    S_d_slider,
    U_amp_slider,
    U_baseline_slider,
    architecture_widget,
    jnp,
    make_jax_simulator,
    make_synthetic_trials,
    n_trials_slider,
    np,
    opto_max_slider,
    opto_min_slider,
    opto_steps_slider,
    pd,
    seed_slider,
    trials_to_frame,
):
    def _params_for_jax(raw_params):
        numeric = {}
        for key, value in raw_params.items():
            numeric[key] = jnp.asarray(value, dtype=jnp.float32)
        return numeric

    def _summarize_results(trials, opto_amps, choices, valid, first_time):
        rows = []
        trial_df = trials_to_frame(trials)
        choice_np = np.asarray(choices)
        valid_np = np.asarray(valid)
        first_time_np = np.asarray(first_time)
        for amp_index, opto_amp in enumerate(np.asarray(opto_amps, dtype=float)):
            light = "off" if np.isclose(opto_amp, 0.0) else "on"
            for ttype_label in sorted(trial_df["ttype_c"].unique()):
                stim_labels = sorted(
                    trial_df.loc[trial_df["ttype_c"] == ttype_label, "stimd_c"].unique()
                )
                for stim_label in stim_labels:
                    for side_label in ("L", "C", "R"):
                        mask = (
                            (trial_df["ttype_c"].to_numpy() == ttype_label)
                            & (trial_df["stimd_c"].to_numpy() == stim_label)
                            & (trial_df["side_label"].to_numpy() == side_label)
                        )
                        if not mask.any():
                            continue
                        condition_indices = np.flatnonzero(mask)
                        n_blocks = min(12, max(1, len(condition_indices) // 20))
                        for block_index, block_indices in enumerate(
                            np.array_split(condition_indices, n_blocks)
                        ):
                            ch = choice_np[amp_index, block_indices]
                            va = valid_np[amp_index, block_indices].astype(bool)
                            target = trials["side"][block_indices]
                            valid_ch = ch[va]
                            n_valid = int(va.sum())
                            denom = max(1, len(ch))
                            probs = [
                                float(np.mean(valid_ch == idx)) if n_valid else 0.0
                                for idx in range(3)
                            ]
                            p_correct = (
                                float(np.mean(valid_ch == target[va])) if n_valid else 0.0
                            )
                            rows.append(
                                {
                                    "opto_amp": float(opto_amp),
                                    "light": light,
                                    "block": int(block_index),
                                    "ttype_c": ttype_label,
                                    "stimd_c": str(stim_label),
                                    "side_label": side_label,
                                    "trial_group": "center"
                                    if side_label == "C"
                                    else "sides",
                                    "n_trials": int(len(ch)),
                                    "n_valid": n_valid,
                                    "p_completed": float(n_valid / denom),
                                    "p_invalid": float(1.0 - n_valid / denom),
                                    "pL": probs[0],
                                    "pC": probs[1],
                                    "pR": probs[2],
                                    "p_correct": p_correct,
                                    "mean_decision_time": float(
                                        np.mean(first_time_np[amp_index, block_indices][va])
                                    )
                                    if n_valid
                                    else np.nan,
                                }
                            )
        return pd.DataFrame(rows)

    def _summarize_sil_results(opto_amps, choices, valid):
        rows = []
        choice_np = np.asarray(choices)
        valid_np = np.asarray(valid)
        for amp_index, opto_amp in enumerate(np.asarray(opto_amps, dtype=float)):
            va = valid_np[amp_index].astype(bool)
            valid_ch = choice_np[amp_index, va]
            n_valid = int(va.sum())
            probs = [
                float(np.mean(valid_ch == idx)) if n_valid else 0.0
                for idx in range(3)
            ]
            rows.append(
                {
                    "opto_amp": float(opto_amp),
                    "light": "off" if np.isclose(opto_amp, 0.0) else "on",
                    "ttype_c": "SIL",
                    "stimd_c": "SIL",
                    "side_label": "SIL",
                    "n_trials": int(choice_np.shape[1]),
                    "n_valid": n_valid,
                    "pL": probs[0],
                    "pC": probs[1],
                    "pR": probs[2],
                }
            )
        return pd.DataFrame(rows)

    def run_synthetic_sweep():
        trials = make_synthetic_trials(n_trials_slider.value, seed_slider.value)
        rng = np.random.default_rng(int(seed_slider.value) + 991)
        noise = rng.normal(
            0.0, 1.0, size=(int(n_trials_slider.value), N_STEPS, 6)
        ).astype(np.float32)
        opto_amps = np.linspace(
            float(opto_min_slider.value),
            float(opto_max_slider.value),
            int(opto_steps_slider.value),
            dtype=np.float32,
        )
        if not np.any(np.isclose(opto_amps, 0.0)):
            opto_amps = np.sort(np.append(opto_amps, np.float32(0.0))).astype(np.float32)

        sim = make_jax_simulator(DT, N_STEPS)
        params = _params_for_jax(architecture_widget.value["params"])
        choices, valid, first_time, _, _ = sim(
            jnp.asarray(opto_amps, dtype=jnp.float32),
            jnp.asarray(trials["side"], dtype=jnp.int32),
            jnp.asarray(trials["stim"], dtype=jnp.int32),
            jnp.asarray(trials["ttype"], dtype=jnp.int32),
            jnp.asarray(trials["t1"], dtype=jnp.float32),
            jnp.asarray(trials["t2"], dtype=jnp.float32),
            jnp.asarray(trials["t3"], dtype=jnp.float32),
            jnp.asarray(trials["t4"], dtype=jnp.float32),
            jnp.asarray(noise, dtype=jnp.float32),
            params,
            jnp.float32(S_amp_slider.value),
            jnp.float32(U_amp_slider.value),
            jnp.float32(U_baseline_slider.value),
            jnp.float32(S_d_slider.value),
        )
        sweep = _summarize_results(trials, opto_amps, choices, valid, first_time)
        sil_choices, sil_valid, _, _, _ = sim(
            jnp.asarray(opto_amps, dtype=jnp.float32),
            jnp.asarray(trials["side"], dtype=jnp.int32),
            jnp.asarray(trials["stim"], dtype=jnp.int32),
            jnp.asarray(trials["ttype"], dtype=jnp.int32),
            jnp.asarray(trials["t1"], dtype=jnp.float32),
            jnp.asarray(trials["t2"], dtype=jnp.float32),
            jnp.asarray(trials["t3"], dtype=jnp.float32),
            jnp.asarray(trials["t4"], dtype=jnp.float32),
            jnp.asarray(noise, dtype=jnp.float32),
            params,
            jnp.float32(0.0),
            jnp.float32(U_amp_slider.value),
            jnp.float32(U_baseline_slider.value),
            jnp.float32(S_d_slider.value),
        )
        sil_triangle = _summarize_sil_results(opto_amps, sil_choices, sil_valid)
        trial_df = trials_to_frame(trials)
        return sweep, sil_triangle, trial_df

    sweep_df, sil_triangle_df, synthetic_trials_df = run_synthetic_sweep()
    return sil_triangle_df, sweep_df


@app.cell
def _(mo, sweep_df):
    triangle_amp_options = sorted(float(v) for v in sweep_df["opto_amp"].unique())
    triangle_amp_labels = [f"{value:+.3g}" for value in triangle_amp_options]
    triangle_amp_selector = mo.ui.dropdown(
        options=triangle_amp_labels,
        value=f"{min(triangle_amp_options, key=abs):+.3g}",
        label="triangle current",
    )
    triangle_amp_selector
    return (triangle_amp_selector,)


@app.cell
def _(np):
    def _resolve_boxplot_colors(colors, n, name):
        if isinstance(colors, str):
            return [colors] * n
        resolved = list(colors)
        if len(resolved) != n:
            raise ValueError(f"{name} must have length {n}, got {len(resolved)}.")
        return resolved

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

            for median, color in zip(
                box["medians"], valid_median_colors, strict=False
            ):
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
def _(
    CHOICE_LABELS,
    SIDE_COLORS,
    custom_boxplot,
    hard_ttype_selector,
    mo,
    np,
    plt,
    sil_triangle_df,
    sweep_df,
    triangle_amp_selector,
):
    def _curve_data(data, ttype=None):
        panel = data.copy()
        if ttype is not None:
            panel = panel[panel["ttype_c"] == ttype]
        grouped = (
            panel.groupby(["opto_amp", "trial_group"], as_index=False)["p_correct"]
            .mean()
            .pivot(index="opto_amp", columns="trial_group", values="p_correct")
            .reset_index()
            .rename(columns={"center": "frac_center", "sides": "frac_sides"})
        )
        if "frac_center" not in grouped:
            grouped["frac_center"] = np.nan
        if "frac_sides" not in grouped:
            grouped["frac_sides"] = np.nan
        baseline = grouped.iloc[(grouped["opto_amp"].abs()).argsort()[:1]]
        b_center = float(baseline["frac_center"].iloc[0])
        b_sides = float(baseline["frac_sides"].iloc[0])
        grouped["delta_center"] = grouped["frac_center"] - b_center
        grouped["delta_sides"] = grouped["frac_sides"] - b_sides
        return grouped.sort_values("opto_amp")

    def _plot_delta(ax, data, title):
        for mask, color, label in (
            (data["opto_amp"] <= 0.0, "tab:blue", "inhibition"),
            (data["opto_amp"] >= 0.0, "tab:red", "excitation"),
        ):
            seg = data[mask].sort_values("opto_amp")
            if len(seg) >= 2:
                ax.plot(
                    seg["delta_center"],
                    seg["delta_sides"],
                    color=color,
                    lw=2.4,
                    label=label,
                )
        pts = ax.scatter(
            data["delta_center"],
            data["delta_sides"],
            c=data["opto_amp"],
            cmap="RdBu_r",
            edgecolor="white",
            linewidth=0.4,
            s=48,
            zorder=3,
        )
        ax.axhline(0, color="#8A8A8A", lw=1)
        ax.axvline(0, color="#8A8A8A", lw=1)
        ax.set_xlabel(r"$\Delta$Acc center")
        ax.set_ylabel(r"$\Delta$Acc side")
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        return pts

    def _heat_data(data, mask):
        panel = data[mask].copy()
        panel["amp_abs"] = panel["opto_amp"].abs().round(6)
        grouped = (
            panel.groupby(["amp_abs", "trial_group"], as_index=False)["p_correct"]
            .mean()
            .pivot(index="amp_abs", columns="trial_group", values="p_correct")
            .reset_index()
            .sort_values("amp_abs")
        )
        for col in ("center", "sides"):
            if col not in grouped:
                grouped[col] = np.nan
        return grouped

    def _ternary_xy(p_l, p_c, p_r):
        total = p_l + p_c + p_r
        if total <= 0 or not np.isfinite(total):
            return np.nan, np.nan
        p_l, p_c, p_r = p_l / total, p_c / total, p_r / total
        return p_r + 0.5 * p_c, (np.sqrt(3.0) / 2.0) * p_c

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

    def _plot_sil_dot(ax, sil_data, opto_amp):
        sil = sil_data[np.isclose(sil_data["opto_amp"], opto_amp)]
        if sil.empty:
            return
        row = sil.iloc[0]
        x, y = _ternary_xy(row["pL"], row["pC"], row["pR"])
        ax.scatter(
            x,
            y,
            s=94,
            color="#111827",
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

    all_curve = _curve_data(sweep_df)
    hard_curve = _curve_data(sweep_df, hard_ttype_selector.value)

    acc_fig, acc_ax = plt.subplots(figsize=(4, 4))
    pts = _plot_delta(acc_ax, all_curve, "All synthetic trials")
    acc_fig.colorbar(pts, ax=acc_ax, fraction=0.046, pad=0.04, label="IC opto current")
    acc_fig.tight_layout()

    hard_fig, hard_ax = plt.subplots(figsize=(4, 4))
    pts = _plot_delta(hard_ax, hard_curve, f"ttype_c={hard_ttype_selector.value}")
    hard_fig.colorbar(pts, ax=hard_ax, fraction=0.046, pad=0.04, label="IC opto current")
    hard_fig.tight_layout()

    heat_fig, heat_axes = plt.subplots(1, 2, figsize=(8.0, 2.9), constrained_layout=True)
    heat_im = None
    for ax, (title, mask, color) in zip(
        heat_axes,
        [
            ("Inhibition", sweep_df["opto_amp"] <= 0.0, "tab:blue"),
            ("Excitation", sweep_df["opto_amp"] >= 0.0, "tab:red"),
        ],
        strict=False,
    ):
        hdata = _heat_data(sweep_df, mask)
        values = hdata[["center", "sides"]].to_numpy(dtype=float).T
        heat_im = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Center", "Sides"])
        ax.set_title(title, color=color)
        ax.set_xlabel("|IC current|")
        ax.set_xticks(range(len(hdata)))
        ax.set_xticklabels([f"{v:.2g}" for v in hdata["amp_abs"]], rotation=45, ha="right")
    heat_axes[0].set_ylabel("Trial group")
    heat_fig.colorbar(heat_im, ax=heat_axes, fraction=0.046, pad=0.04, label="Model accuracy")

    requested_amp = float(triangle_amp_selector.value)
    amps = sorted(float(v) for v in sweep_df["opto_amp"].unique())
    selected_amp = requested_amp
    tri = sweep_df[np.isclose(sweep_df["opto_amp"], selected_amp)].copy()
    tri_mean = (
        tri.groupby(["ttype_c", "side_label"], as_index=False)[["pL", "pC", "pR"]]
        .mean()
        .sort_values(["ttype_c", "side_label"])
    )
    ttype_order = [label for label in ("VG", "DS", "DM", "DL") if label in set(sweep_df["ttype_c"])]
    alpha_values = np.linspace(1.0, 0.35, max(len(ttype_order), 1))
    alpha_map = {label: float(alpha_values[idx]) for idx, label in enumerate(ttype_order)}
    triangle_fig, triangle_ax = plt.subplots(figsize=(4, 4))
    _draw_triangle_axes(triangle_ax)
    for _, row in tri_mean.iterrows():
        x, y = _ternary_xy(row["pL"], row["pC"], row["pR"])
        triangle_ax.scatter(
            x,
            y,
            s=76,
            color=SIDE_COLORS[row["side_label"]],
            alpha=alpha_map[row["ttype_c"]],
            edgecolor="#163B4A",
            linewidth=0.8,
            zorder=3,
        )
    _plot_sil_dot(triangle_ax, sil_triangle_df, selected_amp)
    triangle_ax.set_title(f"IC current {selected_amp:+.3g}")
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
        for side, color in SIDE_COLORS.items()
    ]
    ttype_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="#163B4A",
            markerfacecolor="#163B4A",
            alpha=alpha,
            label=ttype,
        )
        for ttype, alpha in alpha_map.items()
    ]
    triangle_ax.legend(
        handles=side_handles
        + ttype_handles
        + [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#111827",
                markeredgecolor="white",
                label="SIL",
            )
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )
    triangle_fig.tight_layout()

    prob_fig, prob_axes = plt.subplots(
        2, 3, figsize=(10, 4.0), sharey=True, constrained_layout=True
    )
    prob_cols = ["pL", "pC", "pR"]
    light_amp = selected_amp if not np.isclose(selected_amp, 0.0) else max(amps)
    for row_idx, ttype in enumerate(["VG", "DS"]):
        for col_idx, side in enumerate(CHOICE_LABELS):
            ax = prob_axes[row_idx, col_idx]
            panel = sweep_df[
                (sweep_df["ttype_c"] == ttype) & (sweep_df["side_label"] == side)
            ]
            off = panel[np.isclose(panel["opto_amp"], 0.0)]
            on = panel[np.isclose(panel["opto_amp"], light_amp)]
            data = []
            positions = []
            colors = []
            n_choices = len(prob_cols)
            n_light = 2
            group_width = 0.9
            hue_width = group_width / n_light
            group_centers = np.arange(n_choices, dtype=float)
            light_color = "#C0392B" if light_amp > 0 else "#2F80ED"
            for choice_idx, prob_col in enumerate(prob_cols):
                data.extend([off[prob_col].to_numpy(), on[prob_col].to_numpy()])
                positions.extend(
                    [
                        choice_idx + (light_idx - (n_light - 1) / 2.0) * hue_width
                        for light_idx in range(n_light)
                    ]
                )
                colors.extend(["#9CA3AF", light_color])
            custom_boxplot(
                ax,
                data,
                positions=positions,
                widths=hue_width * 1,
                median_colors=colors,
                box_alpha=1.0,
                showfliers=False,
                showcaps=False,
            )
            ax.set_xticks(group_centers)
            ax.set_xticklabels(["L", "C", "R"])
            ax.set_xlim(group_centers[0] - 0.65, group_centers[-1] + 0.65)
            ax.set_ylim(0, 1)
            ax.set_title(f"{ttype}, stim {side}")
            if col_idx == 0:
                ax.set_ylabel("p(choice)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_aspect(2)
    prob_fig.suptitle(f"Choice probability distributions: off vs on ({light_amp:+.3g})")

    line_fig, line_axes = plt.subplots(
        1, 3, figsize=(12.0, 4.0), sharey=True, constrained_layout=True
    )

    def _categorical_accuracy_panel(ax, group_col, title, data, order):
        panel = data[data[group_col].isin(order)].copy()
        off = panel[np.isclose(panel["opto_amp"], 0.0)]
        on = panel[np.isclose(panel["opto_amp"], light_amp)]
        x = np.arange(len(order))
        for label, sub, color in (
            ("light off", off, "#6B7280"),
            ("light on", on, "#C0392B" if light_amp > 0 else "#2F80ED"),
        ):
            grouped = sub.groupby(group_col)["p_correct"].mean()
            y = np.asarray([grouped.get(category, np.nan) for category in order], dtype=float)
            ax.plot(x, y, marker="o", lw=1.9, ms=4.0, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel(group_col)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.axhline(
            1.0 / 3.0,
            color="#8A8A8A",
            lw=1.0,
            ls="--",
            alpha=0.8,
            label="chance",
        )
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    _categorical_accuracy_panel(
        line_axes[0],
        "ttype_c",
        "Accuracy by ttype_c",
        sweep_df,
        [label for label in ("VG", "DS", "DM", "DL") if label in set(sweep_df["ttype_c"])],
    )
    _categorical_accuracy_panel(
        line_axes[1],
        "stimd_c",
        "Accuracy by stimd_c, DS only",
        sweep_df[sweep_df["ttype_c"] == "DS"],
        [
            label
            for label in ("SL", "SM", "SS")
            if label in set(sweep_df.loc[sweep_df["ttype_c"] == "DS", "stimd_c"])
        ],
    )
    _categorical_accuracy_panel(
        line_axes[2],
        "ttype_c",
        "Accuracy by ttype_c, SS only",
        sweep_df[sweep_df["stimd_c"] == "SS"],
        [
            label
            for label in ("DS", "DM", "DL")
            if label in set(sweep_df.loc[sweep_df["stimd_c"] == "SS", "ttype_c"])
        ],
    )
    line_axes[0].set_ylabel("Model accuracy")

    mo.vstack(
        [
            mo.md(
                f"**Synthetic sweep:** {len(sweep_df)} grouped rows from "
                f"{int(sweep_df['n_trials'].sum())} condition-trials."
            ),
            mo.hstack([acc_fig, hard_fig, triangle_fig]),
            line_fig,
            # heat_fig,
            prob_fig,
            # mo.ui.table(sweep_df, page_size=18),
        ]
    )
    return


@app.cell
def _(
    RESULT_DIR,
    SIDE_COLORS,
    imageio,
    mo,
    np,
    plt,
    save_gif_button,
    sil_triangle_df,
    sweep_df,
):
    def _plot_triangle_frame(ax, data, sil_data, opto_amp):
        _draw_triangle_axes(ax)
        ttype_order = [label for label in ("VG", "DS", "DM", "DL") if label in set(data["ttype_c"])]
        alpha_values = np.linspace(1.0, 0.35, max(len(ttype_order), 1))
        alpha_map = {label: float(alpha_values[idx]) for idx, label in enumerate(ttype_order)}
        tri = data[np.isclose(data["opto_amp"], opto_amp)]
        tri_mean = (
            tri.groupby(["ttype_c", "side_label"], as_index=False)[["pL", "pC", "pR"]]
            .mean()
            .sort_values(["ttype_c", "side_label"])
        )
        for _, row in tri_mean.iterrows():
            x, y = _ternary_xy(row["pL"], row["pC"], row["pR"])
            ax.scatter(
                x,
                y,
                s=76,
                color=SIDE_COLORS[row["side_label"]],
                alpha=alpha_map[row["ttype_c"]],
                edgecolor="#163B4A",
                linewidth=0.8,
                zorder=3,
            )
        sil = sil_data[np.isclose(sil_data["opto_amp"], opto_amp)]
        if not sil.empty:
            row = sil.iloc[0]
            x, y = _ternary_xy(row["pL"], row["pC"], row["pR"])
            ax.scatter(
                x,
                y,
                s=94,
                color="#111827",
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )
        ax.set_title(f"IC current {float(opto_amp):+.3g}")

    def write_triangle_gif():
        frames = []
        for opto_amp in sorted(float(v) for v in sweep_df["opto_amp"].unique()):
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            _plot_triangle_frame(ax, sweep_df, sil_triangle_df, opto_amp)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            plt.close(fig)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULT_DIR / "synthetic_selective_inhibition_triangle.gif"
        imageio.mimsave(out, frames, fps=4, loop=0)
        return out

    if save_gif_button.value:
        gif_path = write_triangle_gif()
        status = mo.md(f"Saved triangle GIF: `{gif_path}`")
    else:
        status = mo.md("Click **Save triangle GIF** to export the opto-current animation.")
    status
    return


if __name__ == "__main__":
    app.run()
