import os, json, sys, re
import pathlib
base_path = pathlib.Path().resolve().parents[1]
PROJECT_ROOT = pathlib.Path("../").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
import paths
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns


def entropy_nats(p):
    p = np.asarray(p, dtype=float)
    p = p[(p > 0) & np.isfinite(p)]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0

def H_conditional(df_subj, cond_cols=('stimd_c','ttype_c','x_c'), resp_col='r_c'):
    if df_subj.empty: return np.nan
    cols = list(cond_cols) + [resp_col]
    d = df_subj[cols].copy()
    for c in cols:
        d[c] = d[c].astype('string').str.strip()
    d = d[d[resp_col].isin({'L','C','R'})]
    if d.empty: return np.nan

    N = float(len(d))
    classes = ['L','C','R']
    H = 0.0
    for _, df_s in d.groupby(list(cond_cols)):
        n_s = float(len(df_s))
        counts = df_s[resp_col].value_counts()
        q = np.array([counts.get(c, 0)/n_s for c in classes], dtype=float)
        H += (n_s / N) * entropy_nats(q)
    return float(H)

def rel_vs_ceiling_from_values(nll_mean, H_cs):
    if not np.isfinite(nll_mean) or not np.isfinite(H_cs): return np.nan
    denom = np.log(3) - H_cs
    if denom <= 0: return np.nan
    val = 1.0 - ((nll_mean - H_cs) / denom)
    return float(np.clip(val, 0.0, 1.0))

def make_balanced_subset(df_subj, cond_cols=('stimd_c','ttype_c'), max_total=10000, seed=42):
    d = df_subj.copy()
    d['cond'] = d[cond_cols[0]].astype(str) + '_' + d[cond_cols[1]].astype(str)
    n_conds = d['cond'].nunique()
    if n_conds == 0: return d
    per_cond = max(1, max_total // n_conds)
    samples = []
    rng = np.random.RandomState()
    for cond, dfc in d.groupby('cond'):
        n_sample = min(len(dfc), per_cond)
        samples.append(dfc.sample(n=n_sample, random_state=rng))
    out = pd.concat(samples, ignore_index=True)
    out.drop(columns='cond', inplace=True)
    return out

def subject_name(s: str) -> str:
    if s is None: return ''
    s = str(s).strip()
    if '_' in s:
        return s.split('_', 1)[0]
    m = re.match(r'^([A-Za-z]+[0-9]+)', s)
    return m.group(1) if m else s

def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'

CANONICAL_THETA_NAMES = [
    'sL', 'sC', 'sR',
    'noise_amp',
    'S_amplitude', 'S_d',
    'U_int_amplitude', 'U_int_baseline', 'U_int_onset',
    'U_ext_amplitude'
]

labels = {'sL': f'$s_L$', 'sC': f'$s_C$', 'sR': f'$s_R$', 'noise_amp': f'$\sigma$', 'S_amplitude': r'$S^{amp}$', 'S_d': f'$S^d$', 'U_int_amplitude': r'$U_{int}^{amp}$', 'U_int_baseline': r'$U_{int}^{baseline}$',
          'U_int_onset': r'$U_{int}^{onset}$', 'U_ext_amplitude': r'$U_{ext}^{amp}$', 'rel_vs_ceiling_bal': r'Goodness of fit'}
set2 = sns.color_palette("Set2", 8)
custom_palette = {'sL': set2[0],'sC': set2[1],'sR': set2[2],'U_int_amplitude': 
                  '#C38DFF', 'U_int_baseline':  '#C38DFF','U_int_onset':'#C38DFF',
                  'S_amplitude':'#FFA64C', 'S_d': '#FFA64C',
                  'U_ext_amplitude': '#3385FF',
                  'noise_amp':       '#999999', 'nll/trial': '#999999',
                  'rel_vs_ceiling_bal': '#999999'}

def process_params(df, subdirs=None):
    rows = []
    if subdirs is None: 
        subdirs = [ os.path.join(paths.PARAMS_DIR, name) for name in os.listdir(paths.PARAMS_DIR) if (os.path.isdir(os.path.join(paths.PARAMS_DIR, name)) and name != 'not_used') ]
    else:
        subdirs = [os.path.join(paths.PARAMS_DIR, subdir) for subdir in subdirs]

    print(f"df size: {len(df)}")
    print(f"Detectados modelos: {[os.path.basename(s) for s in subdirs]}")
    nll_eval_df = pd.read_csv(os.path.join(paths.PARAMS_DIR, 'params_evaluated.csv'), sep=';')
    nll_eval_df['subject'] = nll_eval_df['subject'].astype(str).str.strip()
    nll_eval_df['model']   = nll_eval_df['model'].astype(str).str.strip()

    for subdir in subdirs:
        model_name = os.path.basename(subdir)
        for filename in os.listdir(subdir):
            if filename.endswith(".json") or filename.endswith(".pkl"):
                # print(f"Procesando {filename} en modelo {model_name}...")
                subject = filename.replace('result_', '')[:3]
                if filename.endswith(".json"):
                    with open(os.path.join(subdir, filename), "r") as f:
                        result_obj = json.load(f)
                if filename.endswith(".pkl"):
                    with open(os.path.join(subdir, filename), "rb") as f:
                        result_obj = pkl.load(f)
                x = result_obj["x"]
                row = {"subject": subject[:3], "model": model_name}
                for name, val in zip(CANONICAL_THETA_NAMES, x):
                    row[name] = val

                row["nll"] = result_obj["fval"]
                row["nll/trial"] = (result_obj["fval"]/result_obj["n_trials"]
                                    if result_obj.get("n_trials", 0) > 0 else np.nan)
                mask = (
                    (nll_eval_df['subject'] == row['subject']) &
                    (nll_eval_df['model']   == row['model'])
                )

                for pname in CANONICAL_THETA_NAMES:
                    if pname in nll_eval_df.columns:
                        mask &= np.isclose(nll_eval_df[pname].astype(float), row[pname], rtol=1e-6, atol=1e-8)
                match = nll_eval_df[mask]

                if not match.empty and 'nll_eval' in match.columns:
                    row['nll_total'] = float(match['nll_eval'].iloc[0])
                else:
                    print(f"⚠️ No se encontró nll_eval para subject={row['subject']}, "f"model={row['model']}")
                    row['nll_total'] = np.nan

                rows.append(row)

    params_df2 = pd.DataFrame(rows)

    df_all = df.copy()
    # df_all = df_all[df_all['timepoint_4']<= np.percentile(df_all['timepoint_4'], 95)]

    H_full_map = {}
    H_bal_map  = {}
    H_bal_map2 = {}
    for subj, dfg in df_all.groupby('subject'):
        H_full = H_conditional(dfg, cond_cols=('stimd_c','ttype_c','x_c'), resp_col='r_c')
        H_full_map[subj] = H_full
        dfg_bal = make_balanced_subset(dfg, cond_cols=('stimd_c','ttype_c'), max_total=10000)
        H_bal = H_conditional(dfg_bal, cond_cols=('stimd_c','ttype_c','x_c'), resp_col='r_c')
        H_bal_map[subj] = H_bal
        dfg_bal2 = make_balanced_subset(dfg, cond_cols=('stimd_c','ttype_c'), max_total=7500, seed=42)
        H_bal2 = H_conditional(dfg_bal2, cond_cols=('stimd_c','ttype_c','x_c'), resp_col='r_c')
        H_bal_map2[subj] = H_bal2

        # print(f"[{subj}] H_full = {H_full:.6f} | H_bal = {H_bal:.6f}")


    params_df2['subject'] = params_df2['subject'].apply(subject_name)
    params_df2['nll_total/trial'] = params_df2['nll_total'] / df_all.groupby('subject').size().reindex(params_df2['subject']).values
    params_df2['rel_vs_ceiling_full'] = params_df2.apply(
        lambda r: rel_vs_ceiling_from_values(r.get('nll_total/trial', np.nan), H_full_map.get(r['subject'], np.nan)), axis=1)
    params_df2['rel_vs_ceiling_bal'] = params_df2.apply(
        lambda r: rel_vs_ceiling_from_values(r.get('nll/trial', np.nan), H_bal_map.get(r['subject'], np.nan)), axis=1)
    mask_reduced2 = params_df2['model'] == 'spatial_reduced2'
    params_df2.loc[mask_reduced2, 'rel_vs_ceiling_bal'] = params_df2[mask_reduced2].apply(
        lambda r: rel_vs_ceiling_from_values(r.get('nll/trial', np.nan), H_bal_map2.get(r['subject'], np.nan)), axis=1)

    for _, r in params_df2.iterrows():
        subj = r['subject']
        nll_mean = r['nll/trial']
        H_bal = H_bal_map.get(subj, np.nan)
        if np.isfinite(nll_mean) and np.isfinite(H_bal) and (nll_mean + 1e-9) < H_bal:
            print(f"⚠️ NLL/trial ({nll_mean:.6f}) < H_bal(C|S) ({H_bal:.6f}) for {subj}.")


    # mostrar y guardar
    # display(params_df2.sort_values('subject'))
    return params_df2
    # params_df2.to_csv(f'{paths.PARAMS_DIR}/params_balanced.csv', index=False)