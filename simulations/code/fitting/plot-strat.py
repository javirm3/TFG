import pandas as pd
import numpy as np
import sys, matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import seaborn as sns
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import paths
from helpers.plots import truncate_colormap,get_onset_offset,model_mean_for_trials, build_SU_for_trial, side_map
from helpers.sim_core_numba import simulate_counts_one_trial_heun

dt = 0.1 / 40
th = (0.5, 0.5, 0.5)
M = 300
N_T4_BINS=3
N_X_BINS=7
SUBSAMPLE_PER_BIN=600
trunc_purples=truncate_colormap('Purples_r',0,0.7)
trunc_oranges=truncate_colormap('Oranges',0.3,1.0)

def model_nested_worker(args):
    obin,x_cat,df_subj,theta,subsample,model_type=args
    bin_trials=df_subj[(df_subj['outer_bin']==obin)&(df_subj['x_bin']==x_cat)].copy()
    if len(bin_trials)==0: return obin,x_cat,np.nan,0.0
    if len(bin_trials)>subsample:
        parts=[]
        for s in ['L','C','R']:
            g=bin_trials[bin_trials['x_c']==s]
            k=min(len(g),subsample//3)
            if k>0: parts.append(g.sample(k,random_state=0))
        if len(parts): bin_trials=pd.concat(parts,axis=0)
    m,s=model_mean_for_trials(bin_trials,theta=theta,type=model_type)
    return obin,x_cat,m,s

def compute_nested_curves(df_subj,theta,outer_var,x_var,n_t4_bins,n_x_bins,subsample,model_type):
    df_subj=df_subj.copy()
    df_subj['outer_bin'],outer_edges=pd.qcut(df_subj[outer_var],q=n_t4_bins,retbins=True,duplicates='drop')
    df_subj['x_bin'] = pd.Series(index=df_subj.index,dtype='object')
    outer_centros=(df_subj.groupby('outer_bin',observed=True)[outer_var]
                   .median().rename('outer_center').reset_index()
                   .sort_values('outer_center'))
    outer_order=list(outer_centros['outer_bin'])
    outer_centers=outer_centros['outer_center'].to_numpy()
    outer_info={}
    for i,obin in enumerate(outer_order):
        group=df_subj[df_subj['outer_bin']==obin].copy()
        if len(group)==0: 
            continue
        group['x_bin_cat'],x_edges=pd.qcut(group[x_var],n_x_bins,retbins=True,duplicates='drop')
        group['x_bin']=group['x_bin_cat'].astype(str)
        df_subj.loc[group.index,'x_bin']=group['x_bin']

        x_centros=(group.groupby('x_bin',observed=True)[x_var]
                .median().rename('x_center').reset_index()
                .sort_values('x_center'))
        x_centers=x_centros['x_center'].to_numpy()
        x_order=list(x_centros['x_bin'])
        data_mean=[];data_sem=[]
        for cat in x_order:
            bin_trials=group[group['x_bin']==cat].copy()
            if len(bin_trials)>subsample:
                parts=[]
                for s in ['L','C','R']:
                    g=bin_trials[bin_trials['x_c']==s]
                    k=min(len(g),subsample//3)
                    if k>0: parts.append(g.sample(k,random_state=0))
                if len(parts): bin_trials=pd.concat(parts,axis=0)
            if len(bin_trials)==0:
                data_mean.append(np.nan);data_sem.append(0.0)
            else:
                acc=bin_trials['correct_bool'].to_numpy(float)
                p_hat=float(acc.mean())
                data_mean.append(p_hat)
                if len(acc)>1:
                    n=len(acc);data_sem.append(np.sqrt(p_hat*(1-p_hat)/n))
                else:
                    data_sem.append(0.0)
        data_mean=np.asarray(data_mean,float)
        data_sem=np.asarray(data_sem,float)
        model_mean=np.full_like(data_mean,np.nan,float)
        model_sem=np.zeros_like(data_mean,float)
        cat_to_idx={cat:j for j,cat in enumerate(x_order)}
        outer_info[obin]=dict(idx=i,
                              outer_center=float(outer_centers[i]),
                              outer_range=(float(outer_edges[i]),float(outer_edges[i+1])),
                              x_centers=x_centers,
                              x_order=x_order,
                              cat_to_idx=cat_to_idx,
                              data_mean=data_mean,
                              data_sem=data_sem,
                              model_mean=model_mean,
                              model_sem=model_sem)
    if not outer_info: return [],outer_edges
    jobs=[]
    for obin,info in outer_info.items():
        for cat in info['x_order']:
            jobs.append((obin,cat,df_subj,theta,subsample,model_type))
    n_workers=max(1,mp.cpu_count()-5)
    print(f"  Lanzando pool con {n_workers} workers para {x_var}")
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures=[exe.submit(model_nested_worker,job) for job in jobs]
        for f in tqdm(as_completed(futures),total=len(futures),desc=f"  Modelo {x_var}",leave=False):
            obin,x_cat,m,s=f.result()
            info=outer_info.get(obin)
            if info is None: continue
            j=info['cat_to_idx'].get(x_cat)
            if j is None: continue
            info['model_mean'][j]=m
            info['model_sem'][j]=s
    curves=[]
    for obin in outer_order:
        info=outer_info.get(obin)
        if info is None: continue
        curves.append(dict(outer_bin=obin,
                           outer_center=info['outer_center'],
                           outer_range=info['outer_range'],
                           x_centers=info['x_centers'],
                           data_mean=info['data_mean'],
                           data_sem=info['data_sem'],
                           model_mean=info['model_mean'],
                           model_sem=info['model_sem']))
    curves.sort(key=lambda d:d['outer_center'])
    return curves,outer_edges

def plot_delay_stim_nested(subject,delay_curves,stim_curves,n_delay,n_stim,outer_var_delay,outer_var_stim):
    fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True)
    axd,axs=axes
    # delay
    for i,info in enumerate(delay_curves):
        col=trunc_purples(0.3+0.4*i/max(1,len(delay_curves)-1))
        xc=info['x_centers']
        rng=info['outer_range']
        axd.errorbar(xc,info['model_mean'],yerr=info['model_sem'],
                     fmt='-s',color=col,capsize=3,markersize=5,
                     label=f'{outer_var_delay} {rng[0]:.2f}-{rng[1]:.2f}')
        axd.errorbar(xc,info['data_mean'],yerr=info['data_sem'],
                     fmt='--o',color=col,capsize=3,markersize=5)
    axd.axhspan(0,1/3,color='gray',alpha=0.2,zorder=0)
    axd.set_xlabel('Delay duration (s)')
    axd.set_ylabel('Frac. correct responses')
    axd.set_title(f'{subject} – Delay (n={n_delay})')
    axd.set_ylim(0.2,1.05)
    axd.legend(frameon=False,fontsize=8)
    # stim
    for i,info in enumerate(stim_curves):
        col=trunc_oranges(0.3+0.5*i/max(1,len(stim_curves)-1))
        xc=info['x_centers']
        rng=info['outer_range']
        axs.errorbar(xc,info['model_mean'],yerr=info['model_sem'],
                     fmt='-s',color=col,capsize=3,markersize=5,
                     label=f'{outer_var_stim} {rng[0]:.2f}-{rng[1]:.2f}')
        axs.errorbar(xc,info['data_mean'],yerr=info['data_sem'],
                     fmt='--o',color=col,capsize=3,markersize=5)
    axs.axhspan(0,1/3,color='gray',alpha=0.2,zorder=0)
    axs.set_xlabel('Stimulus duration (çs)')
    axs.set_title(f'{subject} – Stim (n={n_stim})')
    axs.legend(frameon=False,fontsize=8)
    sns.despine()
    fig.tight_layout()
    fname=f'fig_delay_stim_nested_{subject}.png'
    fig.savefig(fname,dpi=300)
    print(f'  Figure saved to {fname}')
    plt.close(fig)

def model_pc_per_trial(trials, theta, type="temporal"):
    """
    Devuelve un array con la probabilidad de acierto del modelo por trial.
    Mantiene el orden de 'trials'.
    """
    pcs = []
    for _, r in trials.iterrows():
        side_true = r['x_c']
        code = side_map[side_true]
        S_t, U_t, N = build_SU_for_trial(r, theta, type)
        if N <= 0:
            pcs.append(np.nan)
            continue

        mL, mC, mR = simulate_counts_one_trial_heun(
            S_t, U_t, code,
            theta['sL'], theta['sC'], theta['sR'],
            theta['noise'], dt,
            th[0], th[1], th[2],
            M
        )
        if side_true == 'L':
            p = mL / M
        elif side_true == 'C':
            p = mC / M
        else:
            p = mR / M
        pcs.append(float(p))
    return np.asarray(pcs, dtype=float)


if __name__=="__main__":
    params_df=pd.read_csv(f'{paths.PARAMS_DIR}/params_best_models.csv',sep=';')
    # params_df=params_df[params_df['subject']=='A92']
    # params_plot=params_df.loc[params_df.groupby("subject")["nll/trial"].idxmin()]
    df=pd.read_csv(f'{paths.DATA_PATH}/df_filtered.csv')
    if 'onset' not in df.columns or 'offset' not in df.columns:
        df[['onset','offset']]=df.apply(lambda r:pd.Series(get_onset_offset(r['stimd_c'],r['ttype_c'],
                                                                            r['timepoint_1'],r['timepoint_2'],
                                                                            r['timepoint_3'],r['timepoint_4'])),
                                        axis=1)
    df['stim_duration']=df['offset']-df['onset']
    df['delay_duration']=df['timepoint_4']-df['offset']
    sns.set();sns.set_style('white');sns.set_style('ticks');sns.set_context("talk",font_scale=1)
    subjects=list(params_df['subject'].unique())
    print("Subjects:",subjects)
    for subject in subjects:
        print(f'\nProcesando sujeto {subject}')
        row=params_df.loc[params_df['subject']==subject].iloc[0]
        model_type='spatial' if 'spatial' in str(row['model']) else 'temporal'
        if pd.isna(row['U_int_baseline']): row['U_int_baseline']=-1.0
        if pd.isna(row['U_int_onset']): row['U_int_onset']=0.0
        if pd.isna(row['noise_amp']): row['noise_amp']=1.0
        if pd.isna(row['U_ext_amplitude']): row['U_ext_amplitude']=0.0
        theta=dict(sL=float(row['sL']),sC=float(row['sC']),sR=float(row['sR']),
                   noise=float(row['noise_amp']),S_amp=float(row['S_amplitude']),
                   S_d=float(row['S_d']),U_amp=float(row['U_int_amplitude']),
                   U_base=float(row['U_int_baseline']),U_on=float(row['U_int_onset']),
                   U_ext_amp=float(row['U_ext_amplitude']))
        print("  theta =",theta)
        df_delay=df[(df['subject']==subject)&(df['onset']==0)&
                    (df['timepoint_4']<np.percentile(df['timepoint_4'],95))].copy()
        n_delay=len(df_delay)
        if n_delay==0:
            print("  (sin trials delay)");continue
        delay_curves,_=compute_nested_curves(df_delay,theta,
                                             outer_var='offset',
                                             x_var='delay_duration',
                                             n_t4_bins=N_T4_BINS,
                                             n_x_bins=N_X_BINS,
                                             subsample=SUBSAMPLE_PER_BIN,
                                             model_type=model_type)
        df_stim=df[(df['subject']==subject)&(df['ttype_c']!='VG')&
                   (df['timepoint_4']<np.percentile(df['timepoint_4'],95))].copy()
        n_stim=len(df_stim)
        if n_stim==0:
            print("  (sin trials stim)");continue
        stim_curves,_=compute_nested_curves(df_stim,theta,
                                            outer_var='offset',
                                            x_var='stim_duration',
                                            n_t4_bins=N_T4_BINS,
                                            n_x_bins=N_X_BINS,
                                            subsample=SUBSAMPLE_PER_BIN,
                                            model_type=model_type)
        plot_delay_stim_nested(subject,delay_curves,stim_curves,n_delay,n_stim,
                               outer_var_delay='offset',outer_var_stim='offset')
        
            # -------- FIGURA: Stim duration vs Delay duration --------
    from matplotlib import cm

    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

        # ---------- SCATTER: stim_duration vs delay_duration ----------
    # Usamos el sujeto definido arriba
    df_subj = df[df['subject'] == subject].copy()

    # Nos quedamos con delays DS, DM, DL (quitamos VG y SIL)
    cats_scatter = ['DS', 'DM', 'DL']
    df_scatter = df_subj[df_subj['ttype_c'].isin(cats_scatter)].copy()

    # Construimos theta del sujeto (igual que en model_data_mean)
    row_theta = params_plot.loc[params_plot['subject'] == subject].iloc[0].copy()
    if pd.isna(row_theta['U_int_baseline']):  row_theta['U_int_baseline'] = -1.0
    if pd.isna(row_theta['U_int_onset']):     row_theta['U_int_onset']    = 0.0
    if pd.isna(row_theta['noise_amp']):       row_theta['noise_amp']      = 1.0
    if pd.isna(row_theta['U_ext_amplitude']): row_theta['U_ext_amplitude']= 0.0

    theta_subj = dict(
        sL=float(row_theta['sL']), sC=float(row_theta['sC']), sR=float(row_theta['sR']),
        noise=float(row_theta['noise_amp']),
        S_amp=float(row_theta['S_amplitude']), S_d=float(row_theta['S_d']),
        U_amp=float(row_theta['U_int_amplitude']),
        U_base=float(row_theta['U_int_baseline']),
        U_on=float(row_theta['U_int_onset'])
    )

    # Calculamos P(correct) del modelo trial a trial (usamos el modelo espacial)
    print(f"Simulando prob. de acierto trial a trial para sujeto {subject}...")
    pcs = model_pc_per_trial(df_scatter, theta_subj, type="spatial")
    df_scatter['p_model'] = pcs

    # Panel a) por categoría DS/DM/DL
    sns.scatterplot(
        data=df_scatter,
        x='stim_duration', y='delay_duration',
        hue='ttype_c',
        hue_order=['DS', 'DM', 'DL'],
        palette=delay_duration_colors,
        s=25, alpha=0.7, edgecolor='none',
        ax=axes_sc[0]
    )
    axes_sc[0].set_title('a) Por tipo de delay')
    axes_sc[0].set_xlabel('Stimulus duration (s)')
    axes_sc[0].set_ylabel('Delay duration (s)')
    axes_sc[0].legend(title='Delay', frameon=False)

    # Panel b) Data: correcto / incorrecto
    correct_palette = {True: '#2E7D32', False: '#C62828'}  # verde / rojo
    sns.scatterplot(
        data=df_scatter,
        x='stim_duration', y='delay_duration',
        hue='correct_bool',
        palette=correct_palette,
        s=25, alpha=0.7, edgecolor='none',
        ax=axes_sc[1]
    )
    axes_sc[1].set_title('b) Data: correcto vs incorrecto')
    axes_sc[1].set_xlabel('Stimulus duration (s)')
    axes_sc[1].set_ylabel('')
    axes_sc[1].legend(title='Correcto', frameon=False)

    # Panel c) Model: prob. de acierto en gradiente
    cmap_model = cm.get_cmap('RdYlGn')  # rojo (0) -> verde (1)
    sc = axes_sc[2].scatter(
        df_scatter['stim_duration'],
        df_scatter['delay_duration'],
        c=df_scatter['p_model'],
        cmap=cmap_model,
        vmin=0.0, vmax=1.0,
        s=25, alpha=0.7
    )
    axes_sc[2].set_title('c) Modelo: P(correct)')
    axes_sc[2].set_xlabel('Stimulus duration (s)')
    axes_sc[2].set_ylabel('')

    cbar = fig_sc.colorbar(sc, ax=axes_sc[2])
    cbar.set_label('P(correct) modelo')

    # Opcional: mismo rango de ejes para ver bien la nube
    for ax in axes_sc:
        ax.set_xlim(df_scatter['stim_duration'].min() - 0.1,
                    df_scatter['stim_duration'].max() + 0.1)
        ax.set_ylim(df_scatter['delay_duration'].min() - 0.1,
                    df_scatter['delay_duration'].max() + 0.1)

    sns.despine()
    plt.tight_layout()
    plt.savefig('fig_scatter_stim_vs_delay_subject_{}_3panels.png'.format(subject), dpi=300)
    plt.savefig('fig_scatter_stim_vs_delay_subject_{}_3panels.svg'.format(subject))