"""Data panels for the v4 Psychophysiology figures, computed directly from ./data.
Each panel is drawn at its final printed size (mm) so that fonts stay at 7-8 pt after
assembly in PowerPoint (revision_20261028_figures_v4_pptx.py). Outputs: SVG + 600-dpi PNG.
Run: python revision_20261028_figures_v4_panels.py
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import mne

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'Figures' / 'v4' / 'panels'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261028

# Palette shared with the PowerPoint builder
CTRL, SUB = '#2F6A9F', '#D07A2B'          # groups
CORE, EXT = '#262626', '#8A8A8A'          # cohorts
INK, GRID, SHADE = '#1A1A1A', '#D9D9D9', '#EFEFEF'
MODEL = {'MSAF': '#2F6A9F', 'SDS/SAS': '#7A7A7A', 'Combined': '#D07A2B'}

mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    'lines.linewidth': 1.0, 'legend.frameon': False})
MM = 1 / 25.4
NUMBERS = {}


def read(rel):
    return pd.read_csv(ROOT / rel, encoding='utf-8-sig')


def fig(w, h):
    return plt.figure(figsize=(w * MM, h * MM))


def save(f, name):
    for ext, kw in [('svg', {}), ('png', {'dpi': 600})]:
        f.savefig(OUT / f'{name}.{ext}', **kw)
    plt.close(f)


def pooled_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def jitter(n, width, rng):
    return rng.uniform(-width, width, n)


# ---------------------------------------------------------------- shared data
part = read('data/01_participants/participant_characteristics.csv')
analyzable = part[part.analyzable.eq(1)][['pid', 'sample', 'group']]


# ---------------------------------------------------------------- Figure 2C
def fig2_vas():
    r = read('data/04_laboratory/lab_subjective_ratings.csv')
    r = r[r.session.eq('D0')].pivot(index='pid', columns='timepoint', values='vas_stress')[['T1', 'T2', 'T3']]
    f = fig(66, 50); ax = f.add_axes([0.2, 0.2, 0.76, 0.74])
    x = np.arange(3)
    for row in r.sample(n=min(120, len(r)), random_state=SEED).values:
        ax.plot(x, row, color='#BDBDBD', lw=0.25, alpha=0.6, zorder=1)
    bp = ax.boxplot([r[c].values for c in r.columns], positions=x, widths=0.32, patch_artist=True,
                    showfliers=False, medianprops=dict(color=INK, lw=0.8),
                    boxprops=dict(facecolor='white', edgecolor=INK, lw=0.6),
                    whiskerprops=dict(color=INK, lw=0.6), capprops=dict(color=INK, lw=0.6), zorder=2)
    m = r.mean(); ci = 1.96 * r.std() / np.sqrt(len(r))
    ax.errorbar(x + 0.27, m, yerr=ci, fmt='o', ms=3.2, color=SUB, mfc=SUB, capsize=2, lw=0.8, zorder=3)
    ax.set_xticks(x, ['Baseline', 'Post-stress', 'Recovery'])
    ax.set_ylabel('Subjective stress (VAS, 0–100)'); ax.set_ylim(0, 100); ax.set_xlim(-0.5, 2.6)
    save(f, 'F2C_vas')
    NUMBERS['vas'] = dict(n=int(len(r)), means=m.round(2).tolist())


# ---------------------------------------------------------------- Figure 3
PHASES = ['baseline', 'control', 'stress1', 'stress2', 'stress3', 'recovery1', 'recovery2']
PHLAB = ['B', 'C', 'S1', 'S2', 'S3', 'R1', 'R2']


def fig3_phases():
    lab = read('data/04_laboratory/lab_phase_features_ecg_eda.csv')
    d = lab[lab.session.eq('D0')].merge(analyzable, on='pid')
    d = d[d['sample'].eq('core')]
    spec = [('hr_bpm', 'HR (bpm)', 'F3A_hr'), ('ln_rmssd', 'lnRMSSD (ln ms)', 'F3B_lnrmssd'), ('scl_us', 'SCL (µS)', 'F3C_scl')]
    for col, ylab, name in spec:
        f = fig(58, 50); ax = f.add_axes([0.22, 0.17, 0.75, 0.78])
        ax.axvspan(1.5, 4.5, color=SHADE, lw=0, zorder=0)
        for g, c, off in [('control', CTRL, -0.08), ('subclinical', SUB, 0.08)]:
            z = d[d.group.eq(g)].groupby('phase')[col].agg(['mean', 'std', 'count']).reindex(PHASES)
            se = z['std'] / np.sqrt(z['count'])
            xs = np.arange(7) + off
            ax.errorbar(xs, z['mean'], yerr=se, color=c, marker='o', ms=2.6, lw=0.9, capsize=1.6, elinewidth=0.6)
        ax.set_xticks(range(7), PHLAB); ax.set_xlim(-0.5, 6.5); ax.set_ylabel(ylab)
        ax.text(3, ax.get_ylim()[1], 'Stress', ha='center', va='top', fontsize=6.5, color='#6B6B6B')
        save(f, name)
    NUMBERS['phase_n'] = d.groupby('group').pid.nunique().to_dict()


MARKERS = [('lab_lnrmssd_base', 'Resting lnRMSSD'), ('lab_lnhf_base', 'Resting lnHF-HRV'),
           ('lab_hr_react', 'HR reactivity'), ('lab_lnrmssd_react', 'lnRMSSD reactivity'),
           ('eeg_aper_mean', 'Aperiodic exponent')]


def fig3_forest():
    h1 = read('audit_20261028_h1_student_markers.csv')
    f = fig(98, 72); ax = f.add_axes([0.33, 0.17, 0.64, 0.8])
    ys = np.arange(len(MARKERS))[::-1] * 1.0
    for (feat, lab), y in zip(MARKERS, ys):
        for s, c, off, pcol in [('core', CORE, 0.17, 'p_student_fdr14'), ('extension', EXT, -0.17, 'p_student_fdr5_replication')]:
            r = h1[h1['sample'].eq(s) & h1.feature.eq(feat)].iloc[0]
            sig = r[pcol] < 0.05
            ax.plot([r.d_lo, r.d_hi], [y + off] * 2, color=c, lw=0.9)
            ax.plot(r.d, y + off, marker='o' if s == 'core' else 's', ms=3.6, color=c, mfc=c if sig else 'white', mew=0.9)
    ax.axvline(0, color='#7F7F7F', lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys, [m[1] for m in MARKERS]); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-0.85, 1.0); ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_xlabel("Cohen's d (subclinical − control)")
    ax.set_ylim(-0.6, len(MARKERS) - 0.4)
    save(f, 'F3D_forest')


def fig3_reactivity():
    pf = read('data/06_analysis_ready/person_level_features.csv').drop(columns=['sample']).merge(analyzable, on='pid')
    rng = np.random.default_rng(SEED)
    f = fig(78, 72); ax = f.add_axes([0.2, 0.17, 0.77, 0.73])
    pos = {('core', 'control'): 0, ('core', 'subclinical'): 1, ('extension', 'control'): 2.6, ('extension', 'subclinical'): 3.6}
    for (s, g), x in pos.items():
        v = pf[pf['sample'].eq(s) & pf.group.eq(g)].lab_lnrmssd_react.dropna().values
        c = CTRL if g == 'control' else SUB
        ax.scatter(x + jitter(len(v), 0.16, rng), v, s=3, color=c, alpha=0.45, lw=0, zorder=1)
        ax.boxplot([v], positions=[x], widths=0.5, showfliers=False, patch_artist=True,
                   medianprops=dict(color=INK, lw=0.9), boxprops=dict(facecolor='none', edgecolor=INK, lw=0.7),
                   whiskerprops=dict(color=INK, lw=0.6), capprops=dict(color=INK, lw=0.6), zorder=2)
    ax.axhline(0, color='#7F7F7F', lw=0.5, ls=(0, (3, 2)))
    top = pf.lab_lnrmssd_react.max()
    h1 = read('audit_20261028_h1_student_markers.csv')
    for s, x in [('core', 0.5), ('extension', 3.1)]:
        r = h1[h1['sample'].eq(s) & h1.feature.eq('lab_lnrmssd_react')].iloc[0]
        ax.text(x, top + 0.05, f'd = {r.d:.2f}\n[{r.d_lo:.2f}, {r.d_hi:.2f}]', ha='center', va='bottom', fontsize=6.8)
    ax.set_xticks([0, 1, 2.6, 3.6], ['Control', 'Subclinical', 'Control', 'Subclinical'])
    ax.set_ylim(pf.lab_lnrmssd_react.min() - 0.05, top + 0.22)
    ax.set_ylabel('lnRMSSD reactivity\n(stress − control, ln ms)')
    for x, lab in [(0.5, 'Core'), (3.1, 'Extension')]:
        ax.text(x, -0.15, lab, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.5, fontweight='bold')
    save(f, 'F3E_reactivity')


# ---------------------------------------------------------------- Figure 4 (EEG topography)
def eeg_table():
    e = read('data/04_laboratory/lab_eeg_features.csv')
    q = read('data/04_laboratory/lab_eeg_quality.csv')
    bad = set(q[q.session.eq('D0') & q.excluded_gt20pct_bad.eq(1)].pid)
    e = e[e.session.eq('D0') & e.condition.eq('rest_ec') & ~e.pid.isin(bad)].merge(analyzable, on='pid')
    for b in ['delta', 'theta', 'alpha', 'beta']:
        e[f'ln_{b}'] = np.log(e[f'{b}_abs'])
    return e


def topo_info(chs):
    info = mne.create_info(list(chs), 500., 'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))
    return info


def channel_stats(e, sample, feat):
    x = e[e['sample'].eq(sample)]
    rows = []
    for ch, g in x.groupby('channel', sort=False):
        a = g[g.group.eq('subclinical')][feat]; b = g[g.group.eq('control')][feat]
        rows.append(dict(channel=ch, mean_ctrl=b.mean(), mean_sub=a.mean(),
                         t=stats.ttest_ind(a, b, equal_var=False).statistic))
    return pd.DataFrame(rows)


def draw_topo(ax, values, info, vlim, cmap, mask=None):
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap, vlim=vlim, contours=4,
                                 sensors=True, mask=mask, extrapolate='head', outlines='head',
                                 sphere=(0.0, 0.007, 0.0, 0.11),
                                 mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor='k',
                                                  linewidth=0, markersize=2.6))
    return im


def fig4_topos():
    e = eeg_table()
    mx = read('audit_20261028_h1_run1_eeg_maxT.csv')
    chans = list(e.channel.drop_duplicates())
    info = topo_info(chans)
    # rows A (core) and B (extension): control mean, subclinical mean, Welch t map
    allmeans = []
    st = {s: channel_stats(e, s, 'aperiodic_exponent').set_index('channel').loc[chans] for s in ['core', 'extension']}
    for s in st.values():
        allmeans += list(s.mean_ctrl) + list(s.mean_sub)
    mlim = (np.floor(min(allmeans) * 20) / 20, np.ceil(max(allmeans) * 20) / 20)
    sigmask = mx[mx.metric.eq('aperiodic_exponent')].set_index('channel').loc[chans].p_maxT_fwer.values < 0.05
    for s, name in [('core', 'F4A_core'), ('extension', 'F4B_ext')]:
        z = st[s]
        f = fig(180, 40)
        axs = [f.add_axes([x0, 0.04, 0.17, 0.82]) for x0 in (0.06, 0.25, 0.54)]
        cax1 = f.add_axes([0.445, 0.18, 0.008, 0.55]); cax2 = f.add_axes([0.735, 0.18, 0.008, 0.55])
        im = draw_topo(axs[0], z.mean_ctrl.values, info, mlim, 'viridis')
        draw_topo(axs[1], z.mean_sub.values, info, mlim, 'viridis')
        im2 = draw_topo(axs[2], z.t.values, info, (-4, 4), 'RdBu_r', mask=sigmask if s == 'core' else None)
        for ax, t in zip(axs, ['Control', 'Subclinical', 'Subclinical − control']):
            ax.set_title(t, fontsize=7.5, pad=2)
        cb = f.colorbar(im, cax=cax1); cb.set_label('Exponent', fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2, width=0.5); cb.outline.set_linewidth(0.5)
        cb2 = f.colorbar(im2, cax=cax2); cb2.set_label('Welch t', fontsize=7); cb2.ax.tick_params(labelsize=6.5, length=2, width=0.5); cb2.outline.set_linewidth(0.5)
        save(f, name)
        NUMBERS[f'aperiodic_t_{s}'] = [round(z.t.min(), 3), round(z.t.max(), 3)]
    # row C: oscillatory band t-maps, core
    f = fig(110, 42)
    axs = [f.add_axes([0.02 + i * 0.222, 0.04, 0.19, 0.72]) for i in range(4)]
    cax = f.add_axes([0.92, 0.16, 0.012, 0.5])
    for ax, b in zip(axs, ['delta', 'theta', 'alpha', 'beta']):
        z = channel_stats(e, 'core', f'ln_{b}').set_index('channel').loc[chans]
        p = mx[mx.metric.eq(f'{b}_abs')].set_index('channel').loc[chans]
        assert np.allclose(z.t.values, p.t_welch.values, atol=1e-8), b
        im = draw_topo(ax, z.t.values, info, (-4, 4), 'RdBu_r', mask=p.p_maxT_fwer.values < 0.05)
        rng_ = {'delta': '1–4 Hz', 'theta': '4–8 Hz', 'alpha': '8–13 Hz', 'beta': '13–30 Hz'}[b]
        ax.set_title(f'{b.capitalize()}\n{rng_}', fontsize=7, pad=2, linespacing=1.1)
        NUMBERS[f'{b}_maxT_sig'] = int((p.p_maxT_fwer < 0.05).sum())
    cb = f.colorbar(im, cax=cax); cb.set_label('Welch t', fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2, width=0.5); cb.outline.set_linewidth(0.5)
    save(f, 'F4C_bands')
    # D: channel-mean exponent by cohort and group
    cm = e.groupby(['pid', 'sample', 'group']).aperiodic_exponent.mean().reset_index()
    rng = np.random.default_rng(SEED)
    f = fig(68, 44); ax = f.add_axes([0.22, 0.2, 0.75, 0.62])
    pos = {('core', 'control'): 0, ('core', 'subclinical'): 1, ('extension', 'control'): 2.5, ('extension', 'subclinical'): 3.5}
    for (s, g), x in pos.items():
        v = cm[cm['sample'].eq(s) & cm.group.eq(g)].aperiodic_exponent.values
        c = CTRL if g == 'control' else SUB
        ax.scatter(x + jitter(len(v), 0.15, rng), v, s=2.5, color=c, alpha=0.45, lw=0)
        ax.boxplot([v], positions=[x], widths=0.5, showfliers=False, patch_artist=True,
                   medianprops=dict(color=INK, lw=0.9), boxprops=dict(facecolor='none', edgecolor=INK, lw=0.7),
                   whiskerprops=dict(color=INK, lw=0.6), capprops=dict(color=INK, lw=0.6))
    for s, x in [('core', 0.5), ('extension', 3.0)]:
        z = cm[cm['sample'].eq(s)]
        d = pooled_d(z[z.group.eq('subclinical')].aperiodic_exponent, z[z.group.eq('control')].aperiodic_exponent)
        ax.text(x, 1.02, f'{s.capitalize()}\nd = {d:.2f}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.8)
        NUMBERS[f'aper_d_{s}'] = round(d, 3)
        NUMBERS[f'eeg_n_{s}'] = z.groupby('group').pid.nunique().to_dict()
    ax.set_xticks([0, 1, 2.5, 3.5], ['Ctrl', 'Sub', 'Ctrl', 'Sub'])
    ax.set_ylabel('Aperiodic exponent\n(channel mean)')
    save(f, 'F4D_exponent')


# ---------------------------------------------------------------- Figure 5
def fig5_scatter():
    sf = read('data/09_source_data_figures/Figure4A_lab_vs_ambulatory.csv')
    sf['cohort'] = np.where(sf.pid.str.startswith('C'), 'core', 'extension')
    x, y = sf.lab_hr_reactivity_bpm.values, sf.ambulatory_response_bpm.values
    r = stats.pearsonr(x, y)
    f = fig(86, 68); ax = f.add_axes([0.16, 0.17, 0.8, 0.78])
    for c, col, mk in [('core', CORE, 'o'), ('extension', EXT, 's')]:
        z = sf[sf.cohort.eq(c)]
        ax.scatter(z.lab_hr_reactivity_bpm, z.ambulatory_response_bpm, s=5, marker=mk, color=col, alpha=0.55, lw=0, label=f'{c.capitalize()} (n = {len(z)})')
    X = np.column_stack([np.ones_like(x), x]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    xs = np.linspace(x.min(), x.max(), 100); Xs = np.column_stack([np.ones_like(xs), xs])
    res = y - X @ beta; s2 = res @ res / (len(y) - 2); cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.einsum('ij,jk,ik->i', Xs, cov, Xs)); tq = stats.t.ppf(0.975, len(y) - 2)
    ax.fill_between(xs, Xs @ beta - tq * se, Xs @ beta + tq * se, color='#C9D7E6', lw=0, zorder=0)
    ax.plot(xs, Xs @ beta, color=CTRL, lw=1.2)
    ax.axhline(0, color='#9E9E9E', lw=0.5, ls=(0, (3, 2)))
    ax.text(0.03, 0.97, f'r = {r.statistic:.2f}, p = {r.pvalue:.3f}\nn = {len(sf)}', transform=ax.transAxes, va='top', fontsize=7)
    ax.legend(loc='lower right', handletextpad=0.2, borderaxespad=0.2, markerscale=1.6)
    ax.set_xlabel('Laboratory HR reactivity (stress − control, bpm)')
    ax.set_ylabel('Ambulatory HR response to stressors (bpm)')
    save(f, 'F5A_scatter')
    NUMBERS['lab_field_r'] = [round(r.statistic, 4), float(r.pvalue)]


ABLATIONS = [('MSAF without calibration', 'Calibration removed'),
             ('MSAF without anchored normalization', 'Anchored normalization removed'),
             ('MSAF without FiLM', 'FiLM removed'),
             ('Wrist + EMA (no laboratory)', 'Laboratory visit removed'),
             ('MSAF without EEG', 'EEG removed'),
             ('MSAF without EMA', 'EMA removed')]


def fig5_ablation():
    comp = read('audit_20261028_run1_provided_prediction_comparisons.csv')
    f = fig(86, 68); ax = f.add_axes([0.45, 0.17, 0.4, 0.78])
    ys = np.arange(len(ABLATIONS))[::-1]
    for (b, lab), y in zip(ABLATIONS, ys):
        r = comp[comp.a.eq('MSAF (full)') & comp.b.eq(b)].iloc[0]
        c = SUB if b == 'MSAF without calibration' else CTRL
        ax.plot([r.bootstrap_lo, r.bootstrap_hi], [y, y], color=c, lw=0.9)
        ax.plot(r.delta, y, 'o', ms=3.6, color=c)
        ax.text(1.03, y, f'{r.delta:+.3f}'.replace('-', '−'), transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=6.8)
    ax.axvline(0, color='#7F7F7F', lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys, [a[1] for a in ABLATIONS]); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(-0.06, 0.16); ax.set_xticks([-0.05, 0, 0.05, 0.10])
    ax.set_xlabel('ΔAUC (full MSAF − ablation)')
    save(f, 'F5B_ablation')


# ---------------------------------------------------------------- Figure 6
MAIN3 = [('MSAF (full)', 'MSAF'), ('SDS + SAS', 'SDS/SAS'), ('SDS + SAS + MSAF blocks (incremental)', 'Combined')]


def fig6_roc_cal():
    pr = read('data/07_model/extension_predicted_probabilities.csv')
    met = read('audit_20261028_run1_provided_prediction_metrics.csv').set_index('model')
    f = fig(58, 52); ax = f.add_axes([0.21, 0.19, 0.74, 0.76])
    ax.plot([0, 1], [0, 1], color='#9E9E9E', lw=0.6, ls=(0, (3, 2)))
    for m, lab in MAIN3:
        fpr, tpr, _ = roc_curve(pr.y, pr[m])
        ax.plot(fpr, tpr, color=MODEL[lab], lw=1.0, label=f'{lab} {met.loc[m, "auc"]:.3f}', drawstyle='steps-post')
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc='lower right', title='AUC', title_fontsize=7, handlelength=1.4, borderaxespad=0.1)
    ax.set_aspect('equal', adjustable='box')
    save(f, 'F6A_roc')
    f = fig(58, 52); ax = f.add_axes([0.21, 0.19, 0.74, 0.76])
    ax.plot([0, 1], [0, 1], color='#9E9E9E', lw=0.6, ls=(0, (3, 2)))
    for m, lab in MAIN3:
        frac, mean = calibration_curve(pr.y, pr[m], n_bins=5, strategy='quantile')
        ax.plot(mean, frac, marker='o', ms=2.8, color=MODEL[lab], lw=0.9, label=f'{lab} slope {met.loc[m, "calibration_slope"]:.2f}')
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed proportion'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc='upper left', handlelength=1.2, borderaxespad=0.1, fontsize=6.3)
    ax.set_aspect('equal', adjustable='box')
    save(f, 'F6B_calibration')


def boot_auc(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; out = []
    for _ in range(n):
        idx = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        out.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(out, [2.5, 97.5])


def fig6_subgroups():
    pr = read('data/07_model/extension_predicted_probabilities.csv')
    ref = read('data/02_reference_standard/reference_standard.csv')
    if 'crisis_passive_si' not in ref.columns:  # public release: restricted crisis indicators
        return fig6_subgroups_from_summary()
    d = pr.merge(ref, on='pid'); cr = [c for c in ref.columns if c.startswith('crisis_')]
    d['nsig'] = d[cr].sum(1)
    ctrl = d[d.y.eq(0)]; sub = d[d.y.eq(1)]
    groups = [('Passive SI', sub[sub.crisis_passive_si.eq(1)]), ('No passive SI', sub[sub.crisis_passive_si.eq(0)]),
              ('Multiple signals', sub[sub.nsig.ge(2)]), ('Single signal', sub[sub.nsig.eq(1)])]
    f = fig(58, 52); ax = f.add_axes([0.42, 0.19, 0.54, 0.76])
    ys = [3.3, 2.3, 1.0, 0.0]; rows = []
    for (lab, s), y in zip(groups, ys):
        x = pd.concat([ctrl, s]); a = roc_auc_score(x.y, x['MSAF (full)']); lo, hi = boot_auc(x.y, x['MSAF (full)'])
        ax.plot([lo, hi], [y, y], color=CTRL, lw=0.9); ax.plot(a, y, 'o', ms=3.4, color=CTRL)
        rows.append(dict(group=lab, n_cases=len(s), auc=a, lo=lo, hi=hi))
    ax.axvline(0.5, color='#9E9E9E', lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys, [f'{g[0]} ({len(g[1])})' for g in groups]); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(0.45, 1.0); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xlabel('MSAF AUC vs. all controls')
    save(f, 'F6D_subgroups')
    NUMBERS['subgroup_auc'] = rows


def fig6_subgroups_from_summary():
    rows = json.loads((ROOT / 'Figures' / 'v4' / 'panel_numbers_v4.json').read_text(encoding='utf-8'))['subgroup_auc']
    f = fig(58, 52); ax = f.add_axes([0.42, 0.19, 0.54, 0.76])
    ys = [3.3, 2.3, 1.0, 0.0]
    for r, y in zip(rows, ys):
        ax.plot([r['lo'], r['hi']], [y, y], color=CTRL, lw=0.9); ax.plot(r['auc'], y, 'o', ms=3.4, color=CTRL)
    ax.axvline(0.5, color='#9E9E9E', lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys, [f"{r['group']} ({r['n_cases']})" for r in rows]); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(0.45, 1.0); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xlabel('MSAF AUC vs. all controls')
    save(f, 'F6D_subgroups')
    NUMBERS['subgroup_auc'] = rows


FOREST = [('arch', 'SDS + SAS + MSAF blocks (incremental)', 'SDS/SAS + MSAF blocks'),
          ('arch', 'SDS + SAS', 'SDS/SAS'),
          ('arch', 'MSAF (full)', 'MSAF (full)'),
          ('arch', 'Wrist physiology only', 'Wrist physiology only'),
          ('arch', 'Wrist + EMA (no laboratory)', 'Wrist + EMA'),
          ('arch', 'EMA only', 'EMA only'),
          ('arch', 'Laboratory autonomic only', 'Laboratory autonomic only'),
          ('arch', 'Laboratory EEG only', 'Laboratory EEG only'),
          ('new', 'SDS/SAS + physiology', 'SDS/SAS + physiology'),
          ('new', 'Physiology', 'Physiology')]


def fig6_forest():
    arch = read('audit_20261028_run1_provided_prediction_metrics.csv').set_index('model')
    new = read('audit_20261028_run1_retrained_metrics.csv').set_index('model')
    f = fig(180, 66); ax = f.add_axes([0.2, 0.15, 0.5, 0.76])
    y = 0; ys = []; labels = []; txt = []
    order = list(reversed(FOREST))
    for i, (src, m, lab) in enumerate(order):
        if i > 0 and src != order[i - 1][0]:
            y += 0.6
        t = arch if src == 'arch' else new
        r = t.loc[m]
        c = MODEL['Combined'] if 'MSAF blocks' in lab else (MODEL['SDS/SAS'] if lab == 'SDS/SAS' else (CTRL if src == 'arch' else '#4D4D4D'))
        mk = 'o' if src == 'arch' else 'D'
        ax.plot([r.auc_lo, r.auc_hi], [y, y], color=c, lw=0.9); ax.plot(r.auc, y, mk, ms=3.4 if mk == 'o' else 3.0, color=c)
        ys.append(y); labels.append(lab); txt.append((y, f'{r.auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]', r.sensitivity if 'sensitivity' in r else r.recall, r.specificity))
        y += 1
    sds = arch.loc['SDS + SAS', 'auc']
    ax.axvline(0.5, color='#9E9E9E', lw=0.6, ls=(0, (3, 2))); ax.axvline(sds, color=MODEL['SDS/SAS'], lw=0.6, ls=(0, (1, 1.5)))
    ax.set_yticks(ys, labels); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(0.45, 0.9); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); ax.set_xlabel('Extension AUC (95% bootstrap CI)')
    ax.set_ylim(-0.7, y - 0.3)
    tx = ax.get_yaxis_transform()
    for yy, s, se, sp in txt:
        ax.text(1.04, yy, s, transform=tx, va='center', fontsize=6.8)
        ax.text(1.36, yy, f'{se:.2f}', transform=tx, va='center', ha='center', fontsize=6.8)
        ax.text(1.5, yy, f'{sp:.2f}', transform=tx, va='center', ha='center', fontsize=6.8)
    for xx, h in [(1.04, 'AUC [95% CI]'), (1.36, 'Sens.'), (1.5, 'Spec.')]:
        ax.text(xx, y - 0.25, h, transform=tx, va='bottom', ha='left' if xx < 1.2 else 'center', fontsize=7, fontweight='bold')
    ax.text(-0.36, (ys[0] + ys[1]) / 2, 'Revision-trained\nelastic net', transform=tx, va='center', ha='center', fontsize=6.8, color='#4D4D4D', rotation=90)
    ax.text(-0.36, (ys[2] + ys[-1]) / 2, 'Archived predictions', transform=tx, va='center', ha='center', fontsize=6.8, color='#4D4D4D', rotation=90)
    save(f, 'F6C_forest')


# ---------------------------------------------------------------- Figure 7
SCEN = [('exam', 'Examination'), ('career_entrance_exam', 'Career/entrance exam'), ('no_stressor', 'No stressor'),
        ('academic_routine', 'Academic routine'), ('other', 'Other'), ('interpersonal', 'Interpersonal'),
        ('unclassified', 'Unclassified')]


def fig7_scenarios():
    t = read('revision_20261028_h4_table5.csv'); t = t[t.stratum.eq('overall')].set_index('scenario')
    f = fig(86, 58); ax = f.add_axes([0.42, 0.2, 0.55, 0.76])
    ys = np.arange(len(SCEN))[::-1].astype(float); ys[-1] -= 0.4
    for (k, lab), y in zip(SCEN, ys):
        r = t.loc[k]; c = '#9E9E9E' if k == 'unclassified' else CTRL
        ax.plot([r.ba_lo, r.ba_hi], [y, y], color=c, lw=0.9); ax.plot(r.balanced_accuracy, y, 'o', ms=3.4, color=c)
    allr = t.loc['all']
    ax.axvspan(allr.ba_lo, allr.ba_hi, color=SHADE, lw=0, zorder=0); ax.axvline(allr.balanced_accuracy, color='#4D4D4D', lw=0.7)
    ax.axvline(0.5, color='#9E9E9E', lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys, [f'{lab} ({int(t.loc[k].n_person_days):,})' for k, lab in SCEN]); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(0.33, 0.75); ax.set_xlabel('Balanced accuracy (95% CI)')
    save(f, 'F7A_scenarios')


def fig7_exam_gee():
    ex = read('revision_20261028_h4_exam_comparison.csv'); ex = ex[ex.scope.eq('all_rows')].set_index('contrast')
    f = fig(40, 58); ax = f.add_axes([0.36, 0.2, 0.6, 0.72])
    for x, k, c in [(0, 'non_exam_week', CTRL), (1, 'exam_week', SUB)]:
        r = ex.loc[k]
        ax.errorbar(x, r.balanced_accuracy, yerr=[[r.balanced_accuracy - r.ba_lo], [r.ba_hi - r.balanced_accuracy]], fmt='o', ms=3.6, color=c, capsize=2, lw=0.9)
        ax.text(x, r.ba_lo - 0.008, f'{int(r.n_person_days):,} d', ha='center', va='top', fontsize=6.5)
    ax.set_xticks([0, 1], ['Other\nweeks', 'Exam\nweeks']); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0.5, 0.72)
    ax.set_ylabel('Balanced accuracy')
    save(f, 'F7B_exam')
    ge = read('revision_20261028_h4_models.csv').set_index('model')
    f = fig(44, 58); ax = f.add_axes([0.24, 0.2, 0.72, 0.72])
    for x, k, lab in [(0, 'GEE_six_known_scenarios_adjusted_exchangeable', 'Exch.'), (1, 'GEE_six_known_scenarios_adjusted_independence', 'Indep.')]:
        r = ge.loc[k]
        ax.errorbar(x, r.OR, yerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt='o', ms=3.6, color=CTRL, capsize=2, lw=0.9)
        ax.text(x, r.OR_hi + 0.03, f'{r.OR:.2f}', ha='center', va='bottom', fontsize=6.8)
    ax.axhline(1, color='#9E9E9E', lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([0, 1], ['Exch.', 'Indep.']); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0.6, 1.5)
    ax.set_ylabel('Odds ratio, exam week', labelpad=1)
    save(f, 'F7C_gee')


def fig7_within():
    wi = read('revision_20261028_h4_figure6b.csv').set_index('method')
    f = fig(86, 60); ax = f.add_axes([0.16, 0.24, 0.8, 0.7])
    meth = [('raw', 'Raw'), ('lab_anchored', 'Laboratory\nanchored'), ('first_week_z', 'First-week z')]
    for j, (k, lab) in enumerate(meth):
        r = wi.loc[k]
        for off, v, lo, hi, c, mk in [(-0.12, r.pooled_auc, r.pooled_auc_lo, r.pooled_auc_hi, CORE, 'o'),
                                      (0.12, r.mean_within_person_auc, r.within_auc_lo, r.within_auc_hi, SUB, 's')]:
            ax.errorbar(j + off, v, yerr=[[v - lo], [hi - v]], fmt=mk, ms=3.4, color=c, capsize=2, lw=0.9)
    ax.axhline(0.5, color='#9E9E9E', lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(3), [m[1] for m in meth]); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.48, 0.6)
    ax.set_ylabel('AUC, higher-stress day')
    ax.plot([], [], 'o', color=CORE, label='Pooled'); ax.plot([], [], 's', color=SUB, label='Mean within-person')
    ax.legend(loc='upper left', ncol=2, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.1)
    save(f, 'F7D_within')
    d = read('audit_20261028_daily_refined_within_person_predictions.csv')
    rows = []
    for pid, g in d.groupby('pid'):
        if g.highstress.nunique() == 2:
            rows.append(dict(pid=pid, raw=roc_auc_score(g.highstress, g.raw), lab=roc_auc_score(g.highstress, g.lab_anchored), fwz=roc_auc_score(g.highstress, g.first_week_z)))
    a = pd.DataFrame(rows)
    NUMBERS['within_auc_means'] = dict(n=len(a), raw=a.raw.mean(), lab=a.lab.mean(), fwz=a.fwz.mean())
    f = fig(88, 60); ax = f.add_axes([0.16, 0.24, 0.8, 0.7])
    rng = np.random.default_rng(SEED)
    for j, (k, lab) in enumerate([('raw', 'Raw'), ('lab', 'Laboratory\nanchored'), ('fwz', 'First-week z')]):
        v = a[k].values
        vp = ax.violinplot([v], positions=[j], widths=0.75, showextrema=False)
        for b in vp['bodies']:
            b.set_facecolor('#DCE6F0'); b.set_edgecolor('none'); b.set_alpha(1)
        ax.scatter(j + jitter(len(v), 0.12, rng), v, s=2.5, color=CTRL, alpha=0.5, lw=0)
        ax.boxplot([v], positions=[j], widths=0.22, showfliers=False, medianprops=dict(color=INK, lw=0.9),
                   boxprops=dict(color=INK, lw=0.7), whiskerprops=dict(color=INK, lw=0.6), capprops=dict(color=INK, lw=0.6))
        ax.text(j, 1.02, f'M = {v.mean():.3f}', ha='center', va='bottom', fontsize=6.8)
    ax.axhline(0.5, color='#9E9E9E', lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(3), ['Raw', 'Laboratory\nanchored', 'First-week z']); ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel('Within-person AUC (per participant)')
    save(f, 'F7E_individual')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for fn in [fig2_vas, fig3_phases, fig3_forest, fig3_reactivity, fig4_topos, fig5_scatter, fig5_ablation,
               fig6_roc_cal, fig6_subgroups, fig6_forest, fig7_scenarios, fig7_exam_gee, fig7_within]:
        print('panel', fn.__name__, flush=True); fn()
    (ROOT / 'Figures' / 'v4' / 'panel_numbers_v4_rerun.json').write_text(json.dumps(NUMBERS, indent=2, default=float), encoding='utf-8')
    print(json.dumps(NUMBERS, indent=1, default=float))


if __name__ == '__main__':
    main()
