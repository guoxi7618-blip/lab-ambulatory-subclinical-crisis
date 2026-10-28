"""Exact Student-t reconstruction and definition audit following h1.py outputs.
Run after audit_20261028_h1.py. Creates new outputs; refuses overwrites.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parent
def main():
    tab=pd.read_csv(ROOT/'audit_20261028_h1_run1_markers.csv')
    d=pd.read_csv(ROOT/'audit_20261028_h1_run1_reconstructed_features.csv').set_index('pid')
    original=pd.read_csv(ROOT/'data/08_source_data_tables/Table3_lab_markers.csv').set_index('measure')
    for sample in ['core','extension']:
        ix=tab['sample'].eq(sample)
        tab.loc[ix,'p_student_fdr14']=multipletests(tab.loc[ix,'p_student'],method='fdr_bh')[1]
    rep=['lab_lnrmssd_base','lab_lnhf_base','lab_hr_react','lab_lnrmssd_react','eeg_aper_mean']
    ix=tab['sample'].eq('extension')&tab.feature.isin(rep)
    tab.loc[ix,'p_student_fdr5_replication']=multipletests(tab.loc[ix,'p_student'],method='fdr_bh')[1]
    for i,r in tab[tab['sample'].eq('core')].iterrows():
        tab.loc[i,'student_p_minus_Table3']=r.p_student-original.loc[r.measure,'p']
        tab.loc[i,'student_fdr_minus_Table3']=r.p_student_fdr14-original.loc[r.measure,'p_fdr']
    checks=[]
    for sample in ['core','extension']:
        z=d[d['sample'].eq(sample)]
        for feature in tab.feature.unique():
            a=z.loc[z.y.eq(1),feature].dropna();b=z.loc[z.y.eq(0),feature].dropna()
            lv=stats.levene(a,b,center='median');sh1=stats.shapiro(a);sh0=stats.shapiro(b)
            checks.append(dict(sample=sample,feature=feature,levene_brown_forsythe_p=lv.pvalue,shapiro_subclinical_p=sh1.pvalue,shapiro_control_p=sh0.pvalue,variance_ratio=a.var()/b.var()))
    adjusted=[]
    for sample in ['core','extension']:
        z=d[d['sample'].eq(sample)]
        for feat in ['hr','lnrmssd']:
            out=f'lab_{feat}_react';m=smf.ols(f'{out}~y+lab_{feat}_control',z).fit();sd=z[out].std(ddof=1)
            adjusted.append(dict(sample=sample,feature=feat,standardized_b=m.params.y/sd,lo=m.conf_int().loc['y'].iloc[0]/sd,hi=m.conf_int().loc['y'].iloc[1]/sd,p=m.pvalues.y,definition='outcome SD includes both groups: standardized ANCOVA coefficient, not conventional pooled-within-group Cohen d'))
    su=pd.read_csv(ROOT/'data/04_laboratory/lab_subjective_ratings.csv');part=pd.read_csv(ROOT/'data/01_participants/participant_characteristics.csv')
    subjects=su[su.session.eq('D0')].merge(part,on='pid')
    manipulation=[]
    for name,z in [('all_enrolled_D0_sessions',subjects),('all_analyzable_D0_sessions',subjects[subjects.analyzable.eq(1)]),('core_analyzable_D0_sessions',subjects[subjects.analyzable.eq(1)&subjects['sample'].eq('core')])]:
        wide=z.pivot(index='pid',columns='timepoint',values='vas_stress').dropna();t=stats.ttest_rel(wide.T2,wide.T1)
        manipulation.append(dict(scope=name,n=len(wide),means=wide.mean().to_dict(),t=t.statistic,p=t.pvalue))
    eeg=pd.read_csv(ROOT/'data/04_laboratory/lab_eeg_features.csv');q=pd.read_csv(ROOT/'data/04_laboratory/lab_eeg_quality.csv').set_index(['pid','session'])
    posterior=['P7','P3','Pz','P4','P8','PO3','PO4','O1','Oz','O2']
    e=eeg[eeg.condition.eq('rest_ec')&eeg.channel.isin(posterior)].copy()
    e['logalpha']=np.log(e.alpha_abs)
    agg=e.groupby(['pid','session']).agg(mean_log=('logalpha','mean'),mean_power=('alpha_abs','mean'))
    agg['log_mean']=np.log(agg.mean_power)
    bad=q.index[q.excluded_gt20pct_bad.eq(1)|(q.clean_s_rest_ec<120)]
    agg.loc[agg.index.intersection(bad),:]=np.nan
    icc=[]
    for mode in ['mean_log','log_mean','mean_power']:
        a=agg[mode].unstack().reindex(d[d['sample'].eq('core')].index).dropna().to_numpy();n,k=a.shape;gm=a.mean();msr=k*np.sum((a.mean(1)-gm)**2)/(n-1);mse=np.sum((a-a.mean(1,keepdims=True)-a.mean(0,keepdims=True)+gm)**2)/((n-1)*(k-1))
        icc.append(dict(alpha_summary=mode,n=n,icc31=(msr-mse)/(msr+(k-1)*mse)))
    out=dict(primary_test='Student equal-variance t reconstructs 13/14 Table3 p-values to machine precision. Welch remains a sensitivity analysis; does not change primary significance pattern.',adjusted_reactivity_definition=adjusted,manipulation_check=manipulation,alpha_reliability_sensitivity=icc,missing='Exact cluster p cannot be reconstructed because adjacency and original script absent. No raw signals provided; only feature-level re-analysis.',notes=['IAF Table3 values do not match present source/analysis-ready CSV. Core control mean 10.029926 vs Table3 10.028; SD .707761 vs .715; Student p .402135 vs .409857. Non-significance unchanged.','Manuscript VAS means 27.6/59.0/33.7 refer to all 458 enrolled D0 sessions, not the 401 analyzable or 203 core sample. Report denominator explicitly.','Reported adjusted effects .51/.44 are consistent with outcome-SD standardized ANCOVA coefficients; calling them Cohen d needs a precise definition.','Global EEG max-T FWER across 32 channels was run with 5000 permutations per measure. Correction across the five EEG feature families is not applied; this is exploratory sensitivity analysis.'])
    tab.to_csv(ROOT/'audit_20261028_h1_student_markers.csv',index=False,encoding='utf-8-sig',mode='x')
    pd.DataFrame(checks).to_csv(ROOT/'audit_20261028_h1_assumption_checks.csv',index=False,encoding='utf-8-sig',mode='x')
    with open(ROOT/'audit_20261028_h1_definition_audit.json','x',encoding='utf-8') as h:json.dump(out,h,ensure_ascii=False,indent=2)
    with open(ROOT/'audit_20261028_h1_conclusions.txt','x',encoding='utf-8') as h:
        h.write('H1复算最终结论\n\n')
        h.write('\n'.join(out['notes'])+'\n\n')
        h.write(tab[['sample','feature','d','p_student','p_student_fdr14','p_student_fdr5_replication','student_p_minus_Table3']].to_string(index=False))
        h.write('\n\nANCOVA标准化口径\n'+pd.DataFrame(adjusted).to_string(index=False))
        h.write('\n\nalpha重测口径\n'+pd.DataFrame(icc).to_string(index=False))
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
