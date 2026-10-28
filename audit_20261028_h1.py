"""H1 reproducibility audit. Reads source CSVs only; creates new outputs exclusively.
Run: python audit_20261028_h1.py --prefix audit_20261028_h1_run1
Dependencies: numpy, pandas, scipy, statsmodels. Python 3.11.
EEG max-T is a sensitivity analysis, NOT a reconstruction of undocumented clusters.
"""
from pathlib import Path
import argparse, json, sys, warnings, hashlib, platform, time
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parent
def read(p): return pd.read_csv(ROOT/'data'/p)
def dump(path,obj):
    with open(path,'x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=lambda x:float(x))
def effect(a,b):
    a=np.asarray(pd.Series(a).dropna(),float);b=np.asarray(pd.Series(b).dropna(),float)
    n1,n0=len(a),len(b);df=n1+n0-2
    pooled=np.sqrt(((n1-1)*a.var(ddof=1)+(n0-1)*b.var(ddof=1))/df)
    d=(a.mean()-b.mean())/pooled
    se=np.sqrt((n1+n0)/(n1*n0)+d*d/(2*df))
    welch=stats.ttest_ind(a,b,equal_var=False);student=stats.ttest_ind(a,b,equal_var=True)
    return dict(n_control=n0,n_subclinical=n1,control_m=b.mean(),control_sd=b.std(ddof=1),sub_m=a.mean(),sub_sd=a.std(ddof=1),d=d,d_lo=d-1.96*se,d_hi=d+1.96*se,t_welch=welch.statistic,p_welch=welch.pvalue,t_student=student.statistic,p_student=student.pvalue)
def icc31(a):
    a=np.asarray(a,float);n,k=a.shape;gm=a.mean()
    msr=k*np.sum((a.mean(axis=1)-gm)**2)/(n-1)
    mse=np.sum((a-a.mean(axis=1,keepdims=True)-a.mean(axis=0,keepdims=True)+gm)**2)/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prefix',default='audit_20261028_h1_run1');ap.add_argument('--permutations',type=int,default=5000);args=ap.parse_args()
    prefix=ROOT/args.prefix
    suffixes=['_markers.csv','_reconstructed_features.csv','_results.json','_review.txt','_log.txt','_eeg_maxT.csv']
    for s in suffixes:
        if Path(str(prefix)+s).exists(): raise FileExistsError(f'Refusing overwrite: {prefix}{s}')
    log=open(str(prefix)+'_log.txt','x',encoding='utf-8')
    def progress(s):print(s,flush=True);log.write(s+'\n');log.flush()
    progress('H1 source-feature reconstruction started')
    part=read('01_participants/participant_characteristics.csv').set_index('pid')
    ready=read('06_analysis_ready/person_level_features.csv').set_index('pid')
    lab=read('04_laboratory/lab_phase_features_ecg_eda.csv')
    eeg=read('04_laboratory/lab_eeg_features.csv')
    quality=read('04_laboratory/lab_eeg_quality.csv').set_index(['pid','session'])
    post=['P7','P3','Pz','P4','P8','PO3','PO4','O1','Oz','O2']
    front=['Fp1','Fp2','AF3','AF4','F7','F3','Fz','F4','F8']
    def reconstruct(session):
        a=lab[lab.session.eq(session)];p=a.pivot(index='pid',columns='phase')
        out=pd.DataFrame(index=p.index)
        for name,col in [('hr','hr_bpm'),('lnrmssd','ln_rmssd'),('lnhf','ln_hf_ms2'),('scl','scl_us')]:
            out[f'lab_{name}_base']=p[col]['baseline']
            st=p[col][['stress1','stress2','stress3']].mean(axis=1)
            out[f'lab_{name}_react']=st-p[col]['control']
            out[f'lab_{name}_control']=p[col]['control']
            if name=='hr':out['lab_hr_recovery']=((st-p[col]['recovery2'])/(st-p[col]['baseline'])).where(st-p[col]['baseline']>=1)
        out['lab_nsscr_stress']=p['ns_scr_per_min'][['stress1','stress2','stress3']].mean(axis=1)
        e=eeg[eeg.session.eq(session)]
        ec=e[e.condition.eq('rest_ec')].copy()
        out['eeg_aper_mean']=ec.groupby('pid').aperiodic_exponent.mean()
        out['eeg_alpha_post']=ec[ec.channel.isin(post)].assign(val=lambda d:np.log(d.alpha_abs)).groupby('pid').val.mean()
        out['eeg_iaf']=ec.groupby('pid').iaf_hz.mean()
        alph=ec.pivot(index='pid',columns='channel',values='alpha_abs')
        out['eeg_faa']=np.log(alph.F4)-np.log(alph.F3)
        th=e[e.channel.isin(front)].assign(val=lambda d:np.log(d.theta_abs)).groupby(['pid','condition']).val.mean().unstack()
        out['eeg_theta_change']=th.stress-th.control
        q=quality.xs(session,level='session')
        bad=q.index[(q.excluded_gt20pct_bad.eq(1)) | (q.clean_s_rest_ec<120)]
        out.loc[out.index.intersection(bad),[c for c in out if c.startswith('eeg_')]]=np.nan
        return out
    d0=reconstruct('D0');d31=reconstruct('D31')
    d=d0.join(part[['sample','group','age','sex','bmi','analyzable','severity']]);d['y']=d.group.eq('subclinical').astype(int)
    d=d[d.analyzable.eq(1)]
    d.to_csv(str(prefix)+'_reconstructed_features.csv',encoding='utf-8-sig',mode='x')
    comparisons={}
    for c in ready.columns.intersection(d.columns):
        if c.startswith(('lab_','eeg_')):
            cmp=pd.concat([ready[c],d[c]],axis=1).dropna()
            comparisons[c]={'n':len(cmp),'max_abs_diff':float(np.max(np.abs(cmp.iloc[:,0]-cmp.iloc[:,1]))),'note':'raw SCL reconstructed; analysis-ready stores ln(SCL)' if c=='lab_scl_base' else ''}
    names=[('HR baseline','lab_hr_base'),('HR reactivity (stress-control)','lab_hr_react'),('HR recovery index','lab_hr_recovery'),('ln RMSSD baseline','lab_lnrmssd_base'),('ln RMSSD reactivity','lab_lnrmssd_react'),('ln HF baseline','lab_lnhf_base'),('SCL baseline','lab_scl_base'),('SCL reactivity','lab_scl_react'),('NS-SCR stress','lab_nsscr_stress'),('Aperiodic exponent (EC, mean)','eeg_aper_mean'),('Posterior alpha ln power (EC)','eeg_alpha_post'),('IAF (EC)','eeg_iaf'),('Frontal alpha asymmetry','eeg_faa'),('Frontal theta change (stress-control)','eeg_theta_change')]
    rows=[]
    rep=['lab_lnrmssd_base','lab_lnhf_base','lab_hr_react','lab_lnrmssd_react','eeg_aper_mean']
    for sample in ['core','extension']:
        ds=d[d['sample'].eq(sample)]
        for name,c in names:
            rr=effect(ds.loc[ds.y.eq(1),c],ds.loc[ds.y.eq(0),c]);rr.update(sample=sample,measure=name,feature=c);rows.append(rr)
    tab=pd.DataFrame(rows)
    for sample in ['core','extension']:
        idx=tab['sample'].eq(sample);tab.loc[idx,'p_fdr14']=multipletests(tab.loc[idx,'p_welch'],method='fdr_bh')[1]
    idx=tab['sample'].eq('extension')&tab.feature.isin(rep)
    tab.loc[idx,'p_fdr5_replication']=multipletests(tab.loc[idx,'p_welch'],method='fdr_bh')[1]
    old=read('08_source_data_tables/Table3_lab_markers.csv').set_index('measure')
    for i,row in tab[tab['sample'].eq('core')].iterrows():
        for c,oc in [('d','d'),('p_welch','p'),('p_fdr14','p_fdr')]:tab.loc[i,'difference_from_Table3_'+c]=row[c]-old.loc[row.measure,oc]
    tab.to_csv(str(prefix)+'_markers.csv',index=False,encoding='utf-8-sig',mode='x')
    progress('14 markers and five-marker extension replication complete')
    adjusted=[]
    for sample in ['core','extension']:
        ds=d[d['sample'].eq(sample)]
        for name in ['hr','lnrmssd']:
            m=smf.ols(f'lab_{name}_react ~ y + lab_{name}_control',data=ds).fit()
            ci=m.conf_int().loc['y'];sd=np.sqrt(m.mse_resid)
            adjusted.append(dict(sample=sample,feature=name,n=int(m.nobs),b=m.params.y,se=m.bse.y,p=m.pvalues.y,adjusted_d=m.params.y/sd,d_lo=ci.iloc[0]/sd,d_hi=ci.iloc[1]/sd,definition='group coefficient divided by ANCOVA residual SD; adjust only control-task value'))
    comp=-(stats.zscore(d.lab_lnrmssd_base,ddof=1)+stats.zscore(d.lab_hr_react,ddof=1))/2
    composite=effect(comp[d.y.eq(1)],comp[d.y.eq(0)]);composite['p_one_sided']=composite['p_welch']/2 if composite['t_welch']>0 else 1-composite['p_welch']/2
    retest=[]
    rng=np.random.default_rng(20261028)
    for scope,ids in [('core_analyzable',d.index[d['sample'].eq('core')]),('all_core_with_retest',part.index[part['sample'].eq('core')])]:
        for col in ['lab_lnrmssd_base','lab_hr_base','lab_hr_react','lab_scl_base','eeg_alpha_post','eeg_aper_mean']:
            pair=pd.concat([d0[col],d31[col]],axis=1,keys=['D0','D31']).reindex(ids).dropna()
            ar=pair.to_numpy();boot=[icc31(ar[rng.integers(0,len(ar),len(ar))]) for _ in range(2000)]
            retest.append(dict(scope=scope,feature=col,n=len(pair),icc31=icc31(ar),ci_lo=np.quantile(boot,.025),ci_hi=np.quantile(boot,.975)))
    reev=read('02_reference_standard/reevaluation_day31.csv').set_index('pid')
    delta=pd.concat([d31.lab_lnrmssd_base-d0.lab_lnrmssd_base,reev.severity-part.severity],axis=1,keys=['rmssd_change','severity_change']).reindex(d.index[d['sample'].eq('core')]).dropna()
    corr=stats.pearsonr(delta.rmssd_change,delta.severity_change)
    subjective=read('04_laboratory/lab_subjective_ratings.csv');su=subjective[subjective.session.eq('D0')].pivot(index='pid',columns='timepoint',values='vas_stress').reindex(d.index[d['sample'].eq('core')]).dropna()
    tc=stats.ttest_rel(su.T2,su.T1)
    manipulation=dict(n=len(su),means=su.mean().to_dict(),t_T2minusT1=tc.statistic,p_T2minusT1=tc.pvalue,mean_hr_stress_minus_control=d.loc[d['sample'].eq('core'),'lab_hr_react'].mean())
    results=dict(scope='Source CSV feature-level H1 rerun; no raw signal reprocessing and no verification of preregistration',software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__),feature_reconstruction=comparisons,roi_definitions=dict(posterior=post,frontal=front,note='ROI channel lists inferred by exact numeric reconstruction; original analysis script missing'),adjusted_reactivity=adjusted,pooled_composite=composite,test_retest=retest,severity_change=dict(n=len(delta),pearson_r=corr.statistic,p=corr.pvalue),manipulation_check=manipulation)
    progress('Composite, ANCOVA and retest complete; beginning phase mixed models')
    mixed=[]
    core=lab[lab.session.eq('D0')].merge(d[d['sample'].eq('core')][['y','age','sex','bmi']],left_on='pid',right_index=True)
    for outcome in ['hr_bpm','ln_rmssd','scl_us']:
        progress('Fitting ML random-intercept phase model: '+outcome)
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter('always')
            f=smf.mixedlm(f'{outcome} ~ y*C(phase)+age+C(sex)+bmi',core,groups=core.pid).fit(reml=False,method='lbfgs',maxiter=1000)
            r=smf.mixedlm(f'{outcome} ~ y+C(phase)+age+C(sex)+bmi',core,groups=core.pid).fit(reml=False,method='lbfgs',maxiter=1000)
        lr=2*(f.llf-r.llf);df=len(f.fe_params)-len(r.fe_params)
        mixed.append(dict(outcome=outcome,n_rows=int(f.nobs),n_subjects=core.pid.nunique(),lr_chi2=lr,df=df,p=stats.chi2.sf(lr,df),full_converged=bool(f.converged),reduced_converged=bool(r.converged),llf_full=f.llf,llf_reduced=r.llf,warnings=list(dict.fromkeys(str(w.message) for w in warns)),random_effect='participant intercept',estimation='ML, no random slopes'))
        progress(f'{outcome}: LR={lr:.6f}, df={df}, p={stats.chi2.sf(lr,df):.6g}')
    results['phase_mixed_models']=mixed
    progress('EEG 5000-permutation max-|t| sensitivity analysis begins')
    ec=eeg[eeg.session.eq('D0') & eeg.condition.eq('rest_ec')]
    ids=d.index[d['sample'].eq('core') & d.eeg_aper_mean.notna()]
    labels=d.loc[ids,'y'].to_numpy();n1=int(labels.sum());n0=len(labels)-n1
    metrics=['aperiodic_exponent','delta_abs','theta_abs','alpha_abs','beta_abs']
    perm_rows=[];perm_summary=[]
    for metric in metrics:
        x=ec.pivot(index='pid',columns='channel',values=metric).loc[ids]
        if metric.endswith('_abs'):x=np.log(x)
        ar=x.to_numpy();obs=stats.ttest_ind(ar[labels==1],ar[labels==0],axis=0,equal_var=False).statistic
        null=np.empty(args.permutations)
        for j in range(args.permutations):
            ix=rng.permutation(len(ar));a=ar[ix[:n1]];b=ar[ix[n1:]]
            tt=(a.mean(0)-b.mean(0))/np.sqrt(a.var(0,ddof=1)/n1+b.var(0,ddof=1)/n0)
            null[j]=np.abs(tt).max()
        corrected=(1+(null[:,None]>=np.abs(obs)[None,:]).sum(0))/(args.permutations+1)
        perm_summary.append(dict(metric=metric,n_subclinical=n1,n_control=n0,n_permutations=args.permutations,n_significant_channels=int((corrected<.05).sum()),min_maxT_p=corrected.min(),observed_max_abs_t=np.abs(obs).max()))
        for c,t,pv in zip(x.columns,obs,corrected):perm_rows.append(dict(metric=metric,channel=c,t_welch=t,p_maxT_fwer=pv))
        progress(f'{metric}: max-T min p={corrected.min():.6f}, significant channels={(corrected<.05).sum()}')
    pd.DataFrame(perm_rows).to_csv(str(prefix)+'_eeg_maxT.csv',index=False,encoding='utf-8-sig',mode='x')
    results['eeg_permutation_sensitivity']=perm_summary
    results['cluster_reproduction_status']='UNVERIFIED: original adjacency, cluster-forming threshold, cluster statistic, sidedness handling, seed and analysis scripts absent; MNE unavailable. max-T is not cluster p replication.'
    results['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').rglob('*.csv')}
    dump(str(prefix)+'_results.json',results)
    lines=['H1 全局特征级复算报告','Verification Status: ANALYZED / PARTIALLY REPRODUCED','Material Passport: source=local CSV; raw signal processing unavailable; numerical audit only; no clinical inference.','', '主表14指标使用Welch双侧t检验；同时输出Student结果。Cohen d采用合并标准差，CI为正态近似。BH分别对core14项和extension预选5项实施。','重建SCL baseline使用原始μS；analysis_ready中的同名字段为ln(SCL)，不可直接混用。',tab.to_string(index=False),'','控制任务基线调整：',pd.DataFrame(adjusted).to_string(index=False),'','组×阶段随机截距混合模型：',pd.DataFrame(mixed).to_string(index=False),'','ICC(3,1)：',pd.DataFrame(retest).to_string(index=False),'',f'合并自主神经复合指标: {composite}',f'严重程度变化与lnRMSSD变化: n={len(delta)}, r={corr.statistic}, p={corr.pvalue}','',results['cluster_reproduction_status'],pd.DataFrame(perm_summary).to_string(index=False),'','不能从这些输出确认原始采集真实性、预注册时间或原始信号预处理。ROI由低层特征与已给分析表精确对应推定，原始ROI脚本仍需补充。']
    with open(str(prefix)+'_review.txt','x',encoding='utf-8') as h:h.write('\n'.join(lines))
    progress('Completed all available H1 analyses; output saved.');log.close()
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
