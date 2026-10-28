"""Supplementary numerical audit. Run with python; creates new files exclusively.
Read-only input: data/. Dependencies: numpy pandas scipy sklearn statsmodels.
No raw signal, screening-level risk log, or contact/referral log is inferred.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.power import TTestIndPower
ROOT=Path(__file__).resolve().parent
def read(s):return pd.read_csv(ROOT/'data'/s)
def effect(a,b):
    a=np.asarray(pd.Series(a).dropna());b=np.asarray(pd.Series(b).dropna());n1,n0=len(a),len(b)
    sd=np.sqrt(((n1-1)*a.var(ddof=1)+(n0-1)*b.var(ddof=1))/(n1+n0-2));d=(a.mean()-b.mean())/sd
    se=np.sqrt((n1+n0)/(n1*n0)+d*d/(2*(n1+n0-2)));t=stats.ttest_ind(a,b,equal_var=True)
    return dict(n_subclinical=n1,n_control=n0,d=d,d_lo=d-1.96*se,d_hi=d+1.96*se,d_90lo=d-stats.norm.ppf(.95)*se,d_90hi=d+stats.norm.ppf(.95)*se,p_two_sided=t.pvalue,p_one_sided=t.pvalue/2 if t.statistic>0 else 1-t.pvalue/2,sub_mean=a.mean(),control_mean=b.mean())
def icc21(a):
    a=np.asarray(a,float);n,k=a.shape;gm=a.mean()
    msr=k*np.sum((a.mean(1)-gm)**2)/(n-1)
    msc=n*np.sum((a.mean(0)-gm)**2)/(k-1)
    mse=np.sum((a-a.mean(1,keepdims=True)-a.mean(0,keepdims=True)+gm)**2)/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
def main():
    paths=[ROOT/f'audit_20261028_supplementary_audit{s}' for s in ['.json','.txt','_simpson.csv']]
    for p in paths:
        if p.exists():raise FileExistsError(p)
    dr=read('02_reference_standard/double_rating.csv');part=read('01_participants/participant_characteristics.csv').set_index('pid');ref=read('02_reference_standard/reference_standard.csv').set_index('pid')
    pf=read('06_analysis_ready/person_level_features.csv').set_index('pid').join(part[['sex','severity']])
    rng=np.random.default_rng(20261028)
    ka=cohen_kappa_score(dr.category_rater1,dr.category_rater2);a=dr[['severity_rater1','severity_rater2']].to_numpy();ic=icc21(a)
    kb=[];ib=[]
    for _ in range(2000):
        idx=rng.integers(0,len(a),len(a));ib.append(icc21(a[idx]));s=dr.iloc[idx];kb.append(cohen_kappa_score(s.category_rater1,s.category_rater2))
    agreement=dict(n=len(dr),kappa=ka,kappa_bootstrap_95ci=np.quantile(kb,[.025,.975]).tolist(),severity_icc21=ic,icc_bootstrap_95ci=np.quantile(ib,[.025,.975]).tolist(),observed_agreement=float(dr.category_rater1.eq(dr.category_rater2).mean()),by_sample=dr.groupby('sample').size().to_dict(),limitations='ICC(2,1) computed on rater-slot columns. Four actual counselors rotate across rows, rather than the same two counselors rating every participant; a fully crossed two-rater model is an approximation to this sparse assignment. Original ICC specification needed.',rater_pair_counts=dr.groupby(['rater1','rater2']).size().rename('n').reset_index().to_dict('records'))
    cols=['crisis_acute_stress','crisis_functional_impairment','crisis_anhedonia_withdrawal','crisis_passive_si','crisis_self_harm']
    signal=[];severity=[]
    for scope,ids in [('all_enrolled',part.index),('analyzable',pf.index)]:
        r=ref.loc[ids];s=r[r.category.eq('subclinical_psychological_crisis')]
        for col in cols:signal.append(dict(scope=scope,signal=col,n_positive=int(s[col].sum()),denominator=len(s),pct=100*s[col].mean()))
        signal.append(dict(scope=scope,signal='exactly_one_signal',n_positive=int(s[cols].sum(axis=1).eq(1).sum()),denominator=len(s),pct=100*s[cols].sum(axis=1).eq(1).mean()))
        for cat,z in r.groupby('category'):severity.append(dict(scope=scope,category=cat,n=len(z),mean=z.severity.mean(),sd=z.severity.std(),min=z.severity.min(),max=z.severity.max()))
    rows=[]
    marker=['lab_hr_base','lab_lnrmssd_base','lab_hr_react','lab_lnrmssd_react','eeg_aper_mean','ema_compliance','sds_standard','sas_standard']
    for col in marker:
        for scope,z in [('pooled',pf),('core',pf[pf['sample'].eq('core')]),('extension',pf[pf['sample'].eq('extension')])]:
            e=effect(z.loc[z.y.eq(1),col],z.loc[z.y.eq(0),col]);rows.append(dict(feature=col,stratum=scope,**e))
        for sex,z in pf.groupby('sex'):
            e=effect(z.loc[z.y.eq(1),col],z.loc[z.y.eq(0),col]);rows.append(dict(feature=col,stratum='sex_'+sex,**e))
        for (sample,sex),z in pf.groupby(['sample','sex']):
            e=effect(z.loc[z.y.eq(1),col],z.loc[z.y.eq(0),col]);rows.append(dict(feature=col,stratum=sample+'_'+sex,**e))
    simpson=[]
    st=pd.DataFrame(rows)
    for col in marker:
        z=st[st.feature.eq(col)].set_index('stratum');overall=np.sign(z.loc['pooled','d'])
        for grouping,strata in [('sample',['core','extension']),('sex',['sex_female','sex_male'])]:
            signs=np.sign(z.loc[strata,'d']);simpson.append(dict(feature=col,grouping=grouping,pooled_d=z.loc['pooled','d'],subgroup_d=z.loc[strata,'d'].to_dict(),classic_complete_reversal=bool((signs==-overall).all()),any_subgroup_opposite=bool((signs==-overall).any()),caution='An opposite nonsignificant subgroup estimate is heterogeneity, not proof of Simpson paradox.'))
    power=TTestIndPower().power(effect_size=.46,nobs1=100,alpha=.05,ratio=1,alternative='two-sided')
    def corpower(n,r):
        mu=np.arctanh(r)*np.sqrt(n-3);z=stats.norm.ppf(.975)
        return stats.norm.cdf(-z-mu)+stats.norm.sf(z-mu)
    power_result=dict(h1=dict(n_per_group=100,effect_d_assumed=.46,two_sided_alpha=.05,power=power,n_per_group_for_90pct=TTestIndPower().solve_power(effect_size=.46,power=.9,alpha=.05,ratio=1),note='Treats meta-analytic Hedges g=.46 as planning d=.46. No correction for multiple comparisons, attrition, or smaller subclinical effects.'),h2=dict(n_core=203,r=.20,power_fisher_z_approx=corpower(203,.20),n_200_power=corpower(200,.20),n_required_for_80pct_fisher_z=3+((stats.norm.ppf(.975)+stats.norm.ppf(.8))/np.arctanh(.2))**2,note='Fisher-z approximation for independent participant-level Pearson correlation. Does not establish power of the clustered cross-level interaction model.'))
    dev=(stats.zscore(pf.cal_resthr_minus_lab,ddof=1)-stats.zscore(pf.cal_noct_minus_lab,ddof=1))/2
    ef=effect(dev[pf.y.eq(1)],dev[pf.y.eq(0)]);corr=stats.pearsonr(dev,pf.severity)
    devreport=dict(formula='0.5 * (z(cal_resthr_minus_lab) - z(cal_noct_minus_lab)); pooled n=401 sample-standardization; reconstructed equal weights, exact preregistered weights unavailable',**ef,severity_pearson_r=corr.statistic,severity_pearson_p=corr.pvalue,group_correlations=[])
    for y,z in pf.groupby('y'):
        c=stats.pearsonr(dev.loc[z.index],z.severity);devreport['group_correlations'].append(dict(y=int(y),n=len(z),r=c.statistic,p=c.pvalue))
    ema=read('05_ambulatory/ema_prompts.csv');flow=read('01_participants/participant_flow.csv');reev=read('02_reference_standard/reevaluation_day31.csv')
    diary=ema[ema.prompt_type.eq('diary')]
    pos=ema[ema.passive_si.eq(1)]
    risk=[dict(claim='519 total protocol triggers',reported=519,status='PARTIALLY CHECKABLE / not independently verified',observed='288 + 81 + 150 = 519 arithmetically; the 288 screening triggers lack participant-level data and all triggers lack event logs',missing='screening and risk-event audit logs'),dict(claim='288 screening triggers',reported=288,status='UNVERIFIABLE',observed='Only PHQ-9 total for 460 enrolled participants, no PHQ-9 item 9 or full 2330 screening records',missing='screening item-level records and trigger definition'),dict(claim='81 evaluation triggers',reported=81,status='RECOMPUTED_PROXY',observed=int(ref.cssrs_screen_positive.eq(1).sum()),missing='risk-event logs confirming actual protocol activation; records cover 460 enrolled, not all 1370 evaluations'),dict(claim='150 diary-item triggers',reported=150,status='RECOMPUTED_PROXY',observed=int(len(pos)),positive_unique_participants=int(pos.pid.nunique()),prompt_type_counts=pos.prompt_type.value_counts().to_dict(),missing='event logs to distinguish all positive responses from unique episodes/contact events'),dict(claim='10.3 h median contact delay',reported=10.3,status='UNVERIFIABLE',observed=None,missing='trigger timestamps and first-contact timestamps'),dict(claim='45 same-day referrals',reported=45,status='UNVERIFIABLE',observed=None,missing='risk referral dates and trigger/contact links')]
    ancillary=dict(probable_disorder_referrals_from_flow=int(flow.loc[flow.stage.eq('probable_disorder_referred'),'n'].sum()),day31_referred_n=int(reev.referred.sum()),warning='The 80 probable-disorder screening referrals and Day31 referrals are not evidence for 45 same-day risk-protocol referrals.',positive_diary_rows=len(pos),unique_positive_diary_participants=pos.pid.nunique(),prompt_types=ema.prompt_type.value_counts().to_dict())
    out=dict(material_passport=dict(mode='statistical_validation',verification_status='ANALYZED',inputs='local data CSV only',provenance='Recalculated values; missing operational records explicitly marked'),reference_agreement=agreement,crisis_signal_counts=signal,severity_summary=severity,simpson_scan=simpson,planning_power=power_result,baseline_deviation_reconstruction=devreport,risk_protocol_claims=risk,ancillary_risk_counts=ancillary)
    st.to_csv(paths[2],index=False,encoding='utf-8-sig',mode='x')
    with paths[0].open('x',encoding='utf-8') as h:json.dump(out,h,ensure_ascii=False,indent=2,default=lambda x:float(x))
    with paths[1].open('x',encoding='utf-8') as h:
        h.write('补充数值审计\nVerification Status: ANALYZED\n\n')
        h.write(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:float(x)))
    print(json.dumps(dict(agreement={k:v for k,v in agreement.items() if k!='rater_pair_counts'},signals=signal,power=power_result,deviation=devreport,risk=risk,simpson_flags=[s for s in simpson if s['any_subgroup_opposite']]),ensure_ascii=False,indent=2,default=lambda x:float(x)))
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
