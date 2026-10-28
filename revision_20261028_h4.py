"""H4 estimand and scope revision. Source files are read-only; outputs use exclusive creation.
Run: python revision_20261028_h4.py
Uses supplied daily predictions (not a re-fit of the unavailable original daily estimator).
All model specifications below are fixed before fitting and all results are retained.
Dependencies: numpy, pandas, scipy, scikit-learn, statsmodels.
"""
from pathlib import Path
import sys,json,warnings,hashlib,platform,time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
ROOT=Path(__file__).resolve().parent
PREFIX='revision_20261028_h4'
def read(p):return pd.read_csv(ROOT/p)
def save(s,obj):
    p=ROOT/(PREFIX+s)
    if isinstance(obj,pd.DataFrame):obj.to_csv(p,index=False,encoding='utf-8-sig',mode='x')
    else:
        with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,default=lambda z:float(z))
def main():
    suffixes=['_dates.csv','_table5.csv','_exam_comparison.csv','_models.csv','_model_terms.csv','_figure6b.csv','_results.json','_report.txt','_log.txt']
    for s in suffixes:
        if (ROOT/(PREFIX+s)).exists():raise FileExistsError(ROOT/(PREFIX+s))
    log=(ROOT/(PREFIX+'_log.txt')).open('x',encoding='utf-8')
    def progress(s):print(s,flush=True);log.write(s+'\n');log.flush()
    d=read('data/06_analysis_ready/day_level_extension.csv');ema=read('data/05_ambulatory/ema_prompts.csv')
    ema['actual_date']=pd.to_datetime(ema.prompt_id.str.extract(r'-(\d{8})-')[0],format='%Y%m%d',errors='coerce')
    dates=ema.groupby(['pid','study_day']).agg(actual_date=('actual_date','first'),distinct_dates=('actual_date','nunique')).reset_index()
    d=d.merge(dates,on=['pid','study_day'],how='left',validate='one_to_one')
    assert d.actual_date.notna().all() and d.distinct_dates.eq(1).all()
    d['weekend_actual']=(d.actual_date.dt.dayofweek>=5).astype(int)
    d['scenario_complete']=d.scenario.fillna('unclassified')
    checks=dict(n_person_days=len(d),n_persons=d.pid.nunique(),other_days=int(d.scenario.eq('other').sum()),unclassified_days=int(d.scenario.isna().sum()),weekend_date_mismatches=int(d.weekend.ne(d.weekend_actual).sum()),study_week_mismatches=int(d.week.ne(np.minimum((d.study_day-1)//7+1,4)).sum()),correct_label_mismatches=int(d.correct.ne(d.pred.eq(d.y).astype(int)).sum()),date_validation='Calendar date extracted from EMA prompt_id; all records for the same pid and study_day must agree.',exam_week_limitation='Exam-week labels are supplied; no institutional examination calendar is provided for independent validation.',constant_correct_participants=int(d.groupby('pid').correct.nunique().eq(1).sum()))
    assert checks['correct_label_mismatches']==0
    save('_dates.csv',d[['pid','study_day','actual_date','weekend','weekend_actual','scenario','exam_week','week']])
    persons=np.sort(d.pid.unique());pid_index={p:i for i,p in enumerate(persons)};rng=np.random.default_rng(20261028)
    draws=rng.multinomial(len(persons),np.full(len(persons),1/len(persons)),size=2000).astype(float)
    def metrics(z):
        counts=np.zeros((len(persons),4))
        for pid,v in z.groupby('pid'):
            counts[pid_index[pid]]=[((v.y==1)&(v.pred==1)).sum(),(v.y==1).sum(),((v.y==0)&(v.pred==0)).sum(),(v.y==0).sum()]
        c=counts.sum(0);boot=draws@counts
        with np.errstate(invalid='ignore',divide='ignore'):
            ba=.5*(c[0]/c[1]+c[2]/c[3]);acc=(c[0]+c[2])/(c[1]+c[3])
            bb=.5*(boot[:,0]/boot[:,1]+boot[:,2]/boot[:,3]);ab=(boot[:,0]+boot[:,2])/(boot[:,1]+boot[:,3])
        result=dict(n_person_days=len(z),n_persons=z.pid.nunique(),n_case_days=int(c[1]),n_control_days=int(c[3]),accuracy=acc,accuracy_lo=np.nanquantile(ab,.025),accuracy_hi=np.nanquantile(ab,.975),balanced_accuracy=ba,ba_lo=np.nanquantile(bb,.025),ba_hi=np.nanquantile(bb,.975),sensitivity=c[0]/c[1] if c[1] else np.nan,specificity=c[2]/c[3] if c[3] else np.nan,bootstrap_valid_ba=int(np.isfinite(bb).sum()))
        return result,ab,bb
    allrow=[]
    scenes=['all','exam','academic_routine','career_entrance_exam','interpersonal','no_stressor','other','unclassified']
    strata=[('overall',lambda z:np.ones(len(z),dtype=bool)),('weekday',lambda z:z.weekend_actual.eq(0)),('weekend',lambda z:z.weekend_actual.eq(1)),('exam_week',lambda z:z.exam_week.eq(1)),('non_exam_week',lambda z:z.exam_week.eq(0))]+[(f'week{i}',lambda z,i=i:z.week.eq(i)) for i in range(1,5)]
    for scenario in scenes:
        z=d if scenario=='all' else d[d.scenario_complete.eq(scenario)]
        for stratum,mask in strata:
            zz=z.loc[mask(z)];r,_,_=metrics(zz);allrow.append(dict(scenario=scenario,stratum=stratum,**r,ci_method='2000 participant-cluster bootstrap resamples; percentile 95% CI',note='Sparse subgroup; descriptive only' if len(zz)<50 else ''))
    table=pd.DataFrame(allrow);save('_table5.csv',table)
    scopes={'all_rows':d,'six_known_scenarios':d[d.scenario.notna()].copy(),'five_scenarios_excluding_other':d[d.scenario.notna()&~d.scenario.eq('other')].copy()}
    examrows=[]
    for scope,z in scopes.items():
        r0,a0,b0=metrics(z[z.exam_week.eq(0)]);r1,a1,b1=metrics(z[z.exam_week.eq(1)])
        for value,r in [('non_exam_week',r0),('exam_week',r1)]:examrows.append(dict(scope=scope,contrast=value,**r))
        examrows.append(dict(scope=scope,contrast='exam_minus_non_exam',n_person_days=len(z),n_persons=z.pid.nunique(),accuracy=r1['accuracy']-r0['accuracy'],accuracy_lo=np.nanquantile(a1-a0,.025),accuracy_hi=np.nanquantile(a1-a0,.975),balanced_accuracy=r1['balanced_accuracy']-r0['balanced_accuracy'],ba_lo=np.nanquantile(b1-b0,.025),ba_hi=np.nanquantile(b1-b0,.975),bootstrap_valid_ba=int(np.isfinite(b1-b0).sum())))
    save('_exam_comparison.csv',pd.DataFrame(examrows))
    progress('Date validation, corrected Table5 and participant-cluster CIs completed.')
    models=[];terms=[]
    specs=[]
    for scope in scopes:
        for corr in ['independence','exchangeable']:
            specs.append((scope,'unadjusted',corr,'correct ~ exam_week'))
            if scope!='all_rows':specs.append((scope,'adjusted',corr,'correct ~ exam_week + C(scenario) + weekend_actual + week'))
    for corr in ['independence','exchangeable']:
        specs.append(('all_rows','adjusted_missing_as_category',corr,'correct ~ exam_week + C(scenario_complete) + weekend_actual + week'))
    def extract(name,fit,target,n,ng,**extra):
        ci=fit.conf_int();b=fit.params[target];se=fit.bse[target]
        r=dict(model=name,n_person_days=n,n_persons=ng,term=target,b=b,se=se,OR=np.exp(b),OR_lo=np.exp(ci.loc[target,0]),OR_hi=np.exp(ci.loc[target,1]),p=fit.pvalues[target],**extra)
        for t in fit.params.index:terms.append(dict(model=name,term=t,b=fit.params[t],se=fit.bse[t],p=fit.pvalues[t],ci_lo=ci.loc[t,0],ci_hi=ci.loc[t,1]))
        return r
    for scope,adjust,corr,formula in specs:
        z=scopes[scope];name=f'GEE_{scope}_{adjust}_{corr}';progress(name)
        cov=sm.cov_struct.Independence() if corr=='independence' else sm.cov_struct.Exchangeable()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always');f=smf.gee(formula,'pid',z,family=sm.families.Binomial(),cov_struct=cov).fit(maxiter=200)
        r=extract(name,f,'exam_week',len(z),z.pid.nunique(),family='binomial_logit_GEE',scope=scope,adjustment=adjust,working_correlation=corr,formula=formula,converged=bool(f.converged),correlation_parameter=float(f.cov_struct.dep_params) if corr=='exchangeable' else 0.,warnings=' | '.join(dict.fromkeys(str(v.message) for v in w)),estimand='Population-averaged association with correctness; NOT a subject-specific or fixed-person odds ratio. Robust sandwich covariance.')
        if adjust!='unadjusted':
            ids=[i for i,t in enumerate(f.params.index) if t.startswith('C(scenario')];L=np.eye(len(f.params))[ids]
            wt=f.wald_test(L,scalar=True);r.update(scenario_wald_chi2=float(wt.statistic),scenario_wald_df=len(ids),scenario_wald_p=float(wt.pvalue))
        models.append(r)
    # Prespecified within-between decomposition on all six known scenarios.
    z=scopes['six_known_scenarios'].copy()
    base=pd.DataFrame({'exam_week':z.exam_week,'weekend_actual':z.weekend_actual,'week':z.week},index=z.index)
    dummy=pd.get_dummies(z.scenario,prefix='scenario',drop_first=True,dtype=float)
    xb=pd.concat([base,dummy],axis=1)
    for adjust in ['unadjusted','adjusted']:
        selected=['exam_week'] if adjust=='unadjusted' else list(xb.columns)
        design=pd.DataFrame(index=z.index)
        for c in selected:
            mu=xb[c].groupby(z.pid).transform('mean');design[c+'_within']=xb[c]-mu;design[c+'_between']=mu
        design=sm.add_constant(design);name='GEE_Mundlak_'+adjust;progress(name)
        with warnings.catch_warnings(record=True) as w:
            f=sm.GEE(z.correct,design,groups=z.pid,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Independence()).fit(maxiter=200)
        models.append(extract(name,f,'exam_week_within',len(z),z.pid.nunique(),family='binomial_logit_GEE_within_between',scope='six_known_scenarios',adjustment=adjust,working_correlation='independence',converged=bool(f.converged),formula=' + '.join(design.columns),warnings=' | '.join(str(v.message) for v in w),estimand='Within-person centered exposure association conditional on participant exposure means, estimated by population-averaged GEE; not an individual-specific random-intercept effect.'))
    # Conditional logistic participant fixed effects: informative participants only.
    informative=z.groupby('pid').correct.transform('nunique').eq(2);zz=z.loc[informative].copy()
    for adjust in ['unadjusted','adjusted']:
        X=xb.loc[zz.index,['exam_week']] if adjust=='unadjusted' else xb.loc[zz.index]
        name='conditional_logit_FE_'+adjust;progress(name)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always');f=ConditionalLogit(zz.correct,X,groups=zz.pid).fit(method='BFGS',maxiter=300,disp=False)
        models.append(extract(name,f,'exam_week',len(zz),zz.pid.nunique(),family='conditional_logit_fixed_person',scope='six_known_scenarios_informative_correctness',adjustment=adjust,working_correlation='conditional likelihood',formula=' + '.join(X.columns),warnings=' | '.join(dict.fromkeys(str(v.message) for v in w)),estimand='Within-participant conditional odds ratio among participants with both correct and incorrect days; model-based CI, conditional day independence assumed. Drops participants with invariant correctness.',dropped_persons=z.pid.nunique()-zz.pid.nunique(),dropped_person_days=len(z)-len(zz)))
    save('_models.csv',pd.DataFrame(models));save('_model_terms.csv',pd.DataFrame(terms))
    progress('GEE and participant fixed-effect sensitivity models completed.')
    # Verify Figure6B independently from saved participant-day predictions of the documented reconstruction.
    fp=ROOT/'audit_20261028_daily_refined_within_person_predictions.csv'
    pred=pd.read_csv(fp);methods=['raw','lab_anchored','first_week_z','first_week_centered'];group_auc={}
    unique=np.sort(pred.pid.unique());boot_ix=rng.integers(0,len(unique),(2000,len(unique)));frows=[]
    for method in methods:
        wa=pd.Series({pid:roc_auc_score(v.highstress,v[method]) for pid,v in pred.groupby('pid') if v.highstress.nunique()==2}).reindex(unique)
        if wa.isna().any():raise ValueError('Some participants lack both stress classes; explicitly define paired denominator before averaging')
        group_auc[method]=wa.to_numpy()
        r=dict(method=method,n_person_days=len(pred),n_persons=len(unique),pooled_auc=roc_auc_score(pred.highstress,pred[method]),mean_within_person_auc=wa.mean(),within_auc_lo=np.quantile(wa.to_numpy()[boot_ix].mean(1),.025),within_auc_hi=np.quantile(wa.to_numpy()[boot_ix].mean(1),.975),data_basis='Reconstructed daily EMA uses only physiology-valid prompts; Days8-30; target above participant median stress on valid days; ordinary core-trained logistic regression.')
        orig={'raw':.552,'lab_anchored':.551,'first_week_z':.547}
        r['original_mean_within_auc']=orig.get(method,np.nan);r['difference_from_original_within']=r['mean_within_person_auc']-r['original_mean_within_auc']
        frows.append(r)
    for r in frows:
        delta=group_auc[r['method']]-group_auc['raw'];bd=delta[boot_ix].mean(1)
        r.update(within_delta_vs_raw=delta.mean(),within_delta_lo=np.quantile(bd,.025),within_delta_hi=np.quantile(bd,.975))
    supplied=read('audit_20261028_daily_refined_within_person.csv').set_index('method')
    for r in frows:
        s=supplied.loc[r['method']]
        assert abs(r['pooled_auc']-s.auc)<1e-12 and abs(r['mean_within_person_auc']-s.within_person_auc)<1e-12
        r.update(pooled_auc_lo=s.lo,pooled_auc_hi=s.hi,pooled_delta_vs_raw=s.delta_vs_raw,pooled_delta_lo=s.delta_lo,pooled_delta_hi=s.delta_hi,n_core_person_days=int(s.n_core),source_probability_sha256=hashlib.sha256(fp.read_bytes()).hexdigest())
    save('_figure6b.csv',pd.DataFrame(frows))
    mdf=pd.DataFrame(models);primary=mdf[mdf.model.eq('GEE_six_known_scenarios_adjusted_exchangeable')].iloc[0]
    independence=mdf[mdf.model.eq('GEE_six_known_scenarios_adjusted_independence')].iloc[0]
    fee=mdf[mdf.model.eq('conditional_logit_FE_adjusted')].iloc[0]
    exam=pd.DataFrame(examrows);e0=exam[(exam.scope=='all_rows')&(exam.contrast=='non_exam_week')].iloc[0];e1=exam[(exam.scope=='all_rows')&(exam.contrast=='exam_week')].iloc[0];delta=exam[(exam.scope=='all_rows')&(exam.contrast=='exam_minus_non_exam')].iloc[0]
    methods_text='Day-level correctness was analyzed using binomial generalized estimating equations clustered by participant. Prespecified sensitivity analyses crossed independence and exchangeable working correlations with unadjusted and scenario-, weekend-, and study-week-adjusted specifications. All six observed scenarios, including other, were retained; 18 days without a scenario label were excluded from scenario-adjusted analyses and retained in unadjusted summaries. A separate sensitivity model treated missing scenario labels as a category. Weekend status was verified against the calendar dates encoded in EMA prompt identifiers. Participant-cluster bootstrap confidence intervals used 2,000 resamples. Conditional logistic regression with participant fixed effects was used to assess within-participant associations among participants with variation in correctness.'
    results_text=(f'Across 4,947 extension-sample person-days, unadjusted accuracy was {e1.accuracy:.3f} during examination weeks (1,287 days) and {e0.accuracy:.3f} otherwise (3,660 days); balanced accuracy was {e1.balanced_accuracy:.3f} and {e0.balanced_accuracy:.3f}, respectively. The difference in balanced accuracy was {delta.balanced_accuracy:.3f} (participant-cluster bootstrap 95% CI [{delta.ba_lo:.3f}, {delta.ba_hi:.3f}]). The adjusted examination-week odds ratio for correctness was {primary.OR:.3f} (95% CI [{primary.OR_lo:.3f}, {primary.OR_hi:.3f}], p={primary.p:.4f}) with an exchangeable working correlation and {independence.OR:.3f} (95% CI [{independence.OR_lo:.3f}, {independence.OR_hi:.3f}], p={independence.p:.4f}) with independence. These population-averaged GEE coefficients are specification-dependent and do not establish a uniform decline in marginal performance during examinations. Participant fixed-effects analysis estimated a conditional odds ratio of {fee.OR:.3f} (95% CI [{fee.OR_lo:.3f}, {fee.OR_hi:.3f}]) among {int(fee.n_persons)} informative participants. Failure to reject a difference was not interpreted as evidence of equivalence.')
    figure_text='Using the documented reconstruction, pooled AUCs for raw, laboratory-anchored, and first-week-z-standardized inputs were 0.519, 0.520, and 0.544, whereas the corresponding equally weighted mean within-participant AUCs were 0.560, 0.561, and 0.553. The pooled improvement with first-week standardization therefore did not reflect improved average within-participant discrimination. These are reconstructed analyses; the original values 0.552, 0.551, and 0.547 cannot be combined with the reconstructed pooled estimates. First-week centering alone is a separate sensitivity specification (pooled AUC 0.552; mean within-participant AUC 0.559).'
    notes=['Time-of-day rows are omitted because the supplied table contains daily predictions, not morning/afternoon/evening/night predictions. The four original rows with n=24552 are not supported by these 4947 daily records.','The other category has 428 days and must be included in descriptive Table5. Its exclusion from any inferential scope must be named explicitly. The 18 unclassified days are displayed separately.','Exchangeable GEE is population-averaged, not subject-specific. Differences from independence GEE may reflect cluster weighting, working-correlation assumptions and mean-model misspecification; a directional discrepancy must be reported rather than choosing the significant specification.','Scenario-adjusted models compare a conditional regression association, whereas pooled accuracy and balanced accuracy are descriptive marginal averages. They need not move in the same direction.','No equivalence margin was specified and no equivalence test was run. Non-significance is not evidence of stability/equivalence.','The first-week-reference label must specify centering versus within-person z standardization; they produce materially different pooled AUCs.','The stress median is defined using each participant\'s entire eligible monitoring period; this is a retrospective discrimination task and not prospective forecasting.']
    output=dict(material_passport=dict(mode='H4 statistical_revision',status='RECOMPUTED_FROM_SUPPLIED_PREDICTIONS',inputs='source daily predictions and previously reconstructed within-person predictions',scope='H4 estimand audit; not a claim of rerunning missing original estimator'),checks=checks,model_specifications=[dict(scope=a,adjustment=b,correlation=c,formula=f) for a,b,c,f in specs],primary_reporting_recommendation='Report descriptive examination-week contrasts and the complete GEE specification sensitivity together. Retain all six known scenarios. Do not claim uniform examination-week deterioration or equivalence.',table5_rows=len(table),bootstrap_seed=20261028,bootstrap_n=2000,methods_english=methods_text,results_english=results_text,figure6b_english=figure_text,notes=notes,software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,statsmodels=sm.__version__),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/06_analysis_ready/day_level_extension.csv',ROOT/'data/05_ambulatory/ema_prompts.csv',fp]})
    save('_results.json',output)
    with (ROOT/(PREFIX+'_report.txt')).open('x',encoding='utf-8') as f:
        f.write('H4 修正统计审定\n\n'+json.dumps(checks,ensure_ascii=False,indent=2)+'\n\n')
        f.write(pd.DataFrame(examrows).to_string(index=False)+'\n\n')
        f.write(mdf[['model','n_person_days','n_persons','OR','OR_lo','OR_hi','p']].to_string(index=False)+'\n\n')
        f.write('\n'.join(notes)+'\n\nMETHODS\n'+methods_text+'\n\nRESULTS\n'+results_text+'\n\nFIGURE6B\n'+figure_text+'\n\n'+pd.DataFrame(frows).to_string(index=False))
    progress('H4 revision complete.');log.close()
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
