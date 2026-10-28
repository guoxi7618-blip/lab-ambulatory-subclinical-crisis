"""Read-only audit of supplied tables; creates new, exclusive output files.
Run: python audit_20261028_integrity.py [--prefix NEW_PREFIX]
No source files, configuration or directories are changed.
"""
from pathlib import Path
import argparse, json, hashlib, sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent
ap=argparse.ArgumentParser(); ap.add_argument('--prefix',default='audit_20261028_integrity_v2'); args=ap.parse_args()
OUT=ROOT/args.prefix
if OUT.with_suffix('.json').exists() or (ROOT/(args.prefix+'_review.txt')).exists():
    raise FileExistsError('Output exists; use a new --prefix to preserve history.')
data={str(f.relative_to(ROOT/'data')).replace('\\','/'):pd.read_csv(f) for f in sorted((ROOT/'data').rglob('*.csv'))}
def get(name): return next(d for p,d in data.items() if p.endswith('/'+name+'.csv'))
report={'scope':'All supplied CSV files; no raw signals or provenance authentication available','inventory':[], 'checks':[], 'findings':[], 'limitations':[]}
def native(v):
    if isinstance(v,(np.integer,np.floating)): return v.item()
    if isinstance(v,(np.bool_,)): return bool(v)
    if isinstance(v,pd.Timestamp):return str(v)
    raise TypeError(type(v).__name__)
def check(name, result):report['checks'].append({'check':name,**result})
def finding(level,title,detail,examples=None):
    report['findings'].append({'level':level,'title':title,'detail':detail,'examples':examples or []})
def examples(d,mask,cols=None,n=8): return d.loc[mask,cols or list(d.columns)].head(n).replace({np.nan:None}).to_dict('records')
def cmp(name,actual,expected,tol=1e-8):
    actual,expected=actual.align(expected)
    a=pd.to_numeric(actual); b=pd.to_numeric(expected)
    valid=a.notna()&b.notna(); delta=(a-b).abs(); bad=(delta>tol)&valid
    result={'n':len(a),'both_nonmissing':int(valid.sum()),'mismatches':int(bad.sum()),'missingness_disagreements':int((a.isna()!=b.isna()).sum()),'max_absolute_difference':float(delta.max()) if valid.any() else None,'tolerance':tol,'examples':[{'key':str(i),'stored':float(a.loc[i]),'recomputed':float(b.loc[i])} for i in a.index[bad][:5]]}
    check(name,result); return result

p=get('participant_characteristics'); pid=set(p.pid); pa=p.set_index('pid'); analy=p.query('analyzable==1'); pf=get('person_level_features').set_index('pid')
keys={'participant_characteristics':['pid'],'participant_flow':['batch','stage'],'reference_standard':['pid'],'reevaluation_day31':['pid'],'double_rating':['pid'],'sds_sas_items':['pid','session'],'covariates':['pid'],'lab_phase_features_ecg_eda':['pid','session','phase'],'lab_eeg_features':['pid','session','condition','channel'],'lab_eeg_quality':['pid','session'],'lab_mist_trials':['pid','session','block','trial'],'lab_subjective_ratings':['pid','session','timepoint'],'lab_wrist_concurrent_features':['pid','session','phase'],'ema_prompts':['prompt_id'],'wrist_daily_features':['pid','study_day'],'person_level_features':['pid'],'prompt_level_h2':['prompt_id'],'day_level_extension':['pid','study_day'],'extension_predicted_probabilities':['pid'],'decision_thresholds':['model'],'model_freeze_record':['record'],'Table1_sample':['group'],'Table2_data_quality':['index'],'Table3_lab_markers':['measure'],'Table4_identification':['model','sample'],'Table5_stability':['scenario','stratum'],'TableS6_feature_contributions':['modality','feature'],'TableS6b_eeg_channel_attributions':['channel'],'TableS8_marker_replication':['measure'],'TableS9_simple_model_checks':['model'],'Figure6B_within_person_calibration':['reference'],'Figure3ABC_phase_means':['group','phase'],'Figure3D_eeg_topography':['measure','channel'],'Figure3E_aperiodic_exponent':['pid'],'Figure4A_lab_vs_ambulatory':['pid'],'Figure4B_scenario_means':['y','scenario'],'Figure5_predictions':['pid'],'Figure6A_stability':['scenario','stratum'],'codebook':['file','variable']}
for path,d in data.items():
    k=keys.get(Path(path).stem)
    info={'file':path,'shape':list(d.shape),'sha256':hashlib.sha256((ROOT/'data'/path).read_bytes()).hexdigest(),'duplicate_rows':int(d.duplicated().sum()),'key':k,'duplicate_keys':int(d.duplicated(k).sum()) if k else None,'missing_by_column':{c:int(n) for c,n in d.isna().sum().items() if n},'unknown_pids':sorted(set(d.pid)-pid) if 'pid' in d else []}
    report['inventory'].append(info)
    if info['duplicate_keys'] or info['unknown_pids']:finding('ERROR','Key/reference failure',info)
check('Cohort counts',{'counts':p.groupby(['sample','group','analyzable']).size().rename('n').reset_index().to_dict('records'),'status':p.groupby(['sample','status','analyzable']).size().rename('n').reset_index().to_dict('records')})
check('Person-level analysis cohort membership',{'missing':sorted(set(analy.pid)-set(pf.index)),'unexpected':sorted(set(pf.index)-set(analy.pid))})
cmp('Person labels versus enrolled group',pf.y,pa.group.eq('subclinical').astype(int).reindex(pf.index))
for c in ['sds_standard','sas_standard']:cmp('Person '+c,pf[c],pa[c].reindex(pf.index))
ref=get('reference_standard').set_index('pid')
check('Reference category matches group',{'mismatches':int((ref.category.eq('subclinical_psychological_crisis')!=pa.group.eq('subclinical')).sum())})
cmp('Reference severity',ref.severity,pa.severity)
flow=get('participant_flow')
for batch,d in flow.groupby('batch'):
    vals=d.set_index('stage').n; excluded=vals.drop(['screened','invited','attended_evaluation']).sum()
    check('Participant flow balance '+batch,{'attended':int(vals.attended_evaluation),'disposition_sum':int(excluded),'balanced':bool(excluded==vals.attended_evaluation)})

# Questionnaire scoring: raw item responses need conventional reverse coding.
q=get('sds_sas_items')
for scale,reverse in [('sds',[2,5,6,11,12,14,16,17,18,20]),('sas',[5,9,13,17,19])]:
    items=q[[scale+str(i) for i in range(1,21)]]; scored=items.copy()
    for i in reverse:scored[scale+str(i)]=5-scored[scale+str(i)]
    check(scale+' scoring',{'n':len(q),'item_range':[float(items.min().min()),float(items.max().max())],'outside_1_4':int(((items<1)|(items>4)).sum().sum()),'unreversed_sum_matches':int(items.sum(axis=1).eq(q[scale+'_raw']).sum()),'reverse_coded_sum_matches':int(scored.sum(axis=1).eq(q[scale+'_raw']).sum()),'floor_standard_score_matches':int(np.floor(q[scale+'_raw']*1.25).eq(q[scale+'_standard']).sum()),'reverse_items':reverse})
    cmp(scale+' D0 versus demographics',q.query("session=='D0'").set_index('pid')[scale+'_standard'],pa[scale+'_standard'])
day31_ids=set(q.query("session=='D31'").pid)
reeval=get('reevaluation_day31'); lab=get('lab_phase_features_ecg_eda'); qual=get('lab_eeg_quality'); lab31=set(lab.query("session=='D31'").pid)
check('D31 follow-up coverage',{'questionnaires_n':len(day31_ids),'lab_n':len(lab31),'counselor_n':len(reeval),'questionnaire_without_lab':sorted(day31_ids-lab31),'lab_without_questionnaire':sorted(lab31-day31_ids)})
if day31_ids-lab31:finding('REQUIRES_EXPLANATION','D31 questionnaire coverage differs from laboratory follow-up',f'Questionnaire D31 records exist for {len(day31_ids)} people, versus {len(lab31)} laboratory retests and {len(reeval)} counselor re-evaluations. This may be remote follow-up but is undocumented.',sorted(day31_ids-lab31))

# Actual monitoring denominators and inclusion criteria.
ema=get('ema_prompts'); w=get('wrist_daily_features'); rnd=ema[ema.prompt_type.str.startswith('random')].copy(); rnd['completed']=rnd.status.eq('completed')
compliance=rnd.groupby('pid').completed.mean(); counts=rnd.groupby('pid').completed.agg(['sum','size']); valid_days=w.groupby('pid').valid_day.sum()
for subset,label in [(rnd,'all monitored'),(rnd[rnd.pid.isin(analy.pid)],'analyzable')]:
    check('EMA completion '+label,{'people':subset.pid.nunique(),'scheduled':len(subset),'completed':int(subset.completed.sum()),'pooled_rate':float(subset.completed.mean()),'person_mean':float(subset.groupby('pid').completed.mean().mean()),'person_sd':float(subset.groupby('pid').completed.mean().std())})
cmp('Person compliance',pf.compl,compliance.reindex(pf.index));cmp('Duplicate EMA compliance',pf.ema_compliance,pf.compl)
check('Valid days',{'n_people':len(valid_days),'median':float(valid_days.median()),'q25':float(valid_days.quantile(.25)),'q75':float(valid_days.quantile(.75)),'less_than20':int((valid_days<20).sum()),'valid_but_wear_under16':int(((w.valid_day==1)&(w.wear_hours<16)).sum()),'wear_ge16_but_invalid':int(((w.valid_day==0)&(w.wear_hours>=16)).sum())})
eligible=(valid_days.reindex(pa.index).fillna(0)>=20)&(compliance.reindex(pa.index).fillna(0)>=.5)&pa.index.isin(set(lab.query("session=='D0'").pid))
cmp('Reconstructed inclusion flag',pa.analyzable,eligible.astype(int))
withdrawn=analy[analy.status!='completed']
if len(withdrawn):finding('REQUIRES_EXPLANATION','Withdrawn participants retained when thresholds met',f'{len(withdrawn)} analyzable participants have withdrew_during_tracking status. This can be defensible if retained-data consent permits and inclusion is based on available data, but needs explicit reporting.',withdrawn[['pid','sample','status','withdrawal_study_day']].to_dict('records'))
for src,label in [(w,'wrist'),(ema,'EMA')]:
    j=src.merge(p[['pid','withdrawal_study_day']],on='pid',how='left');bad=j.study_day>j.withdrawal_study_day
    check(label+' records after withdrawal',{'n_rows':int(bad.sum()),'examples':examples(j,bad,['pid','study_day','withdrawal_study_day'])})
for sexgrp in ['subclinical','control']:
    ids=set(p.query('group==@sexgrp').pid); vals=compliance[compliance.index.isin(ids)]
    check('EMA group '+sexgrp,{'person_mean':float(vals.mean()),'n':len(vals),'analyzable_mean':float(vals[vals.index.isin(analy.pid)].mean())})

# Dates are carried only in prompt_id, not explicit session dates.
ema['date']=pd.to_datetime(ema.prompt_id.str.split('-').str[1],format='%Y%m%d',errors='coerce')
check('Monitoring dates',{'invalid_dates':int(ema.date.isna().sum()),'by_sample':ema.merge(p[['pid','sample']],on='pid').groupby('sample').date.agg(['min','max']).astype(str).reset_index().to_dict('records')})
lab_dates=(ema.date-pd.to_timedelta(ema.study_day,unit='d')).groupby(ema.pid).agg(['min','max'])
check('Inferred D0 dates',{'nonconstant_person_start':int((lab_dates['min']!=lab_dates['max']).sum()),'extension_min':str(lab_dates.loc[lab_dates.index.str.startswith('E'),'min'].min()),'extension_max':str(lab_dates.loc[lab_dates.index.str.startswith('E'),'max'].max()),'extension_before_freeze_date':int((lab_dates.loc[lab_dates.index.str.startswith('E'),'min']<pd.Timestamp('2026-02-26')).sum()),'note':'Inferred from prompt_id date minus study_day; not an original laboratory timestamp.'})
for col in ['scheduled_time','push_time','open_time','submit_time']:
    ema[col+'_seconds']=pd.to_timedelta(ema[col],errors='coerce').dt.total_seconds()
done=ema.status.eq('completed'); lag=ema.submit_time_seconds-ema.push_time_seconds
check('EMA submission latency',{'completed_n':int(done.sum()),'completed_after30min':int(((lag>1800)&done).sum()),'negative_latency':int(((lag<0)&done).sum()),'completed_without_submit':int((done&ema.submit_time.isna()).sum()),'max_minutes':float(lag[done].max()/60)})

# Logical numeric ranges, not clinical diagnosis thresholds.
ranges=[('participant_characteristics','age',18,25),('participant_characteristics','phq9_total',0,27),('participant_characteristics','gad7_total',0,21),('participant_characteristics','severity',0,10),('lab_phase_features_ecg_eda','hr_bpm',25,220),('lab_phase_features_ecg_eda','rmssd_ms',0,400),('lab_phase_features_ecg_eda','resp_rate_bpm',3,50),('lab_phase_features_ecg_eda','scl_us',0,100),('lab_phase_features_ecg_eda','scr_amp_us',0,100),('lab_phase_features_ecg_eda','pct_corrected_beats',0,100),('wrist_daily_features','wear_hours',0,24),('wrist_daily_features','sleep_duration_h',0,24),('wrist_daily_features','nocturnal_rmssd_ms',0,500),('wrist_daily_features','resting_hr_bpm',25,220),('wrist_daily_features','steps',0,100000),('wrist_daily_features','mean_skin_temp_c',10,45),('ema_prompts','stress',0,100),('ema_prompts','stressor_severity',1,7)]
for name,col,lo,hi in ranges:
    d=get(name); bad=(d[col]<lo)|(d[col]>hi)
    check('Range '+name+'.'+col,{'plausibility_bounds':[lo,hi],'min':float(d[col].min()),'max':float(d[col].max()),'outside':int(bad.sum()),'examples':examples(d,bad,[c for c in ['pid','session','phase','study_day',col] if c in d])})
cmp('ln RMSSD log consistency',lab.ln_rmssd,np.log(lab.rmssd_ms),.0015)
cov=get('covariates');cmp('BMI from height and weight',cov.bmi,cov.weight_kg/(cov.height_cm/100)**2,.11)
rr=get('double_rating');check('Double rating agreement',{'n':len(rr),'category_kappa':float(cohen_kappa_score(rr.category_rater1,rr.category_rater2)),'disagreement_flag_mismatch':int((rr.disagreement!=rr.category_rater1.ne(rr.category_rater2).astype(int)).sum())})
check('EEG quality',{'session_counts':qual.groupby('session').size().to_dict(),'excluded_counts':qual.groupby('session').excluded_gt20pct_bad.sum().to_dict(),'exclusion_flag_mismatch':int((qual.excluded_gt20pct_bad!=qual.n_bad_channels.gt(6.4).astype(int)).sum()),'packet_loss_median':float(qual.packet_loss_pct.median()),'packet_loss_range':[float(qual.packet_loss_pct.min()),float(qual.packet_loss_pct.max())],'alpha_ratio_gt1_fraction':float(qual.alpha_reactivity_ec_eo.gt(1).mean()),'lsl_median':float(qual.lsl_offset_ms.median()),'lsl_max':float(qual.lsl_offset_ms.max())})

# Reconstruct analysis-ready features from lower-level aggregates.
l0=lab.query("session=='D0'"); base=l0.query("phase=='baseline'").set_index('pid'); ctrl=l0.query("phase=='control'").set_index('pid'); stress=l0[l0.phase.str.startswith('stress')].groupby('pid').mean(numeric_only=True); rec=l0.query("phase=='recovery2'").set_index('pid')
derived={'lab_hr_base':base.hr_bpm,'lab_lnrmssd_base':base.ln_rmssd,'lab_lnhf_base':base.ln_hf_ms2,'lab_hr_react':stress.hr_bpm-ctrl.hr_bpm,'lab_lnrmssd_react':stress.ln_rmssd-ctrl.ln_rmssd,'lab_scl_base':np.log(base.scl_us),'lab_scl_react':stress.scl_us-ctrl.scl_us,'lab_nsscr_stress':stress.ns_scr_per_min}
denom=stress.hr_bpm-base.hr_bpm; derived['lab_hr_recovery']=((stress.hr_bpm-rec.hr_bpm)/denom).where(denom>=1)
for c,v in derived.items():cmp('Reconstruct '+c,pf[c],v.reindex(pf.index),1e-7)
wv=w.query('valid_day==1').copy();wv['lnnoct']=np.log(wv.nocturnal_rmssd_ms)
means={'w_noct_lnrmssd_mean':'lnnoct','w_resthr_mean':'resting_hr_bpm','w_steps_mean':'steps','w_sleep_mean':'sleep_duration_h','w_scr_rest_mean':'scr_rate_rest_per_5min','w_temp_mean':'mean_skin_temp_c'}
for dest,src in means.items():cmp('Reconstruct '+dest,pf[dest],wv.groupby('pid')[src].mean().reindex(pf.index),1e-7)
for dest,src in {'w_noct_lnrmssd_sd':'lnnoct','w_resthr_sd':'resting_hr_bpm','w_sleep_sd':'sleep_duration_h','w_midpoint_sd':'sleep_midpoint_min'}.items():cmp('Reconstruct '+dest,pf[dest],wv.groupby('pid')[src].std().reindex(pf.index),1e-7)
cmp('Reconstruct w_steps_max',pf.w_steps_max,wv.groupby('pid').steps.max().reindex(pf.index))
complete=rnd[rnd.completed].copy();complete['pa']=complete[['pa_happy','pa_relaxed','pa_energetic']].mean(axis=1);complete['na']=complete[['na_down','na_tense','na_irritated']].mean(axis=1);complete['interpersonal']=complete.stressor_type.eq('interpersonal').astype(int)
for dest,src in {'ema_stress_mean':'stress','ema_pa_mean':'pa','ema_na_mean':'na','ema_any_stressor':'stressor_since_last','ema_interpersonal':'interpersonal'}.items():cmp('Reconstruct '+dest,pf[dest],complete.groupby('pid')[src].mean().reindex(pf.index),1e-7)
cmp('Reconstruct ema_stress_sd',pf.ema_stress_sd,complete.groupby('pid').stress.std().reindex(pf.index),1e-7)
diary=ema[(ema.prompt_type=='diary')&done].copy();diary['rumination']=diary[['rumination1','rumination2']].mean(axis=1)
cmp('Reconstruct ema_rumination',pf.ema_rumination,diary.groupby('pid').rumination.mean().reindex(pf.index))
cmp('Reconstruct passive SI any',pf.ema_passive_si_any,diary.groupby('pid').passive_si.max().reindex(pf.index))
cmp('Reconstruct calibration noct',pf.cal_noct_minus_lab,pf.w_noct_lnrmssd_mean-pf.lab_lnrmssd_base)
cmp('Reconstruct calibration HR',pf.cal_resthr_minus_lab,pf.w_resthr_mean-pf.lab_hr_base)
cmp('Reconstruct clamped event response ratio',pf.cal_event_resp_over_react,pf.cal_event_resp/pf.lab_hr_react.clip(lower=1))
ext=get('day_level_extension');jj=ext.merge(w,on=['pid','study_day'],suffixes=('','_source'),validate='one_to_one')
for c in ['wear_hours','valid_day','sleep_duration_h','nocturnal_rmssd_ms','resting_hr_bpm','steps','scr_rate_rest_per_5min','mean_skin_temp_c']:cmp('Day table source '+c,jj[c],jj[c+'_source'])
cmp('Day table cal_noct',ext.cal_noct,np.log(ext.nocturnal_rmssd_ms)-ext.lab_lnrmssd_base)
cmp('Day table cal_hr',ext.cal_hr,ext.resting_hr_bpm-ext.lab_hr_base)
hh=get('prompt_level_h2');j=hh.merge(rnd,on='prompt_id',suffixes=('','_source'),validate='one_to_one')
for c in ['stress','study_day','stressor_since_last']:cmp('H2 source '+c,j[c],j[c+'_source'])
cmp('H2 lab_react',hh.set_index('prompt_id').lab_react,hh.set_index('prompt_id').pid.map(pf.lab_hr_react))
cmp('H2 hr_c',hh.hr_c,hh.hr_pre-hh.groupby('pid').hr_pre.transform('mean'))
check('H2 population',{'people':hh.pid.nunique(),'sample_counts':hh[['pid']].drop_duplicates().merge(p[['pid','sample']],on='pid')['sample'].value_counts().to_dict()})

# EEG relative power sums and exclusion handling.
eeg=get('lab_eeg_features');power=eeg[['delta_rel','theta_rel','alpha_rel','beta_rel']].sum(axis=1);cmp('EEG relative band power sum',power,pd.Series(1.,index=power.index),.00021)
e0=eeg.query("session=='D0' and condition=='rest_ec'");valid_eeg=qual.query("session=='D0' and excluded_gt20pct_bad==0 and clean_s_rest_ec>=120").pid
aper=e0[e0.pid.isin(valid_eeg)].groupby('pid').aperiodic_exponent.mean();cmp('EEG aperiodic rest_ec mean with stated QC',pf.eeg_aper_mean,aper.reindex(pf.index),1e-7)
bad_eeg_ids=set(qual.query("session=='D0' and excluded_gt20pct_bad==1").pid)&set(pf.index)
check('EEG excluded participants',{'n':len(bad_eeg_ids),'retained_nonmissing_aperiodic':int(pf.loc[list(bad_eeg_ids),'eeg_aper_mean'].notna().sum()),'ids':sorted(bad_eeg_ids)})

# Extrema that are mathematically possible but unstable need substantive review.
extreme=pf.cal_scl_palm_equiv_rel.abs()>100
finding('REQUIRES_EXPLANATION','EDA calibration extreme ratios',f'cal_scl_palm_equiv_rel range {pf.cal_scl_palm_equiv_rel.min():.6f} to {pf.cal_scl_palm_equiv_rel.max():.6f}; {int(extreme.sum())}/{len(pf)} have absolute value >100. These are transformed/calibrated units, not physical SCL; inspect small-denominator guards and report sensitivity.',pf.loc[extreme,['cal_scl_palm_equiv_rel','lab_scl_react','cal_scr_rel']].reset_index().to_dict('records'))
speed=w.steps>50000;check('Very high daily steps',{'gt50000':int(speed.sum()),'maximum':int(w.steps.max()),'examples':examples(w,speed,['pid','study_day','steps'])})
trial=get('lab_mist_trials');check('MIST logic',{'timeout_but_correct':int(((trial.timeout==1)&(trial.correct==1)).sum()),'negative_rt':int((trial.rt_ms<0).sum()),'rt_over_limit_without_timeout':int(((trial.rt_ms>trial.time_limit_ms)&(trial.timeout==0)).sum())})
z=lab.groupby(['pid','session']).resp_rate_bpm.nunique();stress_rr=lab[lab.phase.str.startswith('stress')].groupby(['pid','session']).resp_rate_bpm.nunique()
check('Respiration repeated-value pattern',{'sessions':len(stress_rr),'all_three_stress_blocks_identical':int(stress_rr.eq(1).sum()),'note':'Exact repetition may reflect a shared stage aggregate; raw ECG-derived respiration and algorithm are needed to assess.'})
finding('PROVENANCE_LIMIT','Repeated stage respiration values',f'All {int(stress_rr.eq(1).sum())}/{len(stress_rr)} sessions have identical respiration rate across three stress blocks; this needs a documented aggregation rule, not an assumption of fabricated data.')
report['limitations']=['No raw ECG/EDA/EEG/PPG/accelerometry or five-minute window tensor supplied, so signal processing and quality flags cannot be independently re-executed.','No actual OSF URL, preregistration, weights or original scripts; timestamp row is a self-reported record, not independent verification.','Some named wrist features require unavailable window-level data, so consistency of person-level tabulations is not equivalent to authenticating measurements.','Validity-day definition includes sleep worn; no separate sleep-worn flag is supplied. Wear>=16h with valid_day=0 is not by itself an error.','Physiological ranges used here are broad audit screens, not clinical or device-specific reference intervals.','Quantitative errors and unusual patterns cannot establish whether a dataset is simulated or fabricated.']
with OUT.with_suffix('.json').open('x',encoding='utf-8') as h:json.dump(report,h,ensure_ascii=False,indent=2,default=native)
lines=['Data integrity audit — 2026-10-28','All source files read only. Original files preserved.',f"CSV files checked: {len(report['inventory'])}",f"Checks: {len(report['checks'])}"]
for item in report['findings']:lines+=['',item['level']+' | '+item['title'],str(item['detail']),json.dumps(item['examples'],ensure_ascii=False,default=native)]
lines+=['','Numerical consistency checks with deviations (including documented rounding/definition uncertainty):']
for c in report['checks']:
    if c.get('mismatches',0) or c.get('missingness_disagreements',0):lines.append(json.dumps(c,ensure_ascii=False,default=native))
lines+=['','Selected summary checks:']
for c in report['checks']:
    if any(term in c['check'] for term in ['EMA completion','Valid days','Inferred D0','D31','EMA submission','H2 population','EEG quality','Very high','Reconstructed inclusion']):lines.append(json.dumps(c,ensure_ascii=False,default=native))
lines+=['','Limitations:']+report['limitations']
with (ROOT/(args.prefix+'_review.txt')).open('x',encoding='utf-8') as h:h.write('\n'.join(lines))
print(json.dumps({'files':len(data),'checks':len(report['checks']),'findings':report['findings'],'output':str(OUT.with_suffix('.json'))},ensure_ascii=False,default=native))
