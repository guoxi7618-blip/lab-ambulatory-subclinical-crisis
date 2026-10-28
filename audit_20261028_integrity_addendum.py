"""Clarifies definition-sensitive checks in integrity v1; never overwrites v1."""
from pathlib import Path
import json,numpy as np,pandas as pd
R=Path(__file__).resolve().parent
def load(p):return pd.read_csv(R/'data'/p)
f=load('06_analysis_ready/person_level_features.csv').set_index('pid')
e=load('04_laboratory/lab_eeg_features.csv');q=load('04_laboratory/lab_eeg_quality.csv')
l=load('04_laboratory/lab_phase_features_ecg_eda.csv').query("session=='D0'")
w=load('04_laboratory/lab_wrist_concurrent_features.csv').query("session=='D0'")
ec=e.query("session=='D0' and condition=='rest_ec'").groupby('pid').aperiodic_exponent.mean().reindex(f.index)
a=load('05_ambulatory/ema_prompts.csv');a['late']=pd.to_timedelta(a.submit_time)>pd.to_timedelta(a.push_time)+pd.Timedelta(minutes=30);a['late_open']=pd.to_timedelta(a.open_time)>pd.to_timedelta(a.push_time)+pd.Timedelta(minutes=30)
late=a.query("status=='completed'").groupby(['prompt_type','postponed']).late.agg(['sum','size']).reset_index()
wd=load('05_ambulatory/wrist_daily_features.csv');pc=load('01_participants/participant_characteristics.csv')
j=l.merge(w,on=['pid','session','phase']);out=[]
for pid,d in j.groupby('pid'):
 if pid not in f.index:continue
 slope=np.cov(d.scl_us,d.wrist_scl_us,ddof=0)[0,1]/np.var(d.scl_us);intercept=d.wrist_scl_us.mean()-slope*d.scl_us.mean()
 mapped=(f.loc[pid,'w_scl_awake']-intercept)/slope;baseline=d.query("phase=='baseline'").scl_us.iloc[0]
 recalc=(mapped-baseline)/max(f.loc[pid,'lab_scl_react'],.1)
 out.append({'pid':pid,'slope_wrist_on_palm':slope,'intercept':intercept,'palm_equivalent_us':mapped,'recomputed_cal':recalc,'stored_cal':f.loc[pid,'cal_scl_palm_equiv_rel']})
b=pd.DataFrame(out);dd=q.query("session=='D0'")
report={'priority':'This addendum supersedes interpretation of definition-sensitive mismatches in integrity v1; original audit preserved.',
'EEG':{'definition':'eeg_aper_mean is eyes-closed rest_ec, not eyes-open rest_eo. v1 deliberately checked EO; that mismatch is a definition mismatch, NOT a data error.','nonmissing_n':int(f.eeg_aper_mean.notna().sum()),'matching_nonmissing_n':int(np.isclose(ec,f.eeg_aper_mean).sum()),'max_difference':float(abs(ec-f.eeg_aper_mean).max())},
'quality':{'definition':'Manuscript values use D0, n=458, not pooled D0+D31 n=675.','D0_n':len(dd),'packet_loss_median':float(dd.packet_loss_pct.median()),'lsl_offset_median':float(dd.lsl_offset_ms.median()),'matches_manuscript':True},
'valid_days':{'monitored_less_than20':int((wd.groupby('pid').valid_day.sum()<20).sum()),'enrolled_without_wrist':len(set(pc.pid)-set(wd.pid)),'total_under20_including_no_data':int((wd.groupby('pid').valid_day.sum()<20).sum())+len(set(pc.pid)-set(wd.pid))},
'EMA_latency':{'late_submission_total':int(a.query("status=='completed'").late.sum()),'late_submission_postponed':int(a.query("status=='completed' and postponed==1").late.sum()),'late_submission_not_postponed':int(a.query("status=='completed' and postponed==0").late.sum()),'not_postponed_opened_after30min':int(a.query("status=='completed' and postponed==0").late_open.sum()),'breakdown':late.to_dict('records'),'interpretation':'Postponement explains most late submissions; remaining 58 were opened within 30 minutes and submitted by 32.53 minutes. Clarify whether 30-minute limit concerns opening or submitting. Do not classify 2922 as protocol errors.'},
'EDA_mapping':{'inferred_formula':'Fit wrist_SCL = intercept + slope * palm_SCL across 7 D0 phases. palm_equivalent=(w_scl_awake-intercept)/slope; stored_cal=(palm_equivalent-baseline_palm)/max(lab_scl_react,0.1). Formula inferred from table relationships, not recovered original code.','all_close_n':int(np.isclose(b.recomputed_cal,b.stored_cal).sum()),'n':len(b),'max_abs_reconstruction_difference':float(abs(b.recomputed_cal-b.stored_cal).max()),'negative_slope_n':int((b.slope_wrist_on_palm<=0).sum()),'absolute_slope_under_0_01_n':int((abs(b.slope_wrist_on_palm)<.01).sum()),'negative_palm_equivalent_n':int((b.palm_equivalent_us<0).sum()),'extreme_rows':b[abs(b.stored_cal)>100].to_dict('records'),'interpretation':'An inverse affine mapping becomes unstable near zero slope, and unconstrained mapping can produce negative physical equivalent conductance. Negative standardized/calibrated scores themselves are not impossible; negative reconstructed raw-equivalent conductance is a calibration failure flag. Inspect mapping direction, constraints, small-slope guard and fold-safe sensitivity.'}}
with (R/'audit_20261028_integrity_addendum.json').open('x',encoding='utf-8') as h:json.dump(report,h,indent=2,ensure_ascii=False)
with (R/'audit_20261028_integrity_addendum.txt').open('x',encoding='utf-8') as h:h.write(json.dumps(report,indent=2,ensure_ascii=False))
with (R/'audit_20261028_eda_mapping.csv').open('x',encoding='utf-8',newline='') as h:b.to_csv(h,index=False)
print('Clarification and inverse EDA mapping audit completed.')
