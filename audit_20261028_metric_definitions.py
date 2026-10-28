"""Distinguish calibration-in-the-large from jointly estimated intercept; audit Table5."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from audit_20261028_models import auc_components
R=Path(__file__).resolve().parent;D=R/'data';sys.stdout.reconfigure(encoding='utf-8')
p=pd.read_csv(D/'07_model/extension_predicted_probabilities.csv');t=pd.read_csv(D/'08_source_data_tables/Table4_identification.csv').query("sample=='extension_once'").set_index('model');rows=[]
for c in p.columns[2:]:
 fit=sm.GLM(p.y,np.ones((len(p),1)),offset=logit(p[c].clip(1e-6,1-1e-6)),family=sm.families.Binomial()).fit();auc,pos,neg=auc_components(p.y,p[c]);se=np.sqrt(np.var(pos,ddof=1)/len(pos)+np.var(neg,ddof=1)/len(neg));z=t.loc[c];prev=z.ppv_at_prev*(1-z.specificity)/(z.recall*(1-z.ppv_at_prev)+z.ppv_at_prev*(1-z.specificity))
 rows.append({'model':c,'calibration_in_the_large':fit.params.iloc[0],'source_intercept':z.calibration_intercept,'offset_intercept_difference':fit.params.iloc[0]-z.calibration_intercept,'delong_normal_auc_lo':auc-1.959963984540054*se,'delong_normal_auc_hi':auc+1.959963984540054*se,'source_auc_lo':z.auc_lo,'source_auc_hi':z.auc_hi,'auc_lo_difference':auc-1.959963984540054*se-z.auc_lo,'auc_hi_difference':auc+1.959963984540054*se-z.auc_hi,'implicit_prevalence':prev})
with (R/'audit_20261028_metric_definitions.csv').open('x',encoding='utf-8',newline='') as f:pd.DataFrame(rows).to_csv(f,index=False)
d=pd.read_csv(D/'06_analysis_ready/day_level_extension.csv');e=pd.read_csv(D/'05_ambulatory/ema_prompts.csv');e['date']=pd.to_datetime(e.prompt_id.str.split('-').str[1],format='%Y%m%d');dates=e.groupby(['pid','study_day']).date.first().reset_index();d=d.merge(dates,on=['pid','study_day']);wrong=int((d.weekend!=(d.date.dt.dayofweek>=5).astype(int)).sum())
tab=pd.read_csv(R/'audit_20261028_run1_h4_strata.csv');bad=tab[(tab.status=='recomputed')&((tab.recomputed_n!=tab.n_person_days)|(abs(tab.difference)>.00051))]
with (R/'audit_20261028_table5_corrections.csv').open('x',encoding='utf-8',newline='') as f:bad.to_csv(f,index=False)
summary={'calibration_intercept_definition':'source Table4 uses offset calibration-in-the-large; run1 joint intercept is a different estimand, not a data error','implicit_prevalence':float(np.mean([v['implicit_prevalence'] for v in rows])),'prevalence_note':'source predictive values imply 18.1116%, not exactly 18%; source population estimate unavailable','max_delong_lo_difference':float(max(abs(v['auc_lo_difference']) for v in rows)),'max_delong_hi_difference':float(max(abs(v['auc_hi_difference']) for v in rows)),'auc_ci_note':'Original Table4 AUC intervals match normal DeLong intervals. Manuscript says 2000 bootstrap: clarify method. New run1 bootstrap intervals are independently recomputed.','calendar_weekend_mismatch':wrong,'table5_mismatched_rows':len(bad),'table5_note':'All ten weekday/weekend rows have differing participant-day counts; supplied weekend flags match actual EMA dates. Time-of-day four rows cannot be recalculated without window predictions.'}
with (R/'audit_20261028_metric_definitions.json').open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
