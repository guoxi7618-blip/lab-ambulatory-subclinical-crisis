"""Follow-up diagnostics; retains failed/singular first-pass results as audit history."""
from pathlib import Path
import json,warnings,sys
import numpy as np,pandas as pd
import statsmodels.api as sm,statsmodels.formula.api as smf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;D=R/'data';SEED=20261028
def save(name,x):
 with (R/('audit_20261028_refined_'+name)).open('x',encoding='utf-8',newline='') as f:
  if isinstance(x,pd.DataFrame):x.to_csv(f,index=False)
  else:json.dump(x,f,ensure_ascii=False,indent=2,default=lambda v:float(v))
def pack(f):
 ci=f.conf_int();return {'n':int(f.nobs),'converged':bool(getattr(f,'converged',True)),'llf':float(f.llf),'terms':{k:{'b':float(f.params[k]),'p':float(f.pvalues[k]),'lo':float(ci.loc[k,0]),'hi':float(ci.loc[k,1])} for k in f.params.index}}
def main():
 sys.stdout.reconfigure(encoding='utf-8')
 h=pd.read_csv(D/'06_analysis_ready/prompt_level_h2.csv');p=pd.read_csv(D/'06_analysis_ready/person_level_features.csv');h=h.merge(p[['pid','sample']],on='pid')
 out={}
 # Raw HR preserves the between-person intercept; hr_c has already been centered.
 # Both specifications are reported, not selected for p-value or agreement.
 for label,z in [('pooled',h),('core',h[h['sample']=='core'])]:
  for spec,formula in [('minimal','hr_pre ~ stressor * lab_react_z'),('adjusted','hr_pre ~ stressor * lab_react_z + C(activity) + C(time_of_day) + exam_week')]:
   print(label,spec,flush=True)
   with warnings.catch_warnings(record=True) as ww:
    f=smf.mixedlm(formula,z,groups=z.pid,re_formula='1 + stressor').fit(reml=False,method='lbfgs',maxiter=500)
   out[label+'_'+spec]={'formula':formula,'random':'participant intercept + stressor slope','fit':pack(f),'warnings':[str(w.message) for w in ww]}
  f=smf.gee('hr_c ~ stressor * lab_react_z + C(activity) + C(time_of_day) + exam_week',groups='pid',data=z,family=sm.families.Gaussian(),cov_struct=sm.cov_struct.Exchangeable()).fit()
  out[label+'_gee_sensitivity']={'formula':'hr_c ~ stressor * lab_react_z + C(activity) + C(time_of_day) + exam_week','fit':pack(f)}
 save('h2.json',out)
 d=pd.read_csv(D/'06_analysis_ready/day_level_extension.csv');r={}
 for name,z in [('all_scenarios',d),('five_reported_scenarios',d[d.scenario.isin(['exam','academic_routine','career_entrance_exam','interpersonal','no_stressor'])])]:
  for corr in ['independence','exchangeable']:
   f=smf.gee('correct ~ C(scenario) + exam_week + weekend + week',groups='pid',data=z,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Independence() if corr=='independence' else sm.cov_struct.Exchangeable()).fit()
   ts=[i for i,k in enumerate(f.params.index) if k.startswith('C(scenario)')];w=f.wald_test(np.eye(len(f.params))[ts],scalar=True)
   r[name+'_'+corr]={'fit':pack(f),'scenario_test':{'chi2':float(w.statistic),'df':len(ts),'p':float(w.pvalue)}}
 r['missing_scenario_rows']=int(d.scenario.isna().sum());r['scenario_counts']=d.scenario.value_counts(dropna=False).to_dict();save('h4.json',r)
 specs=json.loads((R/'audit_20261028_run1_retrained_parameters.json').read_text());core=p[p['sample']=='core'];ext=p[p['sample']=='extension'];old=pd.read_csv(R/'audit_20261028_run1_retrained_predictions.csv');rows=[]
 for name,s in specs.items():
  m=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('clf',LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,C=s['C'],max_iter=100000,tol=1e-5,random_state=SEED))])
  with warnings.catch_warnings(record=True) as ww:m.fit(core[s['features']],core.y)
  pp=m.predict_proba(ext[s['features']])[:,1];rows.append({'model':name,'C':s['C'],'n_iter':int(m[-1].n_iter_[0]),'max_iter':100000,'converged':int(m[-1].n_iter_[0])<100000,'auc':roc_auc_score(ext.y,pp),'max_abs_prediction_difference_from_run1':np.max(np.abs(pp-old[name])),'warnings':str([str(w.message) for w in ww])})
 save('model_convergence.csv',pd.DataFrame(rows));print('DONE',flush=True)
if __name__=='__main__':main()
