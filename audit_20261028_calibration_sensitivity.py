"""New exploratory calibration sensitivity; extension never used for tuning.
Core-only stratified 5-fold GridSearchCV; preprocessing within each fold.
All outputs exclusive-create. Run from any directory.
"""
from pathlib import Path
import argparse,json,time,warnings
import numpy as np,pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

class CalibrationWinsorizer(BaseEstimator,TransformerMixin):
 def __init__(self,columns=None,enabled=True):self.columns=columns;self.enabled=enabled
 def fit(self,X,y=None):
  self.lower_=X[self.columns].quantile(.01);self.upper_=X[self.columns].quantile(.99);return self
 def transform(self,X):
  X=X.copy()
  if self.enabled:X[self.columns]=X[self.columns].clip(self.lower_,self.upper_,axis=1)
  return X

R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--prefix',default='audit_20261028_calibration_sensitivity');ap.add_argument('--bootstrap',type=int,default=2000);a=ap.parse_args()
for suffix in ['.json','_predictions.csv','_cv.csv']:
 if (R/(a.prefix+suffix)).exists():raise FileExistsError(a.prefix+suffix)
f=pd.read_csv(R/'data/06_analysis_ready/person_level_features.csv');core=f[f['sample']=='core'];ext=f[f['sample']=='extension']
lab=[c for c in f if c.startswith('lab_')];wrist=[c for c in f if c.startswith('w_')];cal=[c for c in f if c.startswith('cal_')]
assert len(lab)==9 and len(wrist)==16 and len(cal)==7
seed=20261028;folds=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(core,core.y));results=[];preds=ext[['pid','y']].copy();cvrecords=[]
for block in ['physiology','SDS_SAS_physiology']:
 for scheme in ['original','without_calibration','winsorized_calibration']:
  cols=lab+wrist+([] if scheme=='without_calibration' else cal)+([] if block=='physiology' else ['sds_standard','sas_standard'])
  pipe=Pipeline([('winsor',CalibrationWinsorizer(cal if scheme!='without_calibration' else [],scheme=='winsorized_calibration')),('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,max_iter=100000,tol=1e-5,random_state=seed))])
  start=time.time();print('START',block,scheme,len(cols),flush=True)
  gs=GridSearchCV(pipe,{'model__C':np.logspace(-3,3,13)},scoring='roc_auc',cv=folds,n_jobs=1,refit=True,return_train_score=False,error_score='raise')
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning);gs.fit(core[cols],core.y)
  prob=gs.predict_proba(ext[cols])[:,1];key=block+'__'+scheme;preds[key]=prob
  result={'block':block,'scheme':scheme,'n_features':len(cols),'features':cols,'C':float(gs.best_params_['model__C']),'core_cv_auc':float(gs.best_score_),'extension_auc':float(roc_auc_score(ext.y,prob)),'final_n_iter':gs.best_estimator_.named_steps['model'].n_iter_.tolist(),'final_converged':bool(gs.best_estimator_.named_steps['model'].n_iter_.max()<100000),'grid_convergence_warning_count':sum(issubclass(w.category,ConvergenceWarning) for w in caught),'seconds':time.time()-start}
  results.append(result)
  for c,mean,sd in zip(gs.cv_results_['param_model__C'],gs.cv_results_['mean_test_score'],gs.cv_results_['std_test_score']):cvrecords.append({'block':block,'scheme':scheme,'C':float(c),'mean_cv_auc':float(mean),'sd_cv_auc':float(sd)})
  print('DONE',json.dumps({k:v for k,v in result.items() if k!='features'}),flush=True)
comparison=[];y=ext.y.to_numpy();rng=np.random.default_rng(seed);ids0=np.where(y==0)[0];ids1=np.where(y==1)[0];resamples=[np.r_[rng.choice(ids0,len(ids0),replace=True),rng.choice(ids1,len(ids1),replace=True)] for _ in range(a.bootstrap)]
for r in results:
 key=r['block']+'__'+r['scheme'];v=preds[key].to_numpy();boots=np.array([roc_auc_score(y[idx],v[idx]) for idx in resamples]);r['extension_auc_bootstrap_ci']=np.quantile(boots,[.025,.975]).tolist()
for block in ['physiology','SDS_SAS_physiology']:
 base=preds[block+'__original'].to_numpy()
 for scheme in ['without_calibration','winsorized_calibration']:
  v=preds[block+'__'+scheme].to_numpy();dif=np.array([roc_auc_score(y[i],v[i])-roc_auc_score(y[i],base[i]) for i in resamples]);comparison.append({'block':block,'contrast':scheme+' minus original','delta_auc':float(roc_auc_score(y,v)-roc_auc_score(y,base)),'paired_stratified_bootstrap_ci':np.quantile(dif,[.025,.975]).tolist(),'bootstrap_n':a.bootstrap})
report={'status':'NEW_EXPLORATORY_SENSITIVITY_NOT_ORIGINAL_REPRODUCTION','seed':seed,'core_n':len(core),'extension_n':len(ext),'protocol':'5 participant-stratified core-only folds; medians, standardization and optional 1/99 percentiles estimated separately in each training fold. C grid 10^-3 to 10^3, 13 values; l1_ratio=.5; no extension tuning. Final refit on all core. Bootstrap uses paired stratified extension participant resamples; not a refit uncertainty estimate.','results':results,'comparisons':comparison}
with (R/(a.prefix+'.json')).open('x',encoding='utf-8') as h:json.dump(report,h,indent=2,ensure_ascii=False)
with (R/(a.prefix+'_predictions.csv')).open('x',encoding='utf-8',newline='') as h:preds.to_csv(h,index=False)
with (R/(a.prefix+'_cv.csv')).open('x',encoding='utf-8',newline='') as h:pd.DataFrame(cvrecords).to_csv(h,index=False)
print('COMPLETE',json.dumps(comparison),flush=True)
