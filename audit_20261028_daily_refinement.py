"""Aligned daily reconstruction using physiology-valid prompts, as empirically identified in supplied table."""
import sys,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from audit_20261028_models import boot_aucs,fit_record,pipeline,threshold,metrics
from sklearn.model_selection import GroupKFold,cross_val_predict
R=Path(__file__).resolve().parent;D=R/'data'
def save(n,x):
 with (R/('audit_20261028_daily_refined_'+n)).open('x',encoding='utf-8',newline='') as f:
  if isinstance(x,pd.DataFrame):x.to_csv(f,index=False)
  else:json.dump(x,f,ensure_ascii=False,indent=2,default=lambda x:float(x))
def main():
 sys.stdout.reconfigure(encoding='utf-8')
 h=pd.read_csv(D/'06_analysis_ready/prompt_level_h2.csv');p=pd.read_csv(D/'06_analysis_ready/person_level_features.csv');w=pd.read_csv(D/'05_ambulatory/wrist_daily_features.csv')
 h['pa']=h[['pa_happy','pa_relaxed','pa_energetic']].mean(axis=1);h['na']=h[['na_down','na_tense','na_irritated']].mean(axis=1)
 a=h.groupby(['pid','study_day']).agg(d_stress=('stress','mean'),d_pa=('pa','mean'),d_na=('na','mean'),d_any=('stressor','mean'),exam_week=('exam_week','first')).reset_index()
 d=w.merge(p[['pid','sample','y','lab_hr_base','lab_lnrmssd_base','lab_hr_react','lab_scl_react','eeg_aper_mean']],on='pid').merge(a,on=['pid','study_day'],how='left');d=d[d.valid_day==1].copy();d['ln_noct']=np.log(d.nocturnal_rmssd_ms)
 extgiven=pd.read_csv(D/'06_analysis_ready/day_level_extension.csv');check=d[d['sample']=='extension'].merge(extgiven[['pid','study_day','d_stress','d_pa','d_na','d_any']],on=['pid','study_day'],suffixes=('_new','_old'))
 save('definitions.json',{'scope':'daily EMA averages use only prompt_level_h2 (physiology-valid) rows, not all completed EMA','n_extension_rows':len(check),'comparisons':{c:{'max_difference':float((check[c+'_new']-check[c+'_old']).abs().max()),'missing_mismatch':int((check[c+'_new'].isna()!=check[c+'_old'].isna()).sum())} for c in ['d_stress','d_pa','d_na','d_any']}})
 med=d.groupby('pid').d_stress.transform('median');d['highstress']=np.where(d.d_stress.notna(),(d.d_stress>med).astype(float),np.nan)
 d['cal_hr']=d.resting_hr_bpm-d.lab_hr_base;d['cal_noct']=d.ln_noct-d.lab_lnrmssd_base
 first=d[d.study_day<=7].groupby('pid')[['resting_hr_bpm','ln_noct']].agg(['mean','std']);first.columns=['first_'+a+'_'+b for a,b in first.columns];d=d.join(first,on='pid');d['first_hr']=d.resting_hr_bpm-d.first_resting_hr_bpm_mean;d['first_noct']=d.ln_noct-d.first_ln_noct_mean;d['first_hr_z']=d.first_hr/d.first_resting_hr_bpm_std;d['first_noct_z']=d.first_noct/d.first_ln_noct_std
 c=d[d.study_day>=8].dropna(subset=['highstress','first_hr','first_noct']);tr=c[c['sample']=='core'];te=c[c['sample']=='extension'];pred=te[['pid','study_day','highstress']].copy();rows=[]
 for name,features in {'raw':['resting_hr_bpm','ln_noct'],'lab_anchored':['cal_hr','cal_noct'],'first_week_centered':['first_hr','first_noct'],'first_week_z':['first_hr_z','first_noct_z']}.items():
  m=Pipeline([('impute',SimpleImputer()),('scale',StandardScaler()),('clf',LogisticRegression(C=1,max_iter=5000,random_state=20261028))]);m.fit(tr[features],tr.highstress);pred[name]=m.predict_proba(te[features])[:,1];within=[roc_auc_score(v.highstress,v[name]) for _,v in pred.groupby('pid') if v.highstress.nunique()==2];rows.append({'method':name,'n_core':len(tr),'n_extension':len(te),'auc':roc_auc_score(te.highstress,pred[name]),'within_person_auc':float(np.mean(within))})
 b=boot_aucs(te.highstress,pred.iloc[:,3:].values,te.pid.values)
 for i,r in enumerate(rows):r['lo'],r['hi']=np.quantile(b[:,i],[.025,.975]);r['delta_vs_raw']=r['auc']-rows[0]['auc'];r['delta_lo'],r['delta_hi']=np.quantile(b[:,i]-b[:,0],[.025,.975])
 save('within_person.csv',pd.DataFrame(rows));save('within_person_predictions.csv',pred)
 print('daily calibration done',flush=True)
 # Random intercept associations retain within-person day structure. This is not causal mediation.
 z=d.dropna(subset=['d_pa','exam_week']);out={}
 for name,form in {'total':'resting_hr_bpm ~ exam_week','a':'d_pa ~ exam_week','direct':'resting_hr_bpm ~ exam_week + d_pa'}.items():
  with warnings.catch_warnings(record=True) as ww:f=smf.mixedlm(form,z,groups=z.pid).fit(reml=False,method='lbfgs',maxiter=500)
  out[name]={'formula':form,'fit':fit_record(f),'warnings':[str(x.message) for x in ww]}
 out['indirect_product']=out['a']['fit']['terms']['exam_week']['b']*out['direct']['fit']['terms']['d_pa']['b'];out['scope']='observational random-intercept association decomposition, not causal effect'
 save('mediation.json',out)
 # Participant fixed-effects alternative and clustered participant bootstrap of product a*b.
 fields=['resting_hr_bpm','exam_week','d_pa'];dm=z[fields]-z.groupby('pid')[fields].transform('mean');dm['pid']=z.pid.values;pieces=[]
 for pid,v in dm.groupby('pid'):
  x=v.exam_week.to_numpy();m=v.d_pa.to_numpy();y=v.resting_hr_bpm.to_numpy();X=np.column_stack([x,m]);pieces.append((x@x,x@m,X.T@X,X.T@y))
 rng=np.random.default_rng(20261028);effects=[]
 for _ in range(2000):
  ix=rng.integers(0,len(pieces),len(pieces));xx=sum(pieces[i][0] for i in ix);xm=sum(pieces[i][1] for i in ix);XX=sum((pieces[i][2] for i in ix),start=np.zeros((2,2)));Xy=sum((pieces[i][3] for i in ix),start=np.zeros(2));effects.append((xm/xx)*np.linalg.solve(XX,Xy)[1])
 xx=sum(v[0] for v in pieces);xm=sum(v[1] for v in pieces);XX=sum((v[2] for v in pieces),start=np.zeros((2,2)));Xy=sum((v[3] for v in pieces),start=np.zeros(2));coef=np.linalg.solve(XX,Xy)
 save('mediation_FE_sensitivity.json',{'n_rows':len(z),'n_persons':z.pid.nunique(),'a':xm/xx,'b':coef[1],'direct':coef[0],'indirect':xm/xx*coef[1],'indirect_ci':np.quantile(effects,[.025,.975]).tolist(),'bootstrap':'2000 participant clusters, within-person demeaned linear coefficients; not causal mediation'})
 # Available full daily rows, including missing EMA handled inside training folds.
 features=['resting_hr_bpm','ln_noct','steps','sleep_duration_h','sleep_midpoint_min','scr_rate_rest_per_5min','mean_skin_temp_c','d_stress','d_pa','d_na','d_any','lab_lnrmssd_base','lab_hr_base','lab_hr_react','lab_scl_react','eeg_aper_mean']
 tr=d[d['sample']=='core'];te=d[d['sample']=='extension'];m=pipeline(C=.1);oo=cross_val_predict(m,tr[features],tr.y,groups=tr.pid,cv=GroupKFold(5),method='predict_proba',n_jobs=2)[:,1];cut=threshold(tr.y,oo);m.fit(tr[features],tr.y);pp=m.predict_proba(te[features])[:,1]
 save('day_retrained_metrics.json',{'features':features,'C':.1,'n_core_days':len(tr),'scope':'Independent participant-grouped reconstruction; no original daily estimator supplied',**metrics(te.y,pp,cut)});save('day_retrained_predictions.csv',te[['pid','study_day','y']].assign(p=pp,pred=(pp>=cut).astype(int)))
 print('DONE',flush=True)
if __name__=='__main__':main()
