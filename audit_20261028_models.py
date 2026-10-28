"""Reproducible H2-H4 audit. No raw inputs are modified; outputs refuse overwrite.
Run: python audit_20261028_models.py --prefix audit_20261028_run1
Reconstructed models are NOT the unavailable frozen MSAF training pipeline.
"""
from pathlib import Path
import argparse, json, sys, hashlib, platform, warnings, time
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.special import expit, logit
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
SEED=20261028
BOOT=2000
ap=argparse.ArgumentParser(); ap.add_argument('--prefix',default='audit_20261028_run1'); args=ap.parse_args()
PREFIX=ROOT/args.prefix
def save(name,obj):
    p=Path(str(PREFIX)+'_'+name)
    with p.open('x',encoding='utf-8',newline='') as h:
        if isinstance(obj,pd.DataFrame): obj.to_csv(h,index=False)
        else: json.dump(obj,h,ensure_ascii=False,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x))
def read(s):return pd.read_csv(DATA/s)
def say(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def auc_components(y,p):
    pos=np.asarray(p)[np.asarray(y)==1]; neg=np.asarray(p)[np.asarray(y)==0]
    k=(pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:])
    return k.mean(),k.mean(axis=1),k.mean(axis=0)
def delong(y,a,b):
    aa,ap,an=auc_components(y,a); bb,bp,bn=auc_components(y,b)
    var=np.var(ap-bp,ddof=1)/len(ap)+np.var(an-bn,ddof=1)/len(an)
    se=np.sqrt(max(var,0)); z=(aa-bb)/se if se else np.nan
    return {'delta':aa-bb,'se':se,'z':z,'p':2*stats.norm.sf(abs(z)),'lo':aa-bb-1.96*se,'hi':aa-bb+1.96*se}
def boot_aucs(y,ps,groups=None):
    """Paired, ordinary (unstratified) participant bootstrap; entire person clusters retained."""
    y=np.asarray(y); ps=np.asarray(ps)
    if ps.ndim==1:ps=ps[:,None]
    rng=np.random.default_rng(SEED); rows=[]
    if groups is not None:
        _,gi=np.unique(groups,return_inverse=True); ng=gi.max()+1
    for _ in range(BOOT):
        if groups is None:
            ix=rng.integers(0,len(y),len(y)); yy=y[ix]; pp=ps[ix]; w=None
        else:
            w=np.bincount(rng.integers(0,ng,ng),minlength=ng)[gi]; yy=y;pp=ps
        if len(np.unique(yy if w is None else yy[w>0]))<2:continue
        rows.append([roc_auc_score(yy,pp[:,j],sample_weight=w) for j in range(ps.shape[1])])
    return np.asarray(rows)
def metrics(y,p,th):
    y=np.asarray(y);p=np.asarray(p); pred=(p>=th).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); sens=tp/(tp+fn); spec=tn/(tn+fp)
    fit=sm.GLM(y,sm.add_constant(logit(np.clip(p,1e-6,1-1e-6))),family=sm.families.Binomial()).fit()
    prev=.18
    return dict(n=len(y),auc=roc_auc_score(y,p),threshold=th,tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn),accuracy=(tp+tn)/len(y),recall=sens,specificity=spec,f1=f1_score(y,pred),balanced_accuracy=(sens+spec)/2,ppv=tp/(tp+fp) if tp+fp else np.nan,npv=tn/(tn+fn) if tn+fn else np.nan,brier=brier_score_loss(y,p),calibration_intercept=fit.params[0],calibration_slope=fit.params[1],ppv_at_prev=prev*sens/(prev*sens+(1-prev)*(1-spec)),npv_at_prev=(1-prev)*spec/((1-prev)*spec+prev*(1-sens)))
def pipeline(C=1):
    return Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('clf',LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,C=C,max_iter=15000,tol=1e-5,random_state=SEED))])
def grid():
    return GridSearchCV(pipeline(),{'clf__C':np.logspace(-3,3,13)},cv=StratifiedKFold(5,shuffle=True,random_state=SEED),scoring='roc_auc',n_jobs=2,refit=True)
def threshold(y,p):
    fpr,tpr,ths=roc_curve(y,p); ok=np.where((tpr>=.8)&np.isfinite(ths))[0]; j=ok[np.argmin(fpr[ok])];return float(ths[j])
def fit_record(fit):
    ci=fit.conf_int()
    return {'converged':getattr(fit,'converged',None),'n':int(fit.nobs),'llf':float(fit.llf),'terms':{k:{'b':float(fit.params[k]),'p':float(fit.pvalues[k]),'lo':float(ci.loc[k,0]),'hi':float(ci.loc[k,1])} for k in fit.params.index}}

def main():
    say('Manifest and provided-prediction audit')
    save('manifest.json',{'seed':SEED,'bootstrap':BOOT,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'statsmodels':statsmodels.__version__,'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*.csv')},'scope':'Prediction recomputation + independently reconstructed models; NOT MSAF retraining','grid_C':list(np.logspace(-3,3,13)),'cv':'participant-level shuffled 5-fold; inner training-fold preprocessing; nested OOF for thresholds; core-only selection'})
    pred=read('07_model/extension_predicted_probabilities.csv'); th=read('07_model/decision_thresholds.csv').set_index('model').decision_threshold
    cols=list(pred.columns[2:]); y=pred.y.to_numpy(); ps=pred[cols].to_numpy()
    boots=boot_aucs(y,ps); out=[]; comparisons=[]
    for i,c in enumerate(cols):
        r={'model':c,**metrics(y,pred[c],th[c])};r.update(dict(zip(['auc_lo','auc_hi'],np.quantile(boots[:,i],[.025,.975]))));out.append(r)
        if c!='MSAF (full)':
            r={'a':'MSAF (full)','b':c,**delong(y,pred['MSAF (full)'],pred[c])};r['bootstrap_lo'],r['bootstrap_hi']=np.quantile(boots[:,0]-boots[:,i],[.025,.975]);comparisons.append(r)
    for a,b in [('SDS + SAS + MSAF blocks (incremental)','SDS + SAS'),('MSAF without EMA','SDS + SAS')]: comparisons.append({'a':a,'b':b,**delong(y,pred[a],pred[b])})
    cmp=pd.DataFrame(comparisons);cmp['p_BH_exploratory_all_tests']=multipletests(cmp.p,method='fdr_bh')[1]
    save('provided_prediction_metrics.csv',pd.DataFrame(out));save('provided_prediction_comparisons.csv',cmp)
    source=read('08_source_data_tables/Table4_identification.csv').query("sample=='extension_once'").set_index('model')
    diffs=[]
    for r in out:
        for key in set(r)&set(source.columns)-{'auc_lo','auc_hi','threshold','n'}:
            if isinstance(r[key],(int,float,np.number)):
                diffs.append({'model':r['model'],'metric':key,'recomputed':r[key],'provided':source.loc[r['model'],key],'difference':r[key]-source.loc[r['model'],key]})
    save('table4_differences.csv',pd.DataFrame(diffs))
    say('Training independent person-level elastic-net models; extension held out')
    p=read('06_analysis_ready/person_level_features.csv');core=p.query("sample=='core'").copy();ext=p.query("sample=='extension'").copy()
    phys=[c for c in p if c.startswith(('lab_','w_','cal_'))];ema=[c for c in p if c.startswith('ema_') and c!='ema_compliance'];q=['sds_standard','sas_standard'];eeg=[c for c in p if c.startswith('eeg_')]
    sets={'SDS/SAS':q,'Physiology':phys,'SDS/SAS + physiology':q+phys,'All blocks without compliance':q+phys+ema,'All blocks with compliance':q+phys+ema+['ema_compliance'],'Compliance alone':['ema_compliance'],'All blocks plus EEG':q+phys+ema+['ema_compliance']+eeg,'All blocks without passive SI':q+phys+[c for c in ema if c!='ema_passive_si_any']+['ema_compliance']}
    model_rows=[]; newpred=ext[['pid','y']].copy();fit_details={};oofall=core[['pid','y']].copy()
    for name,features in sets.items():
        model=grid();model.fit(core[features],core.y)
        # Nested OOF ensures threshold never uses extension and penalty selection never sees outer labels.
        oof=cross_val_predict(grid(),core[features],core.y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED+1),method='predict_proba',n_jobs=1)[:,1]
        pp=model.predict_proba(ext[features])[:,1]; cut=threshold(core.y,oof); newpred[name]=pp;oofall[name]=oof
        row={'model':name,'n_features':len(features),'best_C':model.best_params_['clf__C'],'nested_oof_auc':roc_auc_score(core.y,oof),**metrics(ext.y,pp,cut)};model_rows.append(row)
        fit_details[name]={'features':features,'C':model.best_params_['clf__C'],'cv_mean_auc':model.cv_results_['mean_test_score'].tolist(),'coef':model.best_estimator_[-1].coef_.tolist(),'intercept':model.best_estimator_[-1].intercept_.tolist(),'imputer_statistics':model.best_estimator_[0].statistics_.tolist(),'scaler_mean':model.best_estimator_[1].mean_.tolist(),'scaler_scale':model.best_estimator_[1].scale_.tolist()}
        say(name+': AUC='+str(row['auc']))
    bs=boot_aucs(ext.y,newpred.iloc[:,2:].values)
    for i,r in enumerate(model_rows):r['auc_lo'],r['auc_hi']=np.quantile(bs[:,i],[.025,.975])
    save('retrained_metrics.csv',pd.DataFrame(model_rows));save('retrained_predictions.csv',newpred);save('nested_oof_predictions.csv',oofall);save('retrained_parameters.json',fit_details)
    nc=[]
    for a,b in [('Physiology','SDS/SAS'),('SDS/SAS + physiology','SDS/SAS'),('All blocks without compliance','SDS/SAS'),('All blocks with compliance','All blocks without compliance'),('All blocks plus EEG','All blocks with compliance')]:nc.append({'a':a,'b':b,**delong(ext.y,newpred[a],newpred[b])})
    save('retrained_comparisons.csv',pd.DataFrame(nc))
    # Original predictions: subgroup estimates are exploratory, no model selection.
    chars=read('01_participants/participant_characteristics.csv');ref=read('02_reference_standard/reference_standard.csv')
    sub=pred.merge(chars[['pid','sex','graduating','severity']],on='pid',validate='one_to_one').merge(ref[['pid','crisis_passive_si','crisis_self_harm','crisis_acute_stress','crisis_functional_impairment','crisis_anhedonia_withdrawal']],on='pid')
    sub['signals']=sub.filter(regex='^crisis_').sum(axis=1); sr=[]
    masks={'female':sub.sex=='female','male':sub.sex=='male','graduating':sub.graduating==1,'not_graduating':sub.graduating==0,'passive_SI_vs_controls':(sub.y==0)|(sub.crisis_passive_si==1),'no_passive_SI_vs_controls':(sub.y==0)|(sub.crisis_passive_si==0),'single_signal_vs_controls':(sub.y==0)|(sub.signals==1),'multiple_signal_vs_controls':(sub.y==0)|(sub.signals>1)}
    for label,mask in masks.items():
        z=sub[mask];bb=boot_aucs(z.y,z['MSAF (full)']);lo,hi=np.quantile(bb[:,0],[.025,.975]);sr.append({'subgroup':label,'n':len(z),'cases':int(z.y.sum()),'auc':roc_auc_score(z.y,z['MSAF (full)']),'lo':lo,'hi':hi})
    save('subgroups.csv',pd.DataFrame(sr));save('severity_correlations.json',{'all':stats.spearmanr(sub.severity,sub['MSAF (full)'])._asdict(),'cases':stats.spearmanr(sub.loc[sub.y==1,'severity'],sub.loc[sub.y==1,'MSAF (full)'])._asdict()})
    say('H2 multilevel reconstruction')
    h=read('06_analysis_ready/prompt_level_h2.csv'); h=h.merge(p[['pid','sample','lab_scl_react']],on='pid',validate='many_to_one')
    result={}
    for label,z in [('pooled',h),('core',h.query("sample=='core'"))]:
        form='hr_c ~ stressor * lab_react_z + C(activity) + C(time_of_day) + exam_week'
        with warnings.catch_warnings(record=True) as ws:
            fit=smf.mixedlm(form,z,groups=z.pid).fit(reml=False,method='lbfgs',maxiter=200)
        result[label]={'formula':form,'random':'intercept','fit':fit_record(fit),'warnings':[str(w.message) for w in ws]}
    per=h.groupby(['pid','stressor'])[['hr_pre','scr_pre']].mean().unstack('stressor');diff=per.xs(1,level=1,axis=1)-per.xs(0,level=1,axis=1)
    per=diff.join(p.set_index('pid')[['lab_hr_react','lab_scl_react']]);result['person_level_correlations']={a:stats.pearsonr(per[a].dropna(),per.loc[per[a].notna(),b])._asdict() for a,b in [('lab_hr_react','hr_pre'),('lab_scl_react','scr_pre')]}
    save('h2.json',result)
    say('H4 existing day predictions, person-clustered inference')
    d=read('06_analysis_ready/day_level_extension.csv'); gee=smf.gee('correct ~ C(scenario) + exam_week + weekend + week','pid',d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
    lp=logit(d.p.clip(1e-6,1-1e-6)); mu=lp.mean();means=lp.groupby(d.pid).transform('mean'); between=((means-mu)**2).sum()/((lp-mu)**2).sum()
    h4={'formula':'correct ~ C(scenario) + exam_week + weekend + week','gee':fit_record(gee),'between_person_logit_variance_share':between,'constant_class_fraction':float(d.groupby('pid').pred.nunique().eq(1).mean()),'overall_balanced_accuracy':float(((d.query('y==1').pred==1).mean()+(d.query('y==0').pred==0).mean())/2),'person_days':len(d),'persons':d.pid.nunique()}
    scenario_terms=[i for i,k in enumerate(gee.params.index) if k.startswith('C(scenario)')];R=np.eye(len(gee.params))[scenario_terms];w=gee.wald_test(R,scalar=True);h4['scenario_omnibus']={'chi2':float(w.statistic),'df':len(scenario_terms),'p':float(w.pvalue)}
    save('h4.json',h4)
    # Cluster bootstrap: 198 persons, preserving all days and both classification errors.
    old=read('08_source_data_tables/Table5_stability.csv'); rs=[]
    for _,r in old.iterrows():
        if r.scenario=='all':rs.append({**r.to_dict(),'recomputed_ba':np.nan,'status':'unavailable: no within-day predictions'});continue
        z=d[d.scenario==r.scenario];s=r.stratum
        mask=(z.weekend==int(s=='weekend')) if s in ['weekday','weekend'] else ((z.exam_week==int(s=='exam_week')) if s in ['exam_week','non_exam_week'] else z.week==int(s[-1]))
        z=z[mask];ids=np.sort(d.pid.unique());arr=[]
        for pid in ids:
            v=z[z.pid==pid];arr.append([((v.y==1)&(v.pred==1)).sum(),(v.y==1).sum(),((v.y==0)&(v.pred==0)).sum(),(v.y==0).sum()])
        arr=np.asarray(arr);rng=np.random.default_rng(SEED);idx=rng.integers(0,len(ids),(BOOT,len(ids)));sums=arr[idx].sum(axis=1);b=.5*(sums[:,0]/sums[:,1]+sums[:,2]/sums[:,3]);ba=.5*((z.loc[z.y==1,'pred']==1).mean()+(z.loc[z.y==0,'pred']==0).mean())
        rs.append({**r.to_dict(),'recomputed_ba':ba,'recomputed_lo':np.nanquantile(b,.025),'recomputed_hi':np.nanquantile(b,.975),'recomputed_n':len(z),'difference':ba-r.balanced_accuracy,'status':'recomputed'})
    save('h4_strata.csv',pd.DataFrame(rs))
    say('Within-person calibration reconstruction and exam-week associations')
    wr=read('05_ambulatory/wrist_daily_features.csv');em=read('05_ambulatory/ema_prompts.csv');em=em[em.prompt_type.str.startswith('random')&(em.status=='completed')].copy()
    em['pa']=em[['pa_happy','pa_relaxed','pa_energetic']].mean(axis=1);em['na']=em[['na_down','na_tense','na_irritated']].mean(axis=1)
    em['date']=pd.to_datetime(em.prompt_id.str.split('-').str[1],format='%Y%m%d');daily=em.groupby(['pid','study_day']).agg(d_stress=('stress','mean'),d_pa=('pa','mean'),d_na=('na','mean'),d_any=('stressor_since_last','mean'),date=('date','first')).reset_index()
    daily=wr.merge(daily,on=['pid','study_day']).merge(p[['pid','sample','y','lab_lnrmssd_base','lab_hr_base','lab_hr_react','lab_scl_react','eeg_aper_mean']],on='pid');daily=daily[daily.valid_day==1].copy();daily['ln_noct']=np.log(daily.nocturnal_rmssd_ms)
    ex=h.groupby(['pid','study_day']).exam_week.first().rename('exam_week').reset_index();daily=daily.merge(ex,on=['pid','study_day'],how='left');daily['weekend']=(daily.date.dt.dayofweek>=5).astype(int)
    daily['cal_hr']=daily.resting_hr_bpm-daily.lab_hr_base;daily['cal_noct']=daily.ln_noct-daily.lab_lnrmssd_base
    first=daily.query('study_day<=7').groupby('pid')[['resting_hr_bpm','ln_noct']].mean().add_prefix('first_');daily=daily.join(first,on='pid');daily['first_hr']=daily.resting_hr_bpm-daily.first_resting_hr_bpm;daily['first_noct']=daily.ln_noct-daily.first_ln_noct
    daily['highstress']=(daily.d_stress>daily.groupby('pid').d_stress.transform('median')).astype(int)
    cal=daily.query('study_day>=8').dropna(subset=['first_hr','first_noct']);calcore=cal.query("sample=='core'");calext=cal.query("sample=='extension'");calpred=calext[['pid','study_day','highstress']].copy();calrows=[]
    for name,features in {'raw':['resting_hr_bpm','ln_noct'],'lab_anchored':['cal_hr','cal_noct'],'first_week':['first_hr','first_noct']}.items():
        m=Pipeline([('impute',SimpleImputer()),('scale',StandardScaler()),('clf',LogisticRegression(C=1,max_iter=5000,random_state=SEED))]);m.fit(calcore[features],calcore.highstress);pp=m.predict_proba(calext[features])[:,1];calpred[name]=pp
        z=calext[['pid','highstress']].assign(prob=pp);within=[roc_auc_score(v.highstress,v.prob) for _,v in z.groupby('pid') if v.highstress.nunique()==2];calrows.append({'method':name,'n_core':len(calcore),'n_extension':len(calext),'auc':roc_auc_score(calext.highstress,pp),'mean_within_person_auc':np.mean(within),'status':'reconstructed: median defined on valid EMA days; first week on valid days; original choices unspecified'})
    bb=boot_aucs(calext.highstress,calpred.iloc[:,3:].values,calext.pid.values)
    for i,r in enumerate(calrows):r['lo'],r['hi']=np.quantile(bb[:,i],[.025,.975]);r['delta_vs_raw']=r['auc']-calrows[0]['auc'];r['delta_lo'],r['delta_hi']=np.quantile(bb[:,i]-bb[:,0],[.025,.975])
    save('within_person.csv',pd.DataFrame(calrows));save('within_person_predictions.csv',calpred)
    # Association decomposition, not causal mediation; cluster robust standard errors.
    zd=daily.dropna(subset=['exam_week','d_pa']);exam={}
    for name,formula in {'total':'resting_hr_bpm ~ exam_week','a':'d_pa ~ exam_week','direct':'resting_hr_bpm ~ exam_week + d_pa'}.items():
        f=smf.ols(formula,zd).fit(cov_type='cluster',cov_kwds={'groups':zd.pid});exam[name]=fit_record(f)
    exam['indirect_product']=exam['a']['terms']['exam_week']['b']*exam['direct']['terms']['d_pa']['b'];exam['scope']='Observed days with exam labels; observational decomposition, not causal mediation; original specification unavailable'
    save('exam_associations.json',exam)
    say('Independent day-level model using participant-grouped development folds')
    features=['resting_hr_bpm','ln_noct','steps','sleep_duration_h','sleep_midpoint_min','scr_rate_rest_per_5min','mean_skin_temp_c','d_stress','d_pa','d_na','d_any','lab_lnrmssd_base','lab_hr_base','lab_hr_react','lab_scl_react','eeg_aper_mean']
    dc=daily.query("sample=='core'");de=daily.query("sample=='extension'")
    # Fixed penalty to avoid expensive unreported search, participant-safe OOF threshold.
    m=pipeline(C=.1);oof=cross_val_predict(m,dc[features],dc.y,groups=dc.pid,cv=GroupKFold(5),method='predict_proba',n_jobs=2)[:,1];cut=threshold(dc.y,oof);m.fit(dc[features],dc.y);pp=m.predict_proba(de[features])[:,1]
    save('day_retrained_metrics.json',{'features':features,'C':.1,'scope':'independent reconstruction, not original frozen daily model','n_core_days':len(dc),'n_extension_days':len(de),**metrics(de.y,pp,cut)})
    save('day_retrained_predictions.csv',de[['pid','study_day','y']].assign(p=pp,pred=(pp>=cut).astype(int)))
    save('completion.json',{'status':'completed_available_H2_H3_H4','limitations':['No original MSAF weights/code/5-minute windows; no MSAF retraining','No core original OOF predictions; original CV performance cannot be independently recomputed','No hourly predictions; time-of-day H4 cannot be recomputed','Exact original hyperparameter grid, day model, mediation and within-person choices absent; reconstructions explicitly labeled','Existing analysis-ready features require data-integrity audit; inputs not assumed correct'],'finished':time.strftime('%Y-%m-%d %H:%M:%S')})
    say('COMPLETE')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
