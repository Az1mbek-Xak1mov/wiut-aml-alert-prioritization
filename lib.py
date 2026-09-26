import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def load(sets, split='train'):
    d=pd.concat([pd.read_parquet(f'data/{n}_{split}.parquet') for n in sets],axis=1)
    d=d.loc[:,~d.columns.duplicated()].drop(columns=['eskalatsiya'],errors='ignore')
    return d.replace([np.inf,-np.inf],np.nan).astype('float32')

def target(idx):
    return pd.read_csv('data/train_signals.csv').set_index('signal_id').eskalatsiya.loc[idx]

P=dict(n_estimators=4000,learning_rate=0.01,num_leaves=15,min_child_samples=80,subsample=0.8,subsample_freq=1,colsample_bytree=0.3,reg_lambda=10,verbose=-1,n_jobs=12)

def cv(d,y,seeds=(0,1,2),params=None,ret_imp=False):
    params={**P,**(params or {})}
    oofs=[];imp=pd.Series(0.,index=d.columns);its=[]
    for sd in seeds:
        oof=np.zeros(len(d))
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=sd).split(d,y):
            m=lgb.LGBMClassifier(**params,random_state=sd)
            m.fit(d.iloc[tr],y.iloc[tr],eval_set=[(d.iloc[va],y.iloc[va])],eval_metric='auc',callbacks=[lgb.early_stopping(300,verbose=False)])
            oof[va]=m.predict_proba(d.iloc[va])[:,1]; its.append(m.best_iteration_)
            imp+=pd.Series(m.booster_.feature_importance('gain'),index=d.columns)
        oofs.append(oof)
    aucs=[roc_auc_score(y,o) for o in oofs]
    res=dict(auc=np.mean(aucs),std=np.std(aucs),it=int(np.mean(its)),oof=np.mean(oofs,0))
    if ret_imp: res['imp']=imp.sort_values(ascending=False)
    return res
