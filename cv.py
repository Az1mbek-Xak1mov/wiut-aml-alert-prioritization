import pandas as pd, numpy as np, lightgbm as lgb, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sets=sys.argv[1].split(',') if len(sys.argv)>1 else ['feat','feat2']
d=pd.concat([pd.read_parquet(f'data/{n}_train.parquet') for n in sets],axis=1)
d=d.loc[:,~d.columns.duplicated()]
y=d.pop('eskalatsiya') if 'eskalatsiya' in d else pd.read_csv('data/train_signals.csv').set_index('signal_id').eskalatsiya.loc[d.index]
cols=list(d.columns); print('n features',len(cols))
oof=np.zeros(len(d)); imp=pd.Series(0.,index=cols)
for tr,va in StratifiedKFold(5,shuffle=True,random_state=0).split(d,y):
    m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.01,num_leaves=15,min_child_samples=80,subsample=0.8,subsample_freq=1,colsample_bytree=0.3,reg_lambda=10,verbose=-1)
    m.fit(d.iloc[tr],y.iloc[tr],eval_set=[(d.iloc[va],y.iloc[va])],eval_metric='auc',callbacks=[lgb.early_stopping(300,verbose=False)])
    oof[va]=m.predict_proba(d.iloc[va])[:,1]; imp+=pd.Series(m.booster_.feature_importance('gain'),index=cols)
    print('fold %.4f it %d'%(roc_auc_score(y.iloc[va],oof[va]),m.best_iteration_))
print('OOF AUC %.4f'%roc_auc_score(y,oof))
imp.sort_values(ascending=False).to_csv('data/imp.csv')
print(imp.sort_values(ascending=False).head(30))
