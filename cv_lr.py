import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
sets=sys.argv[1].split(',')
d=pd.concat([pd.read_parquet(f'data/{n}_train.parquet') for n in sets],axis=1); d=d.loc[:,~d.columns.duplicated()]
y=pd.read_csv('data/train_signals.csv').set_index('signal_id').eskalatsiya.loc[d.index]
d=d.drop(columns=['eskalatsiya'],errors='ignore').replace([np.inf,-np.inf],np.nan)
for C in [0.003,0.01,0.03,0.1]:
    p=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),QuantileTransformer(output_distribution='normal',n_quantiles=200),LogisticRegression(C=C,max_iter=3000))
    oof=cross_val_predict(p,d,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba',n_jobs=5)[:,1]
    print(C,'LR OOF %.4f'%roc_auc_score(y,oof))
