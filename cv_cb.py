from lib import *
from catboost import CatBoostClassifier
import sys
imp=pd.read_csv('data/imp_all.csv',index_col=0).iloc[:,0]
k=int(sys.argv[1]); d=load(['feat','feat3','feat4'])[imp.index[:k]]; y=target(d.index)
oofs=[]
for sd in (0,1,2):
    oof=np.zeros(len(d))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=sd).split(d,y):
        m=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=10,rsm=0.5,eval_metric='AUC',od_type='Iter',od_wait=300,verbose=0,random_seed=sd,thread_count=12)
        m.fit(d.iloc[tr],y.iloc[tr],eval_set=(d.iloc[va],y.iloc[va]))
        oof[va]=m.predict_proba(d.iloc[va])[:,1]
    oofs.append(oof); print(sd,roc_auc_score(y,oof),flush=True)
print('CB',k,np.mean([roc_auc_score(y,o) for o in oofs]))
np.save(f'data/oof_cb{k}.npy',np.mean(oofs,0))
