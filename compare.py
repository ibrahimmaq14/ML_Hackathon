"""Trip-grouped model comparisons. Run after roads.py."""
from pathlib import Path
import sys, time, csv
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from threadpoolctl import threadpool_limits

SEED = 24
RESULTS = Path(__file__).parent/'results'


def read(split):
    data = []
    for p in sorted(split.glob('*.npz')):
        with np.load(p) as z:
            data.append((np.concatenate([z['x'],z['neighbors']], axis=-1),z['y'],np.repeat(int(p.stem),len(z['y']))))
    return tuple(np.concatenate(col) for col in zip(*data))


class SageScorer(nn.Module):
    """Directed one-layer mean GraphSAGE with an observation-conditioned scoring head."""
    def __init__(self, graph=True):
        super().__init__()
        self.graph = graph
        self.road = nn.Linear(15 if graph else 5,24)
        self.head = nn.Sequential(nn.Linear(28,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x):
        road = x[:,4:] if self.graph else x[:,4:9]
        emb = torch.relu(self.road(road))
        return self.head(torch.cat([emb,x[:,:4]],dim=1)).flatten()


def neural(a,b,labels,graph):
    torch.manual_seed(SEED)
    scale = StandardScaler().fit(a)
    train = torch.from_numpy(scale.transform(a))
    y = torch.from_numpy(labels.astype('f'))
    model = SageScorer(graph)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003)
    for epoch in range(15):
        for idx in torch.randperm(len(y)).split(2048):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(train[idx]),y[idx])
            loss.backward()
            optimizer.step()
    start = time.perf_counter()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(scale.transform(b)))).numpy()
    return scores, time.perf_counter()-start


def measure(y,p):
    pred = np.arange(y.shape[1]) == p.argmax(1)[:,None]
    return dict(top1=(y & pred).any(1).mean(), macro_f1=f1_score(y.ravel(),pred.ravel(),average='macro'),
                candidate_miss=(~y.any(1)).mean(),binary_log_loss=log_loss(y.ravel(),p.ravel(),labels=[False,True]))


def run():
    cache = Path(sys.argv[1])
    x,y,group = read(cache/'train')
    test_x,test_y,test_group = read(cache/'student_test')
    models = {
        'Logistic regression':lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=250)),
        'Decision tree':lambda:DecisionTreeClassifier(max_depth=10,min_samples_leaf=32,random_state=SEED),
        'Random forest':lambda:RandomForestClassifier(n_estimators=36,max_depth=16,min_samples_leaf=10,n_jobs=3,random_state=SEED),
        'Histogram boosting':lambda:HistGradientBoostingClassifier(max_iter=100,early_stopping=False,random_state=SEED),
    }
    folds = [(f'cv{i+1}',a,x[b],y[b],group[b]) for i,(a,b) in enumerate(GroupKFold(3).split(x,groups=group))]
    folds.append(('student_test',np.arange(len(y)),test_x,test_y,test_group))
    torch.set_num_threads(2)
    with (RESULTS/'comparison.csv').open('w') as f, (RESULTS/'per_trip.csv').open('w') as bytrip:
        writer = csv.DictWriter(f,fieldnames=['split','model','fit_observations','eval_observations','top1','macro_f1','candidate_miss','binary_log_loss','fit_s','inference_s'])
        per = csv.DictWriter(bytrip,fieldnames=['split','model','trip','observations','top1','candidate_miss'])
        writer.writeheader()
        per.writeheader()
        for split, fit,b,truth,groups in folds:
            # Seeded sampling of complete observations keeps CPU training feasible.
            rng = np.random.default_rng(SEED)
            idx = np.sort(rng.choice(fit,min(len(fit),60000),replace=False))
            a = x[idx].reshape(-1,x.shape[-1])
            labels = y[idx].ravel()
            flat = b.reshape(-1,b.shape[-1])
            for name in ['Nearest road',*models,'GraphSAGE','No-neighbor ablation']:
                start = time.perf_counter()
                if name == 'Nearest road':
                    # Ranking only. No fitted probability model.
                    scores = np.exp(-b[:,:,0]).ravel()
                    elapsed = time.perf_counter()-start
                elif name in models:
                    clf = models[name]()
                    clf.fit(a[:,:9],labels)
                    tick = time.perf_counter()
                    scores = clf.predict_proba(flat[:,:9])[:,1]
                    elapsed = time.perf_counter()-tick
                else:
                    scores,elapsed = neural(a,flat,labels,name=='GraphSAGE')
                fit_time = time.perf_counter()-start-elapsed
                scores = scores.reshape(truth.shape)
                metrics = measure(truth,scores)
                if name=='Nearest road':
                    metrics['binary_log_loss']=''
                writer.writerow(dict(split=split,model=name,fit_observations=0 if name=='Nearest road' else len(idx),
                                     eval_observations=len(truth),fit_s=fit_time,inference_s=elapsed,**metrics))
                for trip in np.unique(groups):
                    mask = groups==trip
                    t = truth[mask]
                    correct = t[np.arange(len(t)),scores[mask].argmax(1)]
                    per.writerow(dict(split=split,model=name,trip=f'{trip:08}',observations=len(t),top1=correct.mean(),candidate_miss=(~t.any(1)).mean()))
                f.flush()
                bytrip.flush()
                print(split,name,metrics['top1'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
