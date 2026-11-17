"""Verify diagram counts and ROC coordinates against the original score rows."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,hashlib
R=Path(__file__).resolve().parents[1]
def csvrows(f):return list(csv.DictReader((R/f).open(encoding='utf-8-sig',newline='')))
v5=[json.loads(l) for l in (R/'data/complete-pairs-measured-claim-v2.jsonl').read_text().splitlines()]
v5=[r for r in v5 if r['split']=='test'];assert len(v5)==48
pairs=defaultdict(dict)
for r in v5:pairs[r['pair_id']][r['label']]=r
groups=Counter(); ops=Counter()
for p in pairs.values():
    l=p['legitimate_admin'];a=p['attack_preparation'];lc=l['context'];ac=a['context']
    assert l['operation']==a['operation']
    assert lc['launcher_type']=='interactive_shell' and lc['account_role']==ac['account_role']=='standard'
    fields=['observed_'+k for k in ('whoami','hostname','ipconfig','tasklist')]
    lv=tuple(lc[k] for k in fields);av=tuple(ac[k] for k in fields)
    kind='batch' if ac['launcher_type']=='delayed_child' else 'direct'
    groups[(kind,'collision' if lv==av else 'discovery_diff')]+=1
    ops[l['operation']]+=1
assert groups=={('batch','collision'):8,('batch','discovery_diff'):4,('direct','collision'):8,('direct','discovery_diff'):4}
assert set(ops.values())=={6}
all_scores=csvrows('scores/v5-logistic-predictions.csv')
series={m:[r for r in all_scores if r['split']=='test' and r['model']==m] for m in ['operation_only','measured_process_context']}
series['random_forest']=[r for r in csvrows('scores/random-forest-predictions.csv') if r['collection']=='v5' and r['split']=='test']
coordinates=csvrows('evidence/roc-points.csv');report={}
for model,ss in series.items():
    assert len(ss)==48 and {r['record_id'] for r in ss}=={r['record_id'] for r in v5}
    bins=defaultdict(lambda:[0,0])
    for s in ss:bins[float(s['score'])][s['label']=='attack_preparation']+=1
    xy=[(0.,0.)];tp=fp=0
    for threshold in sorted(bins,reverse=True):
        n,p=bins[threshold];tp+=p;fp+=n;xy.append((fp/24,tp/24))
    old=[(float(r['fpr']),float(r['tpr'])) for r in sorted([r for r in coordinates if r['model']==model],key=lambda r:int(r['sequence']))]
    assert len(xy)==len(old) and all(abs(a-c)<1e-12 and abs(b-d)<1e-12 for (a,b),(c,d) in zip(xy,old))
    auc=sum((b[0]-a[0])*(b[1]+a[1])/2 for a,b in zip(xy,xy[1:]))
    assert abs(auc-(.5 if model=='operation_only' else 8/9))<1e-12
    report[model]={'records':48,'coordinate_count':len(xy),'auc':auc,'matches_exported_coordinates':True}
out={'status':'PASS','schematic':{'batch_pairs':12,'direct_pairs':12,'direct_exact_L2_collisions':8,'direct_discovery_differences':4,'pairs_per_operation':dict(ops)},'roc':report,'scope':'historical v5 test partition; no new captures','inputs':{f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['data/complete-pairs-measured-claim-v2.jsonl','scores/v5-logistic-predictions.csv','scores/random-forest-predictions.csv','evidence/roc-points.csv']}}
(R/'evidence/revision').mkdir(parents=True,exist_ok=True)
(R/'evidence/revision/figure-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS: diagram 12 batch / 12 direct / 8 ties / 4 discovery differences; every ROC coordinate and AUC matches score rows.')
