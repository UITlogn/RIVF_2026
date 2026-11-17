"""Recompute revised-paper evidence from cleaned rows and frozen score exports.

Python standard library only; run from any directory. No VM or workload execution.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math, sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
OUT = ROOT/'evidence/revision'
OUT.mkdir(parents=True,exist_ok=True)
def rows(name):
    return [json.loads(x) for x in (DATA/name).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def scores(name):
    return list(csv.DictReader((ROOT/'scores'/name).open(encoding='utf-8-sig',newline='')))
def vector(r):
    c=r['context']
    return (r['operation'],c['launcher_type'],c['account_role'],*[c['observed_'+k] for k in ('whoami','hostname','ipconfig','tasklist')])
def fingerprint(rs):
    joint=Counter((vector(r),r['label']) for r in rs)
    serial=json.dumps(sorted((list(v),y,n) for (v,y),n in joint.items()),separators=(',',':'))
    return hashlib.sha256(serial.encode()).hexdigest()
def wilson(k,n):
    z=1.959963984540054; p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]
def metrics(rs):
    tp=fp=tn=fn=0; pairs=defaultdict(dict); pos=[]; neg=[]
    for r in rs:
        y=r['label']=='attack_preparation'; s=float(r['score']); a=s>=float(r['threshold'])
        assert a==(r['alerted'].lower()=='true')
        tp+=y and a; fp+=not y and a; fn+=y and not a; tn+=not y and not a
        (pos if y else neg).append(s); pairs[r['pair_id']][y]=s
    w=sum(p[True]>p[False] for p in pairs.values()); t=sum(p[True]==p[False] for p in pairs.values())
    auc=sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
    return dict(tp=tp,fn=fn,fp=fp,tn=tn,auc=auc,wins=w,ties=t,losses=len(pairs)-w-t,pair_rank=(w+t/2)/len(pairs),tpr_ci=wilson(tp,len(pos)),fpr_ci=wilson(fp,len(neg)))
def writecsv(name,rs):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

v5=rows('complete-pairs-measured-claim-v2.jsonl');v6=rows('complete-pairs-temporal-replication-v6.jsonl')
sets={s:[r for r in v5 if r['split']==s] for s in ('development','validation','test')};sets['v6']=v6
models=['operation_only','measured_actor_context','measured_process_context']
s5=scores('v5-logistic-predictions.csv');s6=scores('v6-locked-predictions.csv')
sys.path.insert(0,str(ROOT/'scripts'))
from apply_locked_context_model import score
frozen=json.loads((ROOT/'reports/v5-model-artifacts.json').read_text(encoding='utf-8-sig'))['models']
# The historical v6 export contains L0 and L2. Recompute L1 explicitly.
for r in v6:
    m='measured_actor_context'; s=score(r,frozen[m]); threshold=frozen[m]['threshold']
    s6.append(dict(r,model=m,score=s,threshold=threshold,alerted=str(s>=threshold)))
overlap=[]; coll=[]; design=[]; ablation=[]; report={}
base=fingerprint(sets['development'])
for name,rs in sets.items():
    pairs=defaultdict(list)
    for r in rs:pairs[r['pair_id']].append(r)
    assert len(rs)==48 and len(pairs)==24
    assert all(len(p)==2 and len({r['label'] for r in p})==2 and len({r['operation'] for r in p})==1 for p in pairs.values())
    collisions=[p for p in pairs.values() if vector(p[0])==vector(p[1])]
    assert len(collisions)==8 and len({vector(r) for r in rs})==28 and fingerprint(rs)==base
    overlap.append(dict(collection=name,records=len(rs),pairs=len(pairs),unique_vectors=len({vector(r) for r in rs}),collisions=len(collisions),joint_multiset_sha256=fingerprint(rs),equals_development=True))
    for m in models:
        ss=[r for r in (s6 if name=='v6' else s5) if r['model']==m and (name=='v6' or r['split']==name)]
        assert {r['record_id'] for r in ss}=={r['record_id'] for r in rs}
        index={r['record_id']:r for r in rs}
        assert all(all(s[k]==index[s['record_id']][k] for k in ('pair_id','label','operation')) for s in ss)
        assert all(abs(float(s['score'])-score(index[s['record_id']],frozen[m]))<1e-12 for s in ss)
        result=metrics(ss); expected=[0,12,16][models.index(m)]
        assert (result['tp'],result['fp'],result['ties'])==(expected,0,24-expected)
        ablation.append(dict(collection=name,model=m,**result))
    if name=='test':
        for p in collisions:
            v=vector(p[0]); coll.append(dict(pair_id=p[0]['pair_id'],operation=v[0],launcher=v[1],role=v[2],WHIT=''.join(str(int(x)) for x in v[3:])))
        counter=Counter()
        for p in pairs.values():
            l=next(r for r in p if r['label']=='legitimate_admin');a=next(r for r in p if r['label']=='attack_preparation')
            lv,av=vector(l),vector(a)
            counter[(av[1],lv[1],''.join(str(int(x)) for x in lv[3:]),''.join(str(int(x)) for x in av[3:]))]+=1
        assert len(counter)==6 and set(counter.values())=={4}
        for (a,l,wl,wa),n in sorted(counter.items()):design.append(dict(attack_launcher=a,legitimate_launcher=l,WHIT_L=wl,WHIT_A=wa,pairs=n))
        assert {(r['WHIT_L'],r['WHIT_A']) for r in design}=={('0000','0000'),('1100','0101'),('1111','1111')}

lookup=defaultdict(list)
for r in sets['development']:lookup[vector(r)].append(r['label']=='attack_preparation')
lookup={k:sum(v)/len(v) for k,v in lookup.items()}
lookup_results={}
for name,rs in sets.items():
    fake=[dict(r,score=lookup[vector(r)],threshold=.75,alerted=str(lookup[vector(r)]>=.75)) for r in rs]
    met=metrics(fake);assert (met['tp'],met['fp'],met['ties'])==(16,0,8)
    assert abs(met['auc']-8/9)<1e-12
    lookup_results[name]=met

writecsv('overlap.csv',overlap);writecsv('design.csv',design);writecsv('collisions.csv',sorted(coll,key=lambda r:r['operation']));writecsv('ablation.csv',ablation)
report=dict(status='PASS',kind='offline_recomputation_not_new_collection',python=sys.version,overlap=overlap,lookup_diagnostic=lookup_results,primary_test=next(r for r in ablation if r['collection']=='test' and r['model']==models[2]),pair_rank_ceiling=(16+.5*8)/24,input_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['data/complete-pairs-measured-claim-v2.jsonl','data/complete-pairs-temporal-replication-v6.jsonl','scores/v5-logistic-predictions.csv','scores/v6-locked-predictions.csv','reports/v5-model-artifacts.json']})
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('PASS: four identical feature-label multisets; all primary counts, 8 collision pairs, lookup diagnostic and Wilson intervals reproduced.')
