"""Render the historical ROC coordinates without a held-out-generalization title.

Optional plotting dependency: matplotlib (validated with 3.11.2).
The standard-library numerical audit does not require matplotlib.
"""
from pathlib import Path
from collections import defaultdict
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
groups=defaultdict(list)
for row in csv.DictReader((root/'evidence/roc-points.csv').open(encoding='utf-8-sig',newline='')):
    groups[row['model']].append(row)
print('Models:', ', '.join(groups))
plt.rcParams.update({'font.family':'serif','font.size':14,'axes.labelsize':15})
fig,ax=plt.subplots(figsize=(6.4,4.1),layout='constrained')
names={'operation_only':('Operation only','0.5'), 'measured_process_context':('Logistic L2','#1766a6'), 'random_forest_context':('RF (post-hoc)','#bf3737')}
for model,rows in groups.items():
    xy=[(float(r['fpr']),float(r['tpr'])) for r in sorted(rows,key=lambda r:int(r['sequence']))]
    auc=sum((b[0]-a[0])*(b[1]+a[1])/2 for a,b in zip(xy,xy[1:]))
    name,color=names.get(model,('RF (post-hoc)','#bf3737'))
    assert abs(auc-(.5 if model=='operation_only' else 8/9))<1e-10
    ax.plot([p[0] for p in xy],[p[1] for p in xy],marker='o',markersize=3.5,linewidth=2,label=f'{name} ({auc:.3f})',color=color)
ax.plot([0,1],[0,1],'--',color='black',alpha=.5,linewidth=1)
ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC within the scripted design',xlim=(-.02,1.02),ylim=(-.02,1.02))
ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=13,frameon=False)
target=root/'evidence'
fig.savefig(target/'model-diagnostics.png',dpi=220)
print(target/'model-diagnostics.png')
