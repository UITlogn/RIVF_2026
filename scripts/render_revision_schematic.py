"""Scientific process schematic, generated from the audited design counts."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'evidence/revision/design.csv').open(encoding='utf-8')))
batch=[r for r in rows if r['attack_launcher']=='delayed_child']
direct=[r for r in rows if r['attack_launcher']=='interactive_shell']
nb=sum(int(r['pairs']) for r in batch); nd=sum(int(r['pairs']) for r in direct)
ties=sum(int(r['pairs']) for r in direct if r['WHIT_L']==r['WHIT_A'])
assert (nb,nd,ties)==(12,12,8)
assert all(r['legitimate_launcher']=='interactive_shell' for r in rows)
plt.rcParams.update({'font.family':'serif','font.size':14})
fig,ax=plt.subplots(figsize=(6.6,4.7));fig.subplots_adjust(left=.01,right=.99,top=.98,bottom=.02)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,text,color):
    ax.add_patch(FancyBboxPatch((x-w/2,y-.043),w,.086,boxstyle='round,pad=0.005',linewidth=1.2,facecolor=color,edgecolor='#444444'))
    ax.text(x,y,text,ha='center',va='center',fontsize=13)
def arrow(x0,x1,y):
    ax.annotate('',xy=(x1,y),xytext=(x0,y),arrowprops=dict(arrowstyle='->',color='#444444',lw=1.4))
for title,y in [(f'(a) Scripted batch family: {nb} matched pairs',.96),(f'(b) Scripted direct family: {nd} matched pairs',.46)]:
    ax.text(.02,y,title,fontsize=15,weight='bold',va='center')
for label,y in [('Legitimate',.84),('Attack',.69),('Legitimate',.34),('Attack',.19)]:
    ax.text(.015,y,label,fontsize=12.5,va='center')
    box(.345,y,.22,'PowerShell','#deedf8');box(.86,y,.23,'Operation','#e3efda')
    if y==.69:
        box(.60,y,.17,'cmd.exe','#f6d9dd');arrow(.46,.505,y);arrow(.69,.735,y)
    else:arrow(.46,.735,y)
ax.text(.02,.565,'Pre-target discovery: 0, 2, or 4 tools (Table I).',fontsize=13,va='center')
ax.text(.02,.055,f'{ties} quiet/full pairs: identical L2; {nd-ties} partial pairs: WHIT differs.',fontsize=12.5,va='center')
target=root/'evidence'
fig.savefig(target/'process-chain-comparison.png',dpi=220)
print(target/'process-chain-comparison.png')
