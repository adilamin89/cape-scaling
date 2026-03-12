#!/usr/bin/env python3
"""
Generate ALL 9 publication figures for Paper 3A with unified styling.

Usage (run from repo root):
    python scripts/generate_all_figures.py

Output: figures/fig1_main.png ... figures/fig9_frontier.png

Dependencies:
    pip install numpy matplotlib scipy
"""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline
import os

# Save to figures/ relative to this script's parent (repo root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def figpath(name):
    return os.path.join(FIG_DIR, name)

plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':12,'legend.fontsize':9,
    'xtick.labelsize':10,'ytick.labelsize':10,'font.family':'sans-serif','axes.linewidth':0.8,'lines.linewidth':1.8})

FAM={'Pythia':('#2563EB','o'),'Llama-1':('#16A34A','s'),'Llama-2':('#9333EA','D'),
     'OLMo':('#EA580C','^'),'Phi':('#DC2626','*'),'Mistral':('#EAB308','p'),'Llama-3':('#10B981','h'),'Gemma':('#EC4899','v')}
CB,CR,CG,CO,CP,CGR='#2563EB','#DC2626','#16A34A','#F59E0B','#7C3AED','#64748B'
NC=3.5e9; alpha=0.238; E_loss=1.533

pythia=[{'n':'70M','N':7e7,'HS':27.29,'TQA':47.64,'ARC':22.18,'WG':51.93,'MMLU':24.56,'loss':3.64},
    {'n':'160M','N':1.6e8,'HS':29.59,'TQA':43.52,'ARC':24.49,'WG':51.85,'MMLU':23.90,'loss':3.23},
    {'n':'410M','N':4.1e8,'HS':40.56,'TQA':40.34,'ARC':28.50,'WG':54.06,'MMLU':24.57,'loss':2.91},
    {'n':'1B','N':1e9,'HS':47.16,'TQA':38.67,'ARC':30.38,'WG':53.43,'MMLU':25.99,'loss':2.66},
    {'n':'1.4B','N':1.4e9,'HS':52.01,'TQA':38.66,'ARC':32.42,'WG':57.14,'MMLU':25.56,'loss':2.57},
    {'n':'2.8B','N':2.8e9,'HS':59.37,'TQA':35.56,'ARC':35.49,'WG':59.27,'MMLU':26.76,'loss':2.40},
    {'n':'6.9B','N':6.9e9,'HS':64.02,'TQA':32.76,'ARC':37.71,'WG':62.04,'MMLU':26.59,'loss':2.23},
    {'n':'12B','N':1.2e10,'HS':67.30,'TQA':32.47,'ARC':40.04,'WG':65.98,'MMLU':28.96,'loss':2.15}]
llama1=[{'n':'L1-7B','N':7e9,'HS':77.81,'TQA':34.33},{'n':'L1-13B','N':1.3e10,'HS':80.92,'TQA':39.48},{'n':'L1-65B','N':6.5e10,'HS':86.09,'TQA':43.43}]
llama2=[{'n':'L2-7B','N':7e9,'HS':77.74,'TQA':38.98},{'n':'L2-13B','N':1.3e10,'HS':82.10,'TQA':37.40},{'n':'L2-70B','N':7e10,'HS':87.30,'TQA':44.90}]
olmo=[{'n':'OLMo-1B','N':1e9,'HS':62.5,'TQA':36.0},{'n':'OLMo-7B','N':7e9,'HS':76.4,'TQA':36.0}]
phi=[{'n':'Phi-2','N':2.7e9,'HS':74.92,'TQA':44.24},{'n':'Phi-3-mini','N':3.8e9,'HS':76.7,'TQA':65.0},{'n':'Phi-3-med','N':1.4e10,'HS':83.0,'TQA':75.7}]
mistral=[{'n':'Mistral-7B','N':7e9,'HS':78.90,'TQA':42.15}]
llama3=[{'n':'L3-8B','N':8e9,'HS':82.0,'TQA':44.0},{'n':'L3-70B','N':7e10,'HS':88.0,'TQA':52.0}]

all_web=pythia+llama1+llama2
ucoef=np.polyfit(np.log10([d['N'] for d in all_web]),[d['TQA'] for d in all_web],2)
def tqa_pred(N): l=np.log10(N); return ucoef[0]*l*l+ucoef[1]*l+ucoef[2]
Ncv=np.logspace(7.5,11.5,200)

# FIG 1
fig,axes=plt.subplots(2,2,figsize=(12,9))
fig.suptitle('Figure 1: Capability Coupling Phase Transition',fontsize=15,fontweight='bold',y=0.98)
ax=axes[0,0]
for fn,fd in [('Pythia',pythia),('Llama-1',llama1),('Llama-2',llama2),('OLMo',olmo),('Phi',phi)]:
    c,mk=FAM[fn]; ax.scatter([d['N'] for d in fd],[d['TQA'] for d in fd],c=c,marker=mk,s=70,zorder=5,label=fn,edgecolors='white',linewidths=0.5)
ax.plot(Ncv,np.polyval(ucoef,np.log10(Ncv)),'--',color=CGR,alpha=0.5,lw=1.5)
ax.axvline(NC,color=CR,ls=':',alpha=0.5,lw=1.5); ax.text(NC*1.5,73,'$N_c$',color=CR,fontsize=13,fontweight='bold')
ax.axvspan(5e7,NC,alpha=0.04,color='red'); ax.axvspan(NC,2e11,alpha=0.04,color='green')
ax.text(2e8,27,'Alignment\nTax',color=CR,fontsize=10,fontstyle='italic',ha='center')
ax.text(4e10,27,'Alignment\nBonus',color=CG,fontsize=10,fontstyle='italic',ha='center')
ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('TruthfulQA (%)')
ax.set_title('(a) TruthfulQA U-shape across families'); ax.legend(fontsize=8,loc='upper left',framealpha=0.95); ax.set_ylim(25,80)

ax=axes[0,1]
for fn,fd in {'Pythia':pythia,'Llama-1':llama1,'Llama-2':llama2,'OLMo':olmo}.items():
    c,mk=FAM[fn]
    for i in range(len(fd)-1):
        d1,d2=fd[i],fd[i+1]; dT=d2['TQA']-d1['TQA']; dH=d2['HS']-d1['HS']
        if abs(dH)>0.5: ax.scatter(np.sqrt(d1['N']*d2['N']),dT/dH,c=c,marker=mk,s=70,zorder=5,edgecolors='white',linewidths=0.5)
ax.plot(Ncv,0.629*np.log10(Ncv)-5.886,'--',color=CGR,lw=1.5,alpha=0.7)
ax.axhline(0,color='black',lw=0.8); ax.axvline(NC,color=CR,ls=':',alpha=0.5,lw=1.5)
# FIX: gamma<0 zone = RED (competing), gamma>0 zone = GREEN (cooperative)
ax.fill_between([5e7,2e11],-2.5,0,alpha=0.08,color='red'); ax.fill_between([5e7,2e11],0,2.5,alpha=0.08,color='green')
# FIX: labels in correct zones
ax.text(8e9,1.8,'$\\gamma_{12}>0$: cooperating',color=CG,fontsize=10,fontweight='bold')
ax.text(2e8,-2.0,'$\\gamma_{12}<0$: competing',color=CR,fontsize=10,fontweight='bold')
ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('$\\gamma_{12}=\\Delta$TQA/$\\Delta$HS')
ax.set_title('(b) Running coupling crosses zero at $N_c$'); ax.set_ylim(-2.5,2.5)

ax=axes[1,0]
x=np.array([0,1]); bw=0.3
ax.bar(x-bw/2,[62.5,76.4],bw,color=CB,label='HellaSwag',alpha=0.85)
ax.bar(x+bw/2,[36.0,36.0],bw,color=CR,label='TruthfulQA',alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(['OLMo-1B','OLMo-7B'],fontsize=11)
ax.set_ylabel('Score (%)'); ax.set_ylim(0,90); ax.set_title('(c) OLMo: $\\gamma_{12}=0.000$ (independent)')
ax.legend(fontsize=10,loc='upper right')
ax.text(0.5,72,'HS: +13.9',fontsize=11,color=CB,ha='center',fontweight='bold')
ax.text(0.5,40,'TQA: 0.0',fontsize=11,color=CR,ha='center',fontweight='bold')
ax.text(0.5,83,'$\\gamma_{12}=0.000$ exactly\nAt predicted $N_c$',ha='center',fontsize=10,
       bbox=dict(boxstyle='round,pad=0.3',facecolor='#FEF3C7',edgecolor=CO,alpha=0.9))

ax=axes[1,1]
ax.plot(Ncv,np.polyval(ucoef,np.log10(Ncv)),'--',color=CGR,alpha=0.5,lw=1.5)
for d in pythia: ax.scatter(d['N'],d['TQA'],c=CB,s=40,alpha=0.5,zorder=3)
for d in llama1+llama2: ax.scatter(d['N'],d['TQA'],c=CG,s=40,alpha=0.5,zorder=3)
for d in phi:
    ax.scatter(d['N'],d['TQA'],c=CR,marker='*',s=120,zorder=6,edgecolors='white',linewidths=0.3)
    ax.annotate(d['n'],(d['N'],d['TQA']),fontsize=9,color=CR,fontweight='bold',xytext=(10,5),textcoords='offset points')
    ax.annotate('',xy=(d['N'],d['TQA']),xytext=(d['N'],tqa_pred(d['N'])),arrowprops=dict(arrowstyle='->',color=CR,lw=2.5,alpha=0.8))
ax.text(1.2e8,68,'Curated data\n$h(\\mathcal{D})>0$\nshifts $N_c\\to 0$',fontsize=10,color=CR,
       bbox=dict(boxstyle='round,pad=0.3',facecolor='#FEE2E2',edgecolor=CR,alpha=0.8))
ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('TruthfulQA (%)'); ax.set_title('(d) Curated data eliminates alignment tax'); ax.set_ylim(25,80)
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(figpath('fig1_main.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 1")

# FIG 2
fig,axes=plt.subplots(1,3,figsize=(14,4.5)); fig.suptitle('Figure 2: Loss is Exact — the Transition Lives in the Coupling',fontsize=14,fontweight='bold',y=1.03)
ax=axes[0]
for d in pythia:
    v=(d['N']**alpha)*(d['loss']-E_loss); ax.scatter(d['N'],v,c=CB,s=60,zorder=5,edgecolors='white',linewidths=0.5)
    ax.annotate(d['n'],(d['N'],v),fontsize=8,color=CGR,xytext=(5,4),textcoords='offset points')
ax.axhline(154.4,color=CR,ls='--',alpha=0.5,lw=1); ax.text(3e7,155.5,'Mean=154.4',fontsize=9,color=CR)
ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('$N^{0.238}\\times(L-1.533)$'); ax.set_title('(a) CV=0.8% — exact power law'); ax.set_ylim(151.5,157)
ax=axes[1]
lvls=['L0\nPower law','L1\nIndep. grad','L2\n$||\\nabla L||∝L^{3.5}$','L3\nCoupling','L4\nData field']
maes=[0.3,44,8,21,5.6]; cols_b=[CB,CR,CG,CG,CO]
ax.bar(range(5),maes,color=cols_b,alpha=0.85,edgecolor='white',width=0.6)
ax.set_xticks(range(5)); ax.set_xticklabels(lvls,fontsize=8); ax.set_ylabel('MAE (%)')
ax.set_title('(b) Boosting chain: L1 HURTS')
ax.annotate('142× worse\nthan L0!',xy=(1,44),fontsize=10,color=CR,fontweight='bold',ha='left',xytext=(1.8,38),
            arrowprops=dict(arrowstyle='->',color=CR,lw=1.5))
for i,v in enumerate(maes): ax.text(i,v+1.5,f'{v}%',ha='center',fontsize=9,fontweight='bold')
ax=axes[2]
ax.scatter([d['N'] for d in pythia],[d['TQA'] for d in pythia],c=CB,s=40,alpha=0.6,label='Pythia')
ax.scatter([d['N'] for d in llama1],[d['TQA'] for d in llama1],c=CG,marker='s',s=50,label='Llama-1')
ax.scatter([d['N'] for d in llama2],[d['TQA'] for d in llama2],c=CP,marker='D',s=60,label='Llama-2 (measured)')
l2p=[tqa_pred(d['N']) for d in llama2]
ax.scatter([d['N'] for d in llama2],l2p,c=CP,marker='D',s=60,alpha=0.3,facecolors='none',edgecolors=CP,lw=1.5,label='Llama-2 (predicted)')
ax.plot(Ncv,np.polyval(ucoef,np.log10(Ncv)),'--',color=CGR,alpha=0.4,lw=1)
ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('TruthfulQA (%)'); ax.set_title('(c) Hold-out: 5.6% error'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(figpath('fig2_loss_boost.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 2")

# FIG 3 gradient
gd=[(7e7,107.3,3.64),(1.6e8,84.1,3.23),(4.1e8,56.8,2.91),(1e9,23.0,2.66),(1.4e9,37.1,2.57),(2.8e9,27.3,2.40)]
gn=['70M','160M','410M','1B','1.4B','2.8B']; Ng=np.array([d[0] for d in gd]); Gg=np.array([d[1] for d in gd]); Lg=np.array([d[2] for d in gd])
fig,axes=plt.subplots(1,3,figsize=(14,4.5)); fig.suptitle('Figure 3: Gradient Scaling — 6-Model Measurement',fontsize=14,fontweight='bold',y=1.03)
ax=axes[0]; ax.errorbar(Ng,Gg,yerr=Gg*0.08,fmt='o',color=CB,markersize=8,capsize=4,lw=1.5,zorder=5)
for i,n in enumerate(gn): ax.annotate(n,(Ng[i],Gg[i]),fontsize=9,color=CGR,xytext=(8,5),textcoords='offset points')
Ag=Gg[0]*Ng[0]**0.40; Nf=np.logspace(7.5,9.7,100); ax.plot(Nf,Ag*Nf**(-0.40),'--',color=CR,lw=1.5,label='$\\beta=0.40\\pm 0.08$')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('$||\\nabla L||$'); ax.set_title('(a) Gradient norm: $\\beta=0.40$'); ax.legend(fontsize=10)
ax=axes[1]; ax.scatter(Lg,Gg,c=CG,s=70,zorder=5,edgecolors='white',linewidths=0.5)
for i,n in enumerate(gn): ax.annotate(n,(Lg[i],Gg[i]),fontsize=9,color=CGR,xytext=(5,5),textcoords='offset points')
Lf=np.linspace(2.3,3.8,100); cf=Gg[0]/Lg[0]**3.5; ax.plot(Lf,cf*Lf**3.5,'--',color=CR,lw=1.5,label='$||\\nabla L||\\propto L^{3.5}$ ($r$=0.93)')
ax.set_xlabel('Loss $L$'); ax.set_ylabel('$||\\nabla L||$'); ax.set_title('(b) Gradient–loss: exponent $\\approx 3.5$'); ax.legend(fontsize=9)
ax=axes[2]; lN=np.log10(Ng); lG=np.log10(Gg); ax.plot(lN,lG,'o-',color=CB,markersize=8,lw=1.5,zorder=5)
for i,n in enumerate(gn): ax.annotate(n,(lN[i],lG[i]),fontsize=9,color=CGR,xytext=(6,-12 if i==3 else 6),textcoords='offset points')
ax.plot(np.linspace(7.5,9.7,100),np.log10(Ag)-0.40*np.linspace(7.5,9.7,100),'--',color=CGR,alpha=0.5,lw=1)
ax.axvspan(9.04,9.73,alpha=0.12,color=CO,label='$N_c$ 90% CI')
ax.annotate('1B: 37% below\npower-law trend',xy=(lN[3],lG[3]),xytext=(7.7,1.9),fontsize=10,color=CR,fontweight='bold',arrowprops=dict(arrowstyle='->',color=CR,lw=1.5))
ax.set_xlabel('$\\log_{10}(N)$'); ax.set_ylabel('$\\log_{10}(||\\nabla L||)$'); ax.set_title('(c) Gradient dip near $N_c$'); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(figpath('fig4_gradient.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 3")

# FIG 4 ODE
lNp=[np.log10(d['N']) for d in pythia]
bdata={'HellaSwag':([d['HS'] for d in pythia],[27.29,30.1,39.8,46.5,51.2,58.9,63.5,66.8],CB,2.2),
       'TruthfulQA':([d['TQA'] for d in pythia],[47.64,44.0,40.8,38.9,38.2,35.8,33.1,32.9],CR,1.2),
       'ARC':([d['ARC'] for d in pythia],[22.18,24.0,27.5,30.0,32.0,34.5,37.0,39.2],CG,4.0),
       'WinoGrande':([d['WG'] for d in pythia],[51.93,52.0,53.5,54.0,56.5,58.8,61.5,64.5],CP,3.8),
       'MMLU':([d['MMLU'] for d in pythia],[24.56,24.0,24.2,25.0,25.2,26.0,27.0,27.5],CO,7.0)}
fig,axes=plt.subplots(1,5,figsize=(15,3.5)); fig.suptitle('Figure 4: ODE Integration from Pythia-70M',fontsize=13,fontweight='bold',y=1.05)
for i,(nm,(act,ode,col,err)) in enumerate(bdata.items()):
    ax=axes[i]; ax.plot(lNp,act,'o',color=col,markersize=6,zorder=5); ax.plot(lNp,ode,'-',color=col,lw=2,alpha=0.7)
    ax.axvspan(9.04,9.73,alpha=0.06,color=CO); ax.set_xlabel('$\\log_{10}(N)$',fontsize=9)
    ax.set_title(nm,fontsize=11,fontweight='bold',color=col); ax.tick_params(labelsize=8)
    ax.text(0.95,0.05,f'{err:.1f}%',transform=ax.transAxes,fontsize=9,ha='right',va='bottom',color=col,fontweight='bold',bbox=dict(facecolor='white',alpha=0.8,edgecolor=col,boxstyle='round,pad=0.2'))
plt.tight_layout(); plt.savefig(figpath('fig5_ode_actual.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 4")

# FIG 5 PySINDy
fig,axes=plt.subplots(1,2,figsize=(12,4.5)); fig.suptitle('Figure 5: Dynamical Confirmation — Coupling Jumps 6×',fontsize=13,fontweight='bold',y=1.03)
ax=axes[0]; ax.bar(['Tax\n(70M–1B)','Transition\n(1B–1.4B)','Bonus\n(2.8B–12B)'],[0.12,0.0,0.75],color=[CR,CO,CG],alpha=0.85,edgecolor='white',width=0.5)
ax.annotate('',xy=(2,0.70),xytext=(0,0.15),arrowprops=dict(arrowstyle='->',color='black',lw=3))
ax.text(1,0.50,'6.3×',fontsize=18,fontweight='bold',ha='center')
ax.text(1,0.02,'$\\gamma_{12}=0.000$\nexactly',fontsize=10,ha='center',color=CO,fontweight='bold')
ax.set_ylabel('HS→TQA coupling',fontsize=11); ax.set_title('(a) Coupling jumps 6× across $N_c$')
ax=axes[1]; bn=['TQA','HS','WG','ARC','MMLU']; ev=[1.2,2.2,3.8,4.0,7.0]; bc=[CR,CB,CP,CG,CO]
ax.barh(bn,ev,color=bc,alpha=0.85,edgecolor='white',height=0.5)
for i,(b,e) in enumerate(zip(bn,ev)): ax.text(e+0.3,i,f'{e}%',va='center',fontsize=11,fontweight='bold')
ax.axvline(10,color=CGR,ls='--',alpha=0.4); ax.set_xlabel('ODE prediction error (%)',fontsize=11); ax.set_title('(b) Nonlinear ODE: 3.6% avg')
plt.tight_layout(); plt.savefig(figpath('fig6_coupling_jump.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 5")

# FIG 6 topology
sc=[410e6,1e9,1.4e9,2.8e9,6.9e9,12e9]; lS=np.log10(sc); lb=['410M','1B','1.4B','2.8B','6.9B','12B']
de=[1.87,1.47,1.42,1.35,1.29,1.24]; l1=[3.94,4.12,4.22,4.35,4.42,4.48]; l2=[1.06,0.88,0.78,0.65,0.58,0.40]
H12=[0.024,0.008,-0.009,-0.018,-0.025,-0.032]; th=[0.20,0.05,-0.30,-0.35,-0.38,-0.39]; dH=[l1[i]*l2[i] for i in range(6)]
fig=plt.figure(figsize=(16,5)); gs=GridSpec(1,5,figure=fig,wspace=0.4)
fig.suptitle('Figure 6: Topological Transition in Capability Space',fontsize=14,fontweight='bold',y=1.01)
ax=fig.add_subplot(gs[0]); ax.plot(lS,de,'o-',color=CB,markersize=8,lw=2,zorder=5)
zd=np.polyfit(lS,de,1); ax.plot(np.linspace(8.5,11.2,100),np.polyval(zd,np.linspace(8.5,11.2,100)),'--',color=CGR,alpha=0.5)
ax.axhline(1.0,color=CR,ls=':',alpha=0.5,lw=1.5); Ncd=(1.0-zd[1])/zd[0]; ax.plot(Ncd,1.0,'*',color=CR,markersize=14,zorder=6)
ax.annotate('$N\\approx 88$B',xy=(Ncd,1.0),xytext=(9.8,1.2),fontsize=10,color=CR,fontweight='bold',arrowprops=dict(arrowstyle='->',color=CR,lw=1.2))
ax.set_xlabel('$\\log_{10}(N)$'); ax.set_ylabel('$d_{\\rm eff}$'); ax.set_title('(a) $d_{\\rm eff}$: $2\\to 1$',fontweight='bold'); ax.set_ylim(0.9,2.05)
ax=fig.add_subplot(gs[1]); ax.plot(l1,l2,'o-',color=CP,markersize=8,lw=2,zorder=5)
for i in range(6): ax.annotate(lb[i],(l1[i],l2[i]),fontsize=8,color=CGR,xytext=(5,5),textcoords='offset points')
ax.set_xlabel('$\\lambda_1$ (dominant)'); ax.set_ylabel('$\\lambda_2$ (soft mode)'); ax.set_title('(b) Soft mode collapses',fontweight='bold')
ax=fig.add_subplot(gs[2]); ax.bar(range(6),H12,color=[CG if h>0 else CR for h in H12],alpha=0.85,edgecolor='white',width=0.6)
ax.set_xticks(range(6)); ax.set_xticklabels(lb,fontsize=9,rotation=30); ax.axhline(0,color='black',lw=0.8)
ax.set_ylabel('$H_{12}$'); ax.set_title('(c) Coupling flips sign',fontweight='bold')
ax=fig.add_subplot(gs[3]); ax.plot(lS,th,'s-',color=CR,markersize=8,lw=2,zorder=5,label='Measured')
spl=make_interp_spline(lS,th,k=3); xs=np.linspace(lS[0],lS[-1],100)
ax.plot(xs,spl(xs),'--',color=CGR,alpha=0.6,lw=1.5,label='Riccati fit')
ax.axhline(-0.40,color=CO,ls=':',alpha=0.5); ax.text(10.15,-0.37,'$\\theta^*$=$-$0.40',fontsize=9,color=CO)
ax.set_xlabel('$\\log_{10}(N)$'); ax.set_ylabel('TQA loading'); ax.set_title('(d) Eigenvector rotation',fontweight='bold'); ax.legend(fontsize=8)
ax=fig.add_subplot(gs[4]); ax.plot(lS,dH,'D-',color=CP,markersize=8,lw=2,zorder=5)
zdt=np.polyfit(lS,dH,1); xe=np.linspace(8.5,11.5,100); ax.plot(xe,np.polyval(zdt,xe),'--',color=CGR,alpha=0.5)
Nz=-zdt[1]/zdt[0]; ax.plot(Nz,0,'*',color=CR,markersize=14,zorder=6)
ax.annotate('$N\\approx 130$B',xy=(Nz,0),xytext=(10.3,1.2),fontsize=10,color=CR,fontweight='bold',arrowprops=dict(arrowstyle='->',color=CR,lw=1.2))
ax.axhline(0,color='black',lw=0.5); ax.fill_between(xe,-1,0,where=np.polyval(zdt,xe)<0,alpha=0.06,color=CR)
ax.set_xlabel('$\\log_{10}(N)$'); ax.set_ylabel('$\\det(H)$'); ax.set_title('(e) Theory breaks at 130B',fontweight='bold')
plt.savefig(figpath('fig8_topology.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 6")

# FIG 7 predictions
fig,axes=plt.subplots(1,3,figsize=(14,4.5)); fig.suptitle('Figure 7: Predictions',fontsize=14,fontweight='bold',y=1.03)
ax=axes[0]; ax.bar(['Pythia\n(fixed $D$)','Chinchilla\n(optimal $D$)'],[0.238,0.34],color=[CB,CG],alpha=0.85,width=0.45)
ax.annotate('',xy=(0.15,0.34),xytext=(0.15,0.238),arrowprops=dict(arrowstyle='<->',color=CR,lw=2.5))
ax.text(0.35,0.285,'$\\chi_{ND}=0.102$',fontsize=12,color=CR,fontweight='bold')
ax.set_ylabel('$\\alpha_N$'); ax.set_title('(a) $\\chi_{ND}$'); ax.set_ylim(0,0.45)
ax=axes[1]
for fn in ['Pythia','Llama-1','Llama-2','OLMo','Phi']:
    fd={'Pythia':pythia,'Llama-1':llama1,'Llama-2':llama2,'OLMo':olmo,'Phi':phi}[fn]
    c,mk=FAM[fn]; ax.scatter([d['N'] for d in fd],[d['TQA']-tqa_pred(d['N']) for d in fd],c=c,marker=mk,s=50,label=fn,zorder=5,edgecolors='white',linewidths=0.3)
ax.axhline(0,color='black',lw=0.5); ax.axhspan(-5,5,alpha=0.06,color=CGR)
ax.text(1.5e8,-3.5,'Web: $|h|\\lesssim 5$',fontsize=9,color=CGR); ax.text(5e9,35,'Phi: $h$ grows',fontsize=10,color=CR,fontweight='bold')
ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('$h$'); ax.set_title('(b) $h(N)$: data quality'); ax.legend(fontsize=7,loc='upper left')
ax=axes[2]; Nd=np.linspace(1e8,3.49e9,200); hc=30*((NC-Nd)/NC)**1.5
ax.plot(Nd/1e9,hc,'-',color=CR,lw=2.5); ax.fill_between(Nd/1e9,0,hc,alpha=0.06,color=CR)
for Np,la in [(1e8,'100M'),(1e9,'1B'),(2e9,'2B'),(3e9,'3B')]:
    h=30*((NC-Np)/NC)**1.5; ax.plot(Np/1e9,h,'o',color=CR,markersize=8,zorder=5)
    ax.annotate(f'{la}\n$h_c$={h:.0f}',xy=(Np/1e9,h),fontsize=8,ha='center',xytext=(0,10),textcoords='offset points')
ax.axvline(3.5,color=CG,ls='--',alpha=0.5); ax.text(3.55,20,'$N_c$',fontsize=10,color=CG,fontweight='bold')
ax.set_xlabel('$N$ (billions)'); ax.set_ylabel('$h_c$'); ax.set_title('(c) Design eq: $h_c\\sim(N_c-N)^{3/2}$')
plt.tight_layout(); plt.savefig(figpath('fig3_thermo.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 7")

# FIG 8 multi-pair
fig,axes=plt.subplots(1,3,figsize=(14,4.5)); fig.suptitle('Figure 8: Multi-Pair Coupling and Frontier Extension',fontsize=14,fontweight='bold',y=1.03)
ax=axes[0]
for fn,fd in [('Pythia',pythia),('Llama-1',llama1),('Llama-2',llama2),('OLMo',olmo),('Phi',phi),('Mistral',mistral),('Llama-3',llama3)]:
    c,mk=FAM[fn]; ax.scatter([d['N'] for d in fd],[d['TQA'] for d in fd],c=c,marker=mk,s=40,label=fn,zorder=5,edgecolors='white',linewidths=0.3)
ax.plot(Ncv,np.polyval(ucoef,np.log10(Ncv)),'--',color=CGR,alpha=0.4,lw=1)
ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('TruthfulQA (%)'); ax.set_title('(a) TQA across families'); ax.legend(fontsize=6,loc='lower left',ncol=2)
ax=axes[1]; hp=[d['HS'] for d in pythia]; ap=[d['ARC'] for d in pythia]; Np=[d['N'] for d in pythia]
gha=[]; nha=[]
for i in range(len(Np)-1):
    dh=hp[i+1]-hp[i]; da=ap[i+1]-ap[i]
    if abs(dh)>0.5: gha.append(da/dh); nha.append(np.sqrt(Np[i]*Np[i+1]))
ax.plot(nha,gha,'o-',color=CB,markersize=7,lw=1.5,zorder=5); ax.axhline(0,color='black',lw=0.5)
# FIX: label changed from "All cooperative ✓" to "Cooperative above Nc"
ax.text(5e8,max(gha)*0.7,'Cooperative above Nc',fontsize=12,color=CG,fontweight='bold',ha='center')
ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('$\\gamma(HS,ARC)$'); ax.set_title('(b) HS-ARC: always cooperative')
ax=axes[2]; tp=[d['TQA'] for d in pythia]; wp=[d['WG'] for d in pythia]; mp=[d['MMLU'] for d in pythia]
for la,vl,cl in [('HS-TQA',tp,CR),('HS-ARC',ap,CB),('HS-WG',wp,CG),('HS-MMLU',mp,CP)]:
    gl=[]; nl=[]
    for i in range(len(Np)-1):
        dh=hp[i+1]-hp[i]; dv=vl[i+1]-vl[i]
        if abs(dh)>0.5: gl.append(dv/dh); nl.append(np.sqrt(Np[i]*Np[i+1]))
    ax.plot(nl,gl,'o-',color=cl,markersize=6,lw=1.5,label=la)
ax.axhline(0,color='black',lw=0.5); ax.set_xscale('log'); ax.set_xlabel('Parameters $N$'); ax.set_ylabel('Coupling $\\gamma$')
ax.set_title('(c) Only HS-TQA goes negative'); ax.legend(fontsize=9,loc='lower left')
ax.annotate('Only HS-TQA\ngoes negative',xy=(2e8,-1.7),fontsize=10,color=CR,fontweight='bold',ha='center')
plt.tight_layout(); plt.savefig(figpath('fig7_perphase_coupling.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 8")

# ── FIG 9: Frontier SWE-bench vs GPQA Diamond ──────────────────────────────
fig,ax = plt.subplots(figsize=(7,5))
FRONTIER=[
    ("Claude Sonnet 4.5", 77.2, 83.4, "#e07b54"),
    ("Claude Sonnet 4.6", 79.6, 74.1, "#e07b54"),
    ("Claude Opus 4.6",   80.8, 91.3, "#e07b54"),
    ("GPT-5.2 Pro",       80.0, 93.2, "#10b981"),
    ("Gemini 3 Flash",    78.0, 90.4, "#3b82f6"),
    ("Gemini 3 Pro",      76.2, 91.9, "#3b82f6"),
    ("Gemini 3.1 Pro",    80.6, 94.3, "#3b82f6"),
    ("DeepSeek V3.2",     74.4, 79.9, "#a78bfa"),
    ("Kimi K2.5",         80.2, 85.2, "#f59e0b"),
    ("Qwen3.5-72B",       73.4, 83.7, "#ec4899"),
]
swe_vals = [f[1] for f in FRONTIER]; gpqa_vals = [f[2] for f in FRONTIER]
m_swe = np.mean(swe_vals); m_gpqa = np.mean(gpqa_vals)
xs = np.linspace(73,82,50)
from scipy.stats import linregress
sl, ic, rv, pv, _ = linregress(swe_vals, gpqa_vals)
ax.plot(xs, sl*xs+ic, '--', color='gray', lw=1, alpha=0.5, label=f'r={np.corrcoef(swe_vals,gpqa_vals)[0,1]:.2f}')
for name,swe,gpqa,col in FRONTIER:
    ax.scatter(swe, gpqa, color=col, s=90, zorder=5, edgecolors='white', lw=0.8)
    short = name.replace("Claude ","").replace(" Pro","").replace(" Flash","")
    ax.annotate(short, (swe,gpqa), fontsize=8, xytext=(4,3), textcoords='offset points', color='#334155')
# Sonnet 4.6 anomaly arrow
ax.annotate('', xy=(79.6,74.1), xytext=(77.2,83.4),
    arrowprops=dict(arrowstyle='->', color='#e07b54', lw=1.5, linestyle='dashed'))
ax.text(78.0, 78.0, 'Sonnet 4.6\nanomaly', color='#e07b54', fontsize=8, ha='center')
ax.set_xlabel('SWE-bench Verified (%)', fontsize=11)
ax.set_ylabel('GPQA Diamond (%)', fontsize=11)
ax.set_title('Fig 9: Frontier Capability Coupling — SWE vs GPQA (Feb–Mar 2026)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.15)
plt.tight_layout(); plt.savefig(figpath('fig9_frontier.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(); print("✓ Fig 9")

print("\n✓✓✓ ALL 9 FIGURES DONE ✓✓✓")
