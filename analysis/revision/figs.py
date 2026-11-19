import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#c3c2b7","axes.linewidth":0.8})
r=pd.read_csv("contrasts.csv")
models=[("claude-opus-4-8","Claude Opus 4.8"),("gemini-3.1-flash-lite","Gemini 3.1 Flash-Lite"),("gpt-5.5","GPT-5.5")]
dis=[("anxiety_disorder","Anxiety disorder"),("depression","Depression"),("eating_disorder","Eating disorder"),("schizophrenia","Schizophrenia"),("hyperthyroidism","Hyperthyroidism")]
pr=[("fairness_aware","Fairness-aware","#2a78d6","o",-0.14),("realistic_recruiter","Realistic recruiter","#eb6834","s",0.14)]
fig,axes=plt.subplots(1,3,figsize=(9,3.6),sharey=True)
for ax,(m,ml) in zip(axes,models):
    for x in (-0.5,0.5): ax.axvline(x,color="#c3c2b7",ls="--",lw=0.8)
    ax.axvline(0,color="#6b6a64",lw=0.9)
    for p,pl,c,mk,off in pr:
        sub=r[(r.model==m)&(r.prompt==p)].set_index("disclosure")
        for i,(d,_) in enumerate(dis):
            y=len(dis)-1-i+off; row=sub.loc[d]
            ax.plot([row.rev_lo,row.rev_hi],[y,y],color=c,lw=2,solid_capstyle="round")
            ax.plot(row.rev_diff,y,marker=mk,color=c,ms=6,mec="white",mew=1,ls="none",label=pl if i==0 else None)
    ax.set_title(ml,fontsize=10); ax.set_xlim(-0.65,0.75); ax.set_xticks([-0.5,-0.25,0,0.25,0.5])
    ax.grid(axis="x",color="#e1e0d9",lw=0.5); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(dis))); axes[0].set_yticklabels([d[1] for d in dis][::-1])
fig.supxlabel("Difference from no-disclosure control (points on 1–9 composite)",fontsize=9)
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.5,1.02))
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig("figure1.png",dpi=300,bbox_inches="tight")
# Figure 2 heatmap: job-specific contrasts averaged across prompts
df=pd.read_csv("scored.csv")
jobs=[("executive","Executive"),("administrative","Administrative"),("manual","Manual")]
cmap=LinearSegmentedColormap.from_list("div",["#e34948","#f0efec","#2a78d6"])
norm=TwoSlopeNorm(vmin=-0.8,vcenter=0,vmax=0.8)
fig,axes=plt.subplots(1,3,figsize=(9,3.3),sharey=True)
for ax,(m,ml) in zip(axes,models):
    g=df[df.model==m]; M=np.zeros((len(dis),len(jobs)))
    for i,(d,_) in enumerate(dis):
        for j,(jb,_) in enumerate(jobs):
            gj=g[g.job_type==jb]; M[i,j]=gj[gj.disclosure==d].rev.mean()-gj[gj.disclosure=="control"].rev.mean()
    im=ax.imshow(M,cmap=cmap,norm=norm,aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j,i,f"{M[i,j]:+.2f}",ha="center",va="center",fontsize=8.5,color="#0b0b0b")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Exec.","Admin.","Manual"]); ax.tick_params(length=0); ax.set_title(ml,fontsize=10)
    ax.set_yticks(range(len(dis))); ax.set_yticklabels([d[1] for d in dis])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5,3,1),minor=True); ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which="minor",color="white",lw=2); ax.tick_params(which="minor",length=0)
cb=fig.colorbar(im,ax=axes,shrink=0.85,pad=0.02,ticks=[-0.8,-0.4,0,0.4,0.8])
cb.set_label("Difference from control (points)\nred = lower, blue = higher",fontsize=8.5); cb.outline.set_visible(False)
fig.savefig("figure2.png",dpi=300,bbox_inches="tight")
print("ok")
