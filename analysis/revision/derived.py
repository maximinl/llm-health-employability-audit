import numpy as np, pandas as pd
d=np.load("posterior.npz",allow_pickle=True); post=d["post"]; cells=list(d["cells"]); idx={c:i for i,c in enumerate(cells)}
models=["claude-opus-4-8","gemini-3.1-flash-lite","gpt-5.5"]; prompts=["fairness_aware","realistic_recruiter"]
jobs=["administrative","executive","manual"]; dis=["anxiety_disorder","depression","eating_disorder","schizophrenia","hyperthyroidism"]
def cm(m,p,dd,j): return post[idx[f"{m}|{p}|{dd}|{j}"]]
def S(x): return dict(mean=x.mean(),lo=np.quantile(x,.025),hi=np.quantile(x,.975),P_neg=(x<0).mean(),R50=(abs(x)<.5).mean())
rows=[]
# prompt difference in contrast (RR - FA), per model x disclosure and per model averaged
for m in models:
    allc=[]
    for dd in dis:
        c={p:np.mean([cm(m,p,dd,j)-cm(m,p,"control",j) for j in jobs],0) for p in prompts}
        diff=c["realistic_recruiter"]-c["fairness_aware"]; allc.append(diff)
        rows.append(dict(level="prompt_diff",model=m,disclosure=dd,**S(diff)))
    rows.append(dict(level="prompt_diff_avg",model=m,disclosure="all",**S(np.mean(allc,0))))
# job-specific contrasts
for m in models:
  for p in prompts:
    for dd in dis:
      for j in jobs:
        rows.append(dict(level="job",model=m,prompt=p,disclosure=dd,job=j,**S(cm(m,p,dd,j)-cm(m,p,"control",j))))
r=pd.DataFrame(rows); r.to_csv("derived.csv",index=False)
pd.set_option("display.width",250)
print(r[r.level.str.startswith("prompt")].round(3).to_string())
jb=r[r.level=="job"]
print("job-specific: n",len(jb),"range",jb["mean"].min().round(3),jb["mean"].max().round(3),
 "n mean>0.5",(jb["mean"]>.5).sum(),"n hi>0.5",(jb.hi>.5).sum(),"n lo<-0.5",(jb.lo<-.5).sum(),"n lo<0",(jb.lo<0).sum(), "n hi<0",(jb.hi<0).sum())
print(jb.sort_values("mean").head(4).round(3).to_string()); print(jb.sort_values("mean").tail(6).round(3).to_string())
