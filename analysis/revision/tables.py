import numpy as np, pandas as pd, json
d=np.load("posterior.npz",allow_pickle=True); post=d["post"]; idx={c:i for i,c in enumerate(d["cells"])}
models=["claude-opus-4-8","gemini-3.1-flash-lite","gpt-5.5"]; prompts=["fairness_aware","realistic_recruiter"]
jobs=["administrative","executive","manual"]; dis=["anxiety_disorder","depression","eating_disorder","schizophrenia","hyperthyroidism"]; mental=dis[:4]
def m(mo,p,dd): return np.mean([post[idx[f"{mo}|{p}|{dd}|{j}"]] for j in jobs],0)
def con(mo,p,dd): return m(mo,p,dd)-m(mo,p,"control")
def S(x): return dict(mean=float(x.mean()),lo=float(np.quantile(x,.025)),hi=float(np.quantile(x,.975)),Pneg=float((x<0).mean()),
    R50=float((abs(x)<.5).mean()),R25=float((abs(x)<.25).mean()),R10=float((abs(x)<.1).mean()))
fr=pd.read_csv("contrasts.csv")
t2=[]
for mo in models:
  for p in prompts:
    for dd in dis:
      f=fr[(fr.model==mo)&(fr.prompt==p)&(fr.disclosure==dd)].iloc[0]
      t2.append(dict(model=mo,prompt=p,disclosure=dd,diff=f.rev_diff,lo=f.rev_lo,hi=f.rev_hi,pb=f.p_bonf,ni=f["NI_p_0.5"],tost=f["TOST_p_0.5"],bf=f.BF10,
        ni25=f["NI_p_0.25"],ni10=f["NI_p_0.1"],tost25=f["TOST_p_0.25"],tost10=f["TOST_p_0.1"] if "TOST_p_0.1" in f else None,**{"b_"+k:v for k,v in S(con(mo,p,dd)).items()}))
t3=[]
allc=[con(mo,p,dd) for mo in models for p in prompts for dd in dis]
t3.append(dict(group="Pooled disclosure effect",label="All 30 contrasts",**S(np.mean(allc,0))))
for mo in models: t3.append(dict(group="Pooled disclosure effect",label=mo,**S(np.mean([con(mo,p,dd) for p in prompts for dd in dis],0))))
t3.append(dict(group="Mental health vs hyperthyroidism",label="All models",**S(np.mean([m(mo,p,dd)-m(mo,p,"hyperthyroidism") for mo in models for p in prompts for dd in mental],0))))
for mo in models: t3.append(dict(group="Mental health vs hyperthyroidism",label=mo,**S(np.mean([m(mo,p,dd)-m(mo,p,"hyperthyroidism") for p in prompts for dd in mental],0))))
for dd in mental: t3.append(dict(group="Mental health vs hyperthyroidism",label=dd,**S(np.mean([m(mo,p,dd)-m(mo,p,"hyperthyroidism") for mo in models for p in prompts],0))))
for mo in models: t3.append(dict(group="Prompt difference (realistic recruiter minus fairness-aware)",label=mo,**S(np.mean([con(mo,"realistic_recruiter",dd)-con(mo,"fairness_aware",dd) for dd in dis],0))))
json.dump(dict(t2=t2,t3=t3),open("tables.json","w"),indent=1)
pd.set_option("display.width",250)
print(pd.DataFrame(t3).round(3).to_string())
t2d=pd.DataFrame(t2); print((t2d.tost10<.05).sum() if t2d.tost10.notna().all() else "no tost10")
print("b_Pneg max",t2d.b_Pneg.max(),"b_R50 min",t2d.b_R50.min())
