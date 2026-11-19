import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm
df=pd.read_csv("scored.csv"); icc=pd.read_csv("icc.csv"); icc=icc[icc.composite=="rev"]
items9=["communication","harmony","ambition","competence","responsibility","dependability","resilience","independence","recommendation"]
rows=[]
for (m,p),g in df.groupby(["model","prompt_condition"]):
    I=icc[(icc.model==m)&(icc.prompt==p)].iloc[0]
    pos=g[items9].values.ravel()
    rows.append(dict(model=m,prompt=p,M=g.rev.mean(),SD=g.rev.std(),mn=g.rev.min(),mx=g.rev.max(),
      within=I.within_SD,pct89=(pos>=8).mean()*100,pct9=(pos==9).mean()*100,itemmin=pos.min(),elim_mean=g.elimination.mean(),
      ICC1=f"{I.ICC1:.2f} [{I.ICC1_lo:.2f}, {I.ICC1_hi:.2f}]",ICC1k=f"{I.ICC1k:.2f} [{I.ICC1k_lo:.2f}, {I.ICC1k_hi:.2f}]"))
t=pd.DataFrame(rows); pd.set_option("display.width",250); print(t.round(2).to_string()); t.to_csv("table1.csv",index=False)
# item table
items=items9+["elimination"]
dd=df.assign(disc=df.disclosure!="control")
out={}
for m,g in dd.groupby("model"):
    c=g[~g.disc][items].mean(); d=g[g.disc][items].mean()
    out[(m,"control")]=c; out[(m,"diff")]=d-c
it=pd.DataFrame(out).round(2); print(it); it.to_csv("items.csv")
# job-type means
print(df.groupby("job_type").rev.mean().round(2)); print(df.groupby(["model","job_type"]).rev.mean().round(2).unstack())
print(df.groupby(["model","prompt_condition"]).rev.mean().round(2).unstack())
# ANOVA type II global
fit=smf.ols("rev ~ C(model)*C(prompt_condition)*C(job_type)*C(disclosure)",df).fit()
a=sm.stats.anova_lm(fit,typ=2); a["eta_p2"]=a.sum_sq/(a.sum_sq+a.loc["Residual","sum_sq"])
print(a.round(4).to_string())
