import numpy as np, pandas as pd, pymc as pm, arviz as az
df=pd.read_csv("scored.csv")
df["stratum"]=df.model+"|"+df.prompt_condition
df["cell"]=df.stratum+"|"+df.disclosure+"|"+df.job_type
cells=sorted(df.cell.unique()); strata=sorted(df.stratum.unique())
ci=df.cell.map({c:i for i,c in enumerate(cells)}).values
si=df.stratum.map({s:i for i,s in enumerate(strata)}).values
with pm.Model():
    mu=pm.Normal("mu",5,2,shape=len(cells))
    logsig=pm.Normal("logsig",np.log(0.5),1,shape=len(strata))
    pm.Normal("y",mu[ci],pm.math.exp(logsig[si]),observed=df["rev"].values)
    idata=pm.sample(draws=2000,tune=2000,chains=4,random_seed=2026,progressbar=False)
    ppc=pm.sample_posterior_predictive(idata,random_seed=1,progressbar=False)
s=az.summary(idata,var_names=["mu","logsig"])
print("rhat max",s.r_hat.max(),"ess bulk min",s.ess_bulk.min(),"ess tail min",s.ess_tail.min())
post=idata.posterior["mu"].stack(s=("chain","draw")).values
np.savez("posterior.npz",post=post,cells=np.array(cells))
# PPC: observed vs predicted within-stratum SD of y and proportion of composite >= 8
yrep=ppc.posterior_predictive["y"].stack(s=("chain","draw")).values  # n x S
for k,st in enumerate(strata):
    m=si==k; obs=df["rev"].values[m]
    sds=yrep[m].std(0); print(st,"obs SD %.3f"%obs.std(),"ppc SD 95%% [%.3f, %.3f]"%tuple(np.quantile(sds,[.025,.975])))
