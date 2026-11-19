import pandas as pd, numpy as np
from scipy import stats
df = pd.read_csv("scored.csv")
def icc(g,col):
    groups=[x[col].values for _,x in g.groupby("stimulus_id")]
    n=len(groups); k=len(groups[0]); gm=g[col].mean()
    msb=k*sum((x.mean()-gm)**2 for x in groups)/(n-1)
    msw=sum(((x-x.mean())**2).sum() for x in groups)/(n*(k-1))
    F=msb/msw; d1,d2=n-1,n*(k-1)
    FL=F/stats.f.ppf(.975,d1,d2); FU=F*stats.f.ppf(.975,d2,d1)
    return dict(ICC1=(msb-msw)/(msb+(k-1)*msw), ICC1_lo=(FL-1)/(FL+k-1), ICC1_hi=(FU-1)/(FU+k-1),
                ICC1k=1-1/F, ICC1k_lo=1-1/FL, ICC1k_hi=1-1/FU, within_SD=np.sqrt(msw))
out=[]
for (m,p), g in df.groupby(["model","prompt_condition"]):
    for col in ["orig","rev"]:
        out.append(dict(model=m,prompt=p,composite=col,**icc(g,col),
            identical_stimuli=(g.groupby("stimulus_id")[col].nunique()==1).sum()))
r=pd.DataFrame(out); pd.set_option("display.width",250)
print(r.round(3).to_string()); r.to_csv("icc.csv",index=False)
