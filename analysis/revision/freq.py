import sys, numpy as np, pandas as pd
from scipy import stats
import pingouin as pg  # JZS Bayes factors
df = pd.read_csv(sys.argv[1])
items9 = ["communication","harmony","ambition","competence","responsibility","dependability","resilience","independence","recommendation"]
df["orig"] = df[items9+["elimination"]].mean(axis=1)          # submitted (uncorrected) scoring, for comparison
df["rev"] = (df[items9].sum(axis=1) + (10-df["elimination"]))/10  # corrected: elimination reverse-scored
assert np.allclose(df["rev"], df["mean_employability"], atol=1e-3)
df["nine"] = df[items9].mean(axis=1)
df["decision"] = (df["recommendation"] + (10-df["elimination"]))/2
df.to_csv("scored.csv", index=False)
dis = ["anxiety_disorder","depression","eating_disorder","schizophrenia","hyperthyroidism"]
out=[]
for (m,p), g in df.groupby(["model","prompt_condition"]):
    c = g[g.disclosure=="control"]
    for d in dis:
        x = g[g.disclosure==d]
        row = dict(model=m, prompt=p, disclosure=d)
        for col in ["orig","rev","nine","decision"]:
            a, b = x[col].values, c[col].values
            diff = a.mean()-b.mean()
            va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
            se = np.sqrt(va+vb)
            dfw = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)) if se>0 else np.inf
            tcrit = stats.t.ppf(.975, dfw)
            row[f"{col}_diff"]=diff; row[f"{col}_lo"]=diff-tcrit*se; row[f"{col}_hi"]=diff+tcrit*se
            if col=="rev":
                t = diff/se; pw = 2*stats.t.sf(abs(t), dfw)
                row["p"]=pw; row["p_bonf"]=min(1,pw*30); row["se"]=se; row["dfw"]=dfw
                for dl in [0.5,0.25,0.1]:
                    p_lower = stats.t.sf((diff+dl)/se, dfw)   # H0: diff <= -dl
                    p_upper = stats.t.cdf((diff-dl)/se, dfw)  # H0: diff >= +dl
                    row[f"NI_p_{dl}"]=p_lower; row[f"TOST_p_{dl}"]=max(p_lower,p_upper)
                # blocked model: cell means disclosure x job within stratum, pooled within-cell variance
                sub = g[g.disclosure.isin(["control",d])]
                gg = g.groupby(["disclosure","job_type"])[col]
                ss = ((g[col]-gg.transform("mean"))**2).sum(); dfres = len(g)-18
                s2 = ss/dfres; seb = np.sqrt(s2*(2/15))
                tb = diff/seb if seb>0 else np.inf
                row["block_se"]=seb; row["block_lo"]=diff-stats.t.ppf(.975,dfres)*seb; row["block_hi"]=diff+stats.t.ppf(.975,dfres)*seb
                row["block_p_bonf"]=min(1,2*stats.t.sf(abs(tb),dfres)*30)
                row["block_NI_p_0.25"]=stats.t.sf((diff+0.25)/seb, dfres)
                row["block_NI_p_0.1"]=stats.t.sf((diff+0.1)/seb, dfres)
                row["block_TOST_p_0.25"]=max(stats.t.sf((diff+0.25)/seb, dfres), stats.t.cdf((diff-0.25)/seb, dfres))
                tt = stats.ttest_ind(a,b)  # Student t for JZS
                row["BF10"] = float(pg.bayesfactor_ttest(tt.statistic, len(a), len(b), r=0.707))
                row["d"] = diff/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        out.append(row)
res = pd.DataFrame(out); res.to_csv("contrasts.csv", index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(res[["model","prompt","disclosure","orig_diff","orig_lo","orig_hi","rev_diff","rev_lo","rev_hi","nine_diff","decision_diff"]].round(3).to_string())
