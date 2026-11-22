import pandas as pd
o = pd.read_parquet("results/step0/census_blood_obs.parquet")
o["uid"] = o.dataset_id.astype(str) + ":" + o.donor_id.astype(str)
e = o.self_reported_ethnicity.astype(str)
afr = e.str.contains("African|Ethiopian", case=False)
unk = e.str.lower().eq("unknown")
print("Total donors:", o.uid.nunique())
print("African ancestry:", o[afr].uid.nunique())
print("Not recorded:", o[unk].uid.nunique())
