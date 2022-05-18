import pandas as pd
import numpy as np

dfs = []
paths = [
    "/local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/mnli/miss_rate.csv",
    "/local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/qnli/miss_rate.csv",
    "/local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/qqp/miss_rate.csv",
    "/local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/sst-2/miss_rate.csv",
]
for p in paths:
    dfs.append([p, pd.read_csv(p)])
import scipy.stats as st

for p, df in dfs:
    data = df.miss / df.total
    print(p)
    print(st.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data)))
    print(max(data))

# /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/sst-2/miss_rate.csv
# (-0.0013247401887720514, 0.018063928918950645)
# 0.030534351145038167
# /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/qnli/miss_rate.csv
# (0.001244755937890738, 0.006136074703850561)
# 0.00980392156862745
# /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/qqp/miss_rate.csv
# (0.003212509558241031, 0.011670125552614308)
# 0.019011406844106463
# /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/mnli/miss_rate.csv
# (0.0028822402538490497, 0.01076488972382562)
# 0.016339869281045753
