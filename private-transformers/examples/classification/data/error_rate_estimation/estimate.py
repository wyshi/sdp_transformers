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
# (-0.0013247401887720514, 0.018063928918950645), sensitive_portion = 3.01%
# 0.030534351145038167
recall = (0.40199335548172754, 1)
# /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/qnli/miss_rate.csv
# (0.001244755937890738, 0.006136074703850561), sensitive_portion = 17.18%
# 0.00980392156862745
(0.9642836163920223, 0.9927546220146057)
# /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/qqp/miss_rate.csv
# (0.003212509558241031, 0.011670125552614308), sensitive_portion = 8.30%
# 0.019011406844106463
(0.8593960776793457, 0.9612950655633611)
# /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/mnli/miss_rate.csv
# (0.0028822402538490497, 0.01076488972382562), sensitive_portion = 8.63%
# 0.016339869281045753
(0.8752619962476753, 0.9666020828059206)
