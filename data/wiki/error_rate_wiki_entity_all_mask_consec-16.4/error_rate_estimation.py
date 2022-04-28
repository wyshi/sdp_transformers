import pandas as pd
import numpy as np

df = pd.read_csv(
    "/local/data/wyshi/sdp_transformers/data/wiki/error_rate_wiki_entity_all_mask_consec-16.4/error_rate_no_mask.csv"
)
import scipy.stats as st

data = df.miss / df.total
st.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))
(0.0035598401982927727, 0.007182347301707227)
max(data)
0.009765625
