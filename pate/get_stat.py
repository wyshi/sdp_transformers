import pandas as pd
import argparse
import os
import glob

GLOB = glob.glob("/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/pred/*.csv")

for path in GLOB:
    df = pd.read_csv(
        path,
        # header=None,
    )
    if df.shape[1] == 4:
        print(path)
        print(df["correct_before_noise"].sum() / df.shape[0])
        print(df["correct_after_noise"].sum() / df.shape[0])
        print(df["max_cnt"].value_counts() / df.shape[0])
        print(df["truth_tok_cnt"].value_counts() / df.shape[0])
        print()
