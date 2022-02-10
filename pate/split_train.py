import numpy as np
from pathlib import Path

N = 20
# TRAIN_PATH = "../data/wikitext-2/train.txt"
# TRAIN_SPLIT_SAVE_PATH = f"../data/wikitext-2/train_split_{N}"
TRAIN_PATH = "/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt"
TRAIN_SPLIT_SAVE_PATH = (
    f"/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train_split_{N}"
)

with open(TRAIN_PATH, "r", encoding="utf8") as fh:
    lines = fh.readlines()

save_dir = TRAIN_SPLIT_SAVE_PATH
Path(save_dir).mkdir(parents=True, exist_ok=True)

lines_split = np.array_split(lines, N)

for i, lines in enumerate(lines_split):
    with open(f"{save_dir}/train_{i}.txt", "w") as fh:
        fh.writelines(list(lines))
