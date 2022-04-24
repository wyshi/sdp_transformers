import numpy as np
import os

with open("/local/data/wyshi/sdp_transformers/data/abcd/abcd_my_delex-entity_only_high_3.13828/train.txt") as fh:
    lines = fh.read()

np.random.seed(1111)
dials = lines.split("<|endoftext|>")

selected_ids = np.random.choice(len(dials), size=10, replace=False)
selected_dials = [dials[_id] for _id in selected_ids]

SAVE_DIR = "/local/data/wyshi/sdp_transformers/data/abcd/error_rate_abcd_my_delex-entity_only_high_3.13828/"
os.makedirs(SAVE_DIR, exist_ok=True)
for i, dial in enumerate(selected_dials):
    save_dial_dir = os.path.join(SAVE_DIR, f"dial_{i}.txt")
    with open(save_dial_dir, "w") as fh:
        fh.writelines(dial)
        print(save_dial_dir)
