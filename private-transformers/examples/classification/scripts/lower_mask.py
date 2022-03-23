from glob import glob
import os

DATA_DIR = "/local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask"

for task_path in glob(os.path.join(DATA_DIR, "*")):
    for folder in glob(os.path.join(task_path, "*")):
        for file in glob(os.path.join(folder, "*")):

            if "train" in file and "tsv" in file:
                print(file)
                with open(file, encoding="utf-8-sig") as fh:
                    lines = fh.readlines()
                post_lines = [line.replace("<MASK>", "<mask>") for line in lines]
                with open(file, "w", encoding="utf-8-sig") as fh:
                    fh.writelines(post_lines)
