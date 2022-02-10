# CUDA_VISIBLE_DEVICES=5 python run_clm_no_trainer.py
# --model_name_or_path gpt2-medium
# --train_file /local-scratch1/data/wyshi/privacy/data/wikitext-2/train_split_10/train_0.txt
# --output_dir ~/privacy/pate/checkpoint/20210117/train10/clm-0
# --per_device_train_batch_size 1
# --per_device_eval_batch_size 1
# --validation_file /local-scratch1/data/wyshi/privacy/data/wikitext-2/valid.txt

import os

CUDA = "CUDA_VISIBLE_DEVICES=7"
CMD = "python run_clm_no_trainer.py"
MODEL = "--model_name_or_path gpt2-medium"
TRAIN = "--train_file"
VALID = (
    "--validation_file /local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/valid.txt"
)
OUTPUT = "--output_dir"
LOG = "--log_file"
TRAIN_SIZE = "--per_device_train_batch_size 1"
EVAL_SIZE = "--per_device_eval_batch_size 1"

TRAIN_DIR = "/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train_split_20/"
OUTPUT_DIR = "/local-scratch1/data/wyshi/privacy/pate/checkpoint/20220129/train20/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

N = 20

COMPONENTS = [
    CUDA,
    CMD,
    MODEL,
    TRAIN_SIZE,
    EVAL_SIZE,
    VALID,
]
PREFIX = " ".join(COMPONENTS)
train_txts = [TRAIN + " " + os.path.join(TRAIN_DIR, f"train_{i}.txt") for i in range(N)]
output_dirs = [OUTPUT + " " + os.path.join(OUTPUT_DIR, f"clm_{i}") for i in range(N)]
log_dirs = [LOG + " " + os.path.join(OUTPUT_DIR, f"clm_{i}.log") for i in range(N)]

for train_txt, output_dir, log_dir in zip(train_txts, output_dirs, log_dirs):
    print(f"{PREFIX} {train_txt} {output_dir} {log_dir}")
