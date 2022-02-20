"""
python /local-scratch1/data/wyshi/privacy/pate/predict_random.py -d cuda:5 -m /local-scratch1/data/wyshi/privacy/pate/checkpoint/20220129/train5 -p pred_random_not_limit_to_digits.txt 
"""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from collections import Counter
import numpy as np
from tqdm import tqdm
import logging
import warnings
import argparse
import pandas as pd
import os
import random


random.seed(1111)
warnings.filterwarnings("ignore")

GAMMA = 0.5
N = 5
SAVE_DIR = "/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/pred"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--pred_file",
        "-p",
        type=str,
        help="the file name to save the predicted txt",
    )
    # parser.add_argument(
    #     "--csv_file",
    #     "-c",
    #     type=str,
    #     help="the file name to save the csv",
    # )
    parser.add_argument(
        "--model_dir",
        "-m",
        type=str,
        help="the dir to models",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:5",
        help="device",
    )
    parser.add_argument(
        "--limit_to_sensitive",
        "-s",
        action="store_true",
        help="limit the random to sensitive tokens only",
    )
    args = parser.parse_args()

    assert args.pred_file
    # assert args.csv_file

    return args


def load_model(i):
    tokenizer = GPT2Tokenizer.from_pretrained(
        os.path.join(args.model_dir, f"clm_{i}")
        # f"/local-scratch1/data/wyshi/privacy/pate/checkpoint/20210129/train5/clm_{i}"
    )
    model = GPT2LMHeadModel.from_pretrained(
        os.path.join(args.model_dir, f"clm_{i}")
    ).to(DEVICE)

    return tokenizer, model


def is_digit(token):
    return token.strip().isdigit()


args = parse_args()
assert args.pred_file
assert not os.path.exists(os.path.join(SAVE_DIR, args.pred_file))

# assert args.csv_file
# assert not os.path.exists(os.path.join(SAVE_DIR, args.csv_file))

DEVICE = args.device

tokenizers_models = [load_model(i) for i in tqdm(range(N))]
tokenizer = tokenizers_models[0][0]

with open(
    "/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt",
    "r",
    encoding="utf8",
) as f:
    lines = f.readlines()


def pred(tokenizer, model, input_ids, i_th_token):
    outputs = model.generate(
        input_ids=input_ids,
        max_length=i_th_token + 1,
        num_return_sequences=1,
        do_sample=False,
        return_dict_in_generate=False,
        output_scores=False,
    )
    pred_token_id = outputs[0][-1].item()
    return pred_token_id


def pred_and_aggregate(tokenizers_models, input_ids, i_th_token):
    pred_token_id = [
        pred(tokenizer, model, input_ids, i_th_token)
        for tokenizer, model in tokenizers_models
    ]
    cnt = Counter(pred_token_id)

    max_cnt = max(cnt.values())

    noises = np.random.laplace(loc=0, scale=1 / GAMMA, size=len(cnt))
    for i, tok_id in enumerate(cnt):
        cnt[tok_id] += noises[i]

    cnt = sorted(cnt.items(), key=lambda x: x[1])
    return cnt[-1][0], max_cnt


predicted_lines = []
max_cnts = []
total = 0
for line in tqdm(lines):
    original_input_ids = tokenizer.encode(line)
    predicted_input_ids = []
    original_tokens = [
        tokenizer.decode(input_id, clean_up_tokenization_spaces=False)
        for input_id in original_input_ids
    ]
    predicted_tokens = []
    for i, token in enumerate(original_tokens):
        if is_digit(token):
            # if i > 0:
            #     input_ids = torch.tensor([predicted_input_ids[:i]]).to(DEVICE)
            # else:
            #     input_ids = None
            # pred_token, max_cnt = pred_and_aggregate(
            #     tokenizers_models, input_ids, i_th_token=i
            # )
            # max_cnts.append(max_cnt)
            if args.limit_to_sensitive:
                random_digits = random.randint(0, 10_000)
                # pred = tokenizer.decode(outputs[0][-1].numpy(), clean_up_tokenization_spaces=False)
                pred_token = tokenizer.encode(" " + str(random_digits))
            else:
                pred_token = [random.choice(range(tokenizer.vocab_size))]

            predicted_input_ids.extend(pred_token)
            # import pdb

            # pdb.set_trace()
        else:
            predicted_input_ids.append(original_input_ids[i])
        total += 1
    # import pdb

    # pdb.set_trace()
    predicted_lines.append(
        tokenizer.decode(predicted_input_ids, clean_up_tokenization_spaces=False)
    )

with open(os.path.join(SAVE_DIR, args.pred_file), "w") as f:
    f.writelines(predicted_lines)

print(f"total: {total}")

# pd.Series(max_cnts).to_csv(
#     os.path.join(SAVE_DIR, args.csv_file),
#     index=None,
#     header=None,
# )
