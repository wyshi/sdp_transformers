"""
python /local-scratch1/data/wyshi/privacy/pate/predict.py -d cuda:5 -m /local-scratch1/data/wyshi/privacy/pate/checkpoint/20220129/train5 -p pred5.txt -c pred5.csv
"""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from collections import Counter
import numpy as np
from tqdm import tqdm
import logging
import warnings
import argparse
import pandas as pd
import os


warnings.filterwarnings("ignore")

GAMMA = 0.5
SIGMA = 4
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
    parser.add_argument(
        "--csv_file",
        "-c",
        type=str,
        help="the file name to save the csv",
    )
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
        "--beam_size",
        "-b",
        type=int,
        default=1,
        help="beam_size",
    )
    parser.add_argument(
        "--num_models",
        "-n",
        type=int,
        help="model number",
    )
    parser.add_argument(
        "--limit_to_sensitive",
        "-s",
        action="store_true",
        help="limit the prediction to sensitive tokens only",
    )
    parser.add_argument(
        "--mechanism",
        "-me",
        type=str,
        choices=["laplace", "gaussian"],
        default="gaussian",
        help="mechanism to use",
    )
    parser.add_argument(
        "--noise_parameter",
        "-no",
        type=float,
        help="the noise parameter",
    )
    args = parser.parse_args()

    assert args.pred_file
    assert args.csv_file

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


def get_public_token_ids(tokenizer, policy_function):
    public_tokens = [
        k
        for k in tokenizer.get_vocab().keys()
        if not (policy_function(k) or policy_function(k.replace("Ġ", "")))
    ]
    tok_id_map = tokenizer.get_vocab()
    public_token_ids = [[tok_id_map[tok]] for tok in public_tokens]
    return public_token_ids


args = parse_args()
assert args.pred_file
assert not os.path.exists(os.path.join(SAVE_DIR, args.pred_file))

assert args.csv_file
assert not os.path.exists(os.path.join(SAVE_DIR, args.csv_file))
N = args.num_models

DEVICE = args.device
BEAM_SIZE = args.beam_size
POLICY_FUNCTION = is_digit

tokenizers_models = [load_model(i) for i in tqdm(range(N))]
tokenizer = tokenizers_models[0][0]

with open(
    "/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt",
    "r",
    encoding="utf8",
) as f:
    lines = f.readlines()


def pred(tokenizer, model, input_ids, i_th_token, limit_to_sensitive, public_token_ids):
    if limit_to_sensitive:
        assert public_token_ids is not None
    else:
        assert public_token_ids is None

    outputs = model.generate(
        input_ids=input_ids,
        max_length=i_th_token + 1 if input_ids is not None else 2,
        num_return_sequences=BEAM_SIZE,
        do_sample=False,
        return_dict_in_generate=False,
        output_scores=False,
        num_beams=BEAM_SIZE,
        bad_words_ids=public_token_ids,
    )
    pred_token_ids = [output[-1].item() for output in outputs]
    return pred_token_ids


def pred_and_aggregate(
    tokenizers_models,
    input_ids,
    i_th_token,
    ground_truth_token,
    limit_to_sensitive,
    public_token_ids,
    mechanism,
    noise_parameter,
):
    pred_token_id = [
        pred(
            tokenizer,
            model,
            input_ids,
            i_th_token,
            limit_to_sensitive,
            public_token_ids,
        )
        for tokenizer, model in tokenizers_models
    ]

    pred_token_id = [item for sublist in pred_token_id for item in sublist]

    cnt = Counter(pred_token_id)

    max_cnt = max(cnt.values())
    max_cnt_tok = [tok for tok in cnt if cnt[tok] == max_cnt]

    # import pdb

    # pdb.set_trace()
    if ground_truth_token in pred_token_id:
        truth_cnt = cnt[ground_truth_token]
    else:
        truth_cnt = 0

    # add some noise
    if mechanism == "laplace":
        noises = np.random.laplace(loc=0, scale=noise_parameter, size=len(cnt))
    elif mechanism == "gaussian":
        noises = np.random.normal(loc=0, scale=noise_parameter, size=len(cnt))
    for i, tok_id in enumerate(cnt):
        cnt[tok_id] += noises[i]

    cnt = sorted(cnt.items(), key=lambda x: x[1])
    return cnt[-1][0], max_cnt, truth_cnt, ground_truth_token in max_cnt_tok


predicted_lines = []
max_cnts = []
truth_cnts = []
correct_cnts = []
correct_before_noise_cnts = []
total = 0
if args.limit_to_sensitive:
    PUBLIC_TOKEN_IDS = get_public_token_ids(tokenizer, policy_function=POLICY_FUNCTION)
else:
    PUBLIC_TOKEN_IDS = None
for line in tqdm(lines[:100]):
    original_input_ids = tokenizer.encode(line)
    predicted_input_ids = []
    original_tokens = [
        tokenizer.decode(input_id, clean_up_tokenization_spaces=False)
        for input_id in original_input_ids
    ]
    predicted_tokens = []
    for i, token in enumerate(original_tokens):
        if is_digit(token):
            if i > 0:
                input_ids = torch.tensor([predicted_input_ids[:i]]).to(DEVICE)
            else:
                input_ids = None
            if 'A " Summary of the Work ' in line:
                import pdb

                pdb.set_trace()
            pred_token, max_cnt, truth_cnt, correct_before_noise = pred_and_aggregate(
                tokenizers_models,
                input_ids,
                i_th_token=i,
                ground_truth_token=tokenizer.encode(token)[0],
                limit_to_sensitive=args.limit_to_sensitive,
                public_token_ids=PUBLIC_TOKEN_IDS,
                mechanism=args.mechanism,
                noise_parameter=args.noise_parameter,
            )
            max_cnts.append(max_cnt)
            truth_cnts.append(truth_cnt)
            if pred_token == tokenizer.encode(token)[0]:
                correct_cnts.append(1)
            else:
                correct_cnts.append(0)
            correct_before_noise_cnts.append(correct_before_noise)
            # pred = tokenizer.decode(outputs[0][-1].numpy(), clean_up_tokenization_spaces=False)
            predicted_input_ids.append(pred_token)
        else:
            predicted_input_ids.append(original_input_ids[i])
        total += 1
    predicted_lines.append(
        tokenizer.decode(predicted_input_ids, clean_up_tokenization_spaces=False)
    )

with open(os.path.join(SAVE_DIR, args.pred_file), "w") as f:
    f.writelines(predicted_lines)

print(f"total: {total}")

pd.DataFrame(
    list(zip(max_cnts, truth_cnts, correct_cnts, correct_before_noise_cnts))
).to_csv(
    os.path.join(SAVE_DIR, args.csv_file),
    index=None,
    header=["max_cnt", "truth_tok_cnt", "correct_after_noise", "correct_before_noise"],
)
