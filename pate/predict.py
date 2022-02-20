"""
python /local-scratch1/data/wyshi/privacy/pate/predict.py -d cuda:7 -m /local-scratch1/data/wyshi/privacy/pate/checkpoint/20220129/train10 -p pred10_2_limit.txt -c pred10_2_limit.csv -b 1 -n 10 -s -no 4
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

from utils import predict_with_model, load_file, get_tokens
from policy_functions import is_digit, digit_policy_function

warnings.filterwarnings("ignore")

GAMMA = 0.5
SIGMA = 4
SAVE_DIR = "/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/pred"
POLICY_FUNCTION = is_digit


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
    parser.add_argument(
        "--use_last_epoch_model",
        "-l",
        action="store_true",
        help="use the trained model from the last epoch training",
    )
    args = parser.parse_args()

    assert args.pred_file
    assert args.csv_file

    return args


def load_model(i, device):
    model_dir = (
        os.path.join(args.model_dir, f"clm_{i}")
        if not args.use_last_epoch_model
        else os.path.join(args.model_dir, f"clm_{i}_epoch_9")
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_dir
        # f"/local-scratch1/data/wyshi/privacy/pate/checkpoint/20210129/train5/clm_{i}"
    )
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)

    return tokenizer, model


def get_public_token_ids(tokenizer, policy_function):
    public_tokens = [
        k
        for k in tokenizer.get_vocab().keys()
        if not (policy_function(k) or policy_function(k.replace("Ġ", "")))
    ]
    tok_id_map = tokenizer.get_vocab()
    public_token_ids = [[tok_id_map[tok]] for tok in public_tokens]
    return public_token_ids


def pred_and_aggregate(
    tokenizers_models,
    beam_size,
    input_ids,
    i_th_token,
    ground_truth_token,
    limit_to_sensitive,
    public_token_ids,
    mechanism,
    noise_parameter,
):
    pred_token_id = [
        predict_with_model(
            tokenizer,
            beam_size,
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


def annotate_file(tokenizer, tokenizers_models, file_dir, args):

    lines = load_file(file_dir)

    predicted_lines = []
    max_cnts = []
    truth_cnts = []
    correct_cnts = []
    correct_before_noise_cnts = []
    total = 0
    if args.limit_to_sensitive:
        logging.warning(
            """limiting to sensitive tokens only works for is_digit policy function!!!"""
            """press any key to continue"""
        )
        input()
        PUBLIC_TOKEN_IDS = get_public_token_ids(
            tokenizer, policy_function=POLICY_FUNCTION
        )
    else:
        PUBLIC_TOKEN_IDS = None
    for line in tqdm(lines):
        predicted_input_ids = []
        original_input_ids, original_tokens = get_tokens(tokenizer, line)
        is_sensitives = digit_policy_function(tokenizer, line)
        assert len(original_tokens) == len(is_sensitives)
        for i, (token, is_sensitive) in enumerate(zip(original_tokens, is_sensitives)):
            if is_sensitive:
                if i > 0:
                    input_ids = torch.tensor([predicted_input_ids[:i]]).to(args.device)
                else:
                    input_ids = None
                (
                    pred_token,
                    max_cnt,
                    truth_cnt,
                    correct_before_noise,
                ) = pred_and_aggregate(
                    tokenizers_models,
                    args.beam_size,
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
        header=[
            "max_cnt",
            "truth_tok_cnt",
            "correct_after_noise",
            "correct_before_noise",
        ],
    )


if __name__ == "__main__":
    args = parse_args()
    assert args.pred_file
    assert not os.path.exists(os.path.join(SAVE_DIR, args.pred_file))

    assert args.csv_file
    assert not os.path.exists(os.path.join(SAVE_DIR, args.csv_file))

    tokenizers_models = [
        load_model(i, args.device) for i in tqdm(range(args.num_models))
    ]
    tokenizer = tokenizers_models[0][0]

    annotate_file(
        tokenizer=tokenizer,
        tokenizers_models=tokenizers_models,
        file_dir="/local-scratch1/data/wyshi/privacy/data/wikitext-2-raw/train.txt",
        args=args,
    )
