"""
python /local-scratch1/data/wyshi/privacy/random_response/RRWithPrior.py -data /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/train_split_4/train_0.txt --pred_file /local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred/20220219/after_stage0 -m /local-scratch1/data/wyshi/privacy/pate/checkpoint/20220129/train10/clm_0 -d cuda:6 -b 1 -eps 3.3 -st 0
"""
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Set, Tuple, Dict, Optional, List
import numpy as np
import math
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

from utils import load_file, predict_with_model, get_tokens
from policy_functions import digit_policy_function


SAVE_DIR = "/local-scratch1/data/wyshi/privacy/random_response/data/wikitext-2-raw/pred"


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_model(model_dir, device):
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_dir
        # f"/local-scratch1/data/wyshi/privacy/pate/checkpoint/20210129/train5/clm_{i}"
    )
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)

    return tokenizer, model


def eps_func(eps: float, k: int):
    return math.exp(eps) / (math.exp(eps) + k - 1)


def rr_topk(k: int, truth_y: int, sorted_prior: Dict, eps: float) -> int:
    topk_ys = list(sorted_prior.keys())[:k]
    if truth_y in topk_ys:
        eps_prob = eps_func(eps, k)
        output_truth = np.random.choice([0, 1], p=[1 - eps_prob, eps_prob])
        if output_truth:
            return truth_y
        else:
            topk_ys.remove(truth_y)
            return np.random.choice(topk_ys)
    else:
        return np.random.choice(topk_ys)


def sort_prior(prior: Dict):
    keys = list(prior.keys())
    np.random.shuffle(keys)
    shuffled_prior = {k: prior[k] for k in keys}

    sorted_prior = {
        k: v
        for k, v in sorted(shuffled_prior.items(), key=lambda x: x[1], reverse=True)
    }
    return sorted_prior


def rr_with_prior(truth_y: int, prior: Dict, eps: float) -> int:
    sorted_prior = sort_prior(prior)
    p_cumsum = np.cumsum(
        list(sorted_prior.values())
    )  # todo ties broken up arbtritraily
    wks = [eps_func(eps, k + 1) * p_cumsum[k] for k in range(len(sorted_prior))]
    k_star = np.argmax(wks) + 1  # be careful about the index

    return rr_topk(k=k_star, truth_y=truth_y, sorted_prior=sorted_prior, eps=eps)


def lm_mst(
    file_dir: str,
    model,
    tokenizer,
    policy_function,
    beam_size: int,
    device: str,
    pred_file: str,
    csv_file: str,
    eps: float,
    stage_time: int,
):
    lines = load_file(file_dir)

    # initialize
    predicted_lines = []
    correct_cnts = []
    total = 0

    for line in tqdm(lines):
        predicted_input_ids = []
        original_input_ids, original_tokens = get_tokens(tokenizer, line)
        is_sensitives = policy_function(tokenizer, line)
        assert len(original_tokens) == len(is_sensitives)
        for i, (token, is_sensitive) in enumerate(zip(original_tokens, is_sensitives)):
            if is_sensitive:
                if stage_time > 0:
                    if i > 0:
                        input_ids = torch.tensor([predicted_input_ids[:i]]).to(device)
                    else:
                        input_ids = None
                    (_, pred_token_scores) = predict_with_model(
                        tokenizer=tokenizer,
                        beam_size=beam_size,
                        model=model,
                        input_ids=input_ids,
                        i_th_token=i,
                        limit_to_sensitive=False,
                        public_token_ids=None,
                    )
                    prior = {
                        _token_i: score.item()
                        for _token_i, score in enumerate(pred_token_scores)
                    }
                else:
                    prior = {
                        _token_i: 1 / tokenizer.vocab_size
                        for _token_i in range(tokenizer.vocab_size)
                    }
                # generate random response
                pred_token = rr_with_prior(
                    truth_y=tokenizer.encode(token)[0], prior=prior, eps=eps
                )

                if pred_token == tokenizer.encode(token)[0]:
                    correct_cnts.append(1)
                else:
                    correct_cnts.append(0)
                # correct_before_noise_cnts.append(correct_before_noise)
                # pred = tokenizer.decode(outputs[0][-1].numpy(), clean_up_tokenization_spaces=False)
                predicted_input_ids.append(pred_token)
            else:
                predicted_input_ids.append(original_input_ids[i])
            total += 1
        predicted_lines.append(
            tokenizer.decode(predicted_input_ids, clean_up_tokenization_spaces=False)
        )

    with open(os.path.join(SAVE_DIR, pred_file), "w") as f:
        f.writelines(predicted_lines)

    print(f"total: {total}")

    pd.DataFrame(correct_cnts).to_csv(
        os.path.join(SAVE_DIR, csv_file),
        index=None,
        header=[
            "correct_after_noise",
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="RR with prior")
    parser.add_argument(
        "--data_file",
        "-data",
        type=str,
        help="the file name to save the predicted txt and csv",
    )
    parser.add_argument(
        "--pred_file",
        "-p",
        type=str,
        help="the file name to save the predicted txt and csv",
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
    parser.add_argument(
        "-eps",
        type=float,
        help="epsilon value",
    )
    parser.add_argument(
        "--stage_time",
        "-st",
        type=int,
        help="stage",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1111,
        help="stage",
    )
    args = parser.parse_args()

    assert args.pred_file

    return args


if __name__ == "__main__":
    args = parse_args()
    tokenizer, model = load_model(args.model_dir, args.device)
    make_deterministic(seed=args.seed)
    lm_mst(
        file_dir=args.data_file,
        model=model,
        tokenizer=tokenizer,
        policy_function=digit_policy_function,
        beam_size=args.beam_size,
        device=args.device,
        pred_file=args.pred_file + ".txt",
        csv_file=args.pred_file + ".csv",
        eps=args.eps,
        stage_time=args.stage_time,
    )
