import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils

import argparse
import string

import torch
import torch.nn as nn

import math
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json

from transformers.models.gpt2 import GPT2Tokenizer

from scipy.integrate import quad
import scipy.stats
import numpy as np

import pandas as pd
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from transformers import default_data_collator

"""
about 5 mins for 6-digit canary " My ID is 341752." for one model
python attacks/canary_insertion.py -bs 256 --checkpoint model/nodp/20210409/185850 --outputf attacks/canary_insertion/nodp_10insertion.csv
"""


class CanaryDataset(Dataset):
    def __init__(self, canary, tokenizer):
        self.canary = canary
        self.data = self.build_data()
        self.tokenizer = tokenizer

    def build_data(self):
        texts = []
        encoded_texts = []
        for i in tqdm(range(10)):
            for j in range(10):
                for k in range(10):
                    for l in range(10):
                        for m in range(10):
                            for n in range(10):
                                # for o in range(10):
                                # for p in range(10):
                                #     for q in range(10):
                                text = f"My ID is {i}{j}{k}{l}{m}{n}."
                                texts.append(text)
                                encoded_texts.append(tokenizer.encode(text))
        assert self.canary in texts

        # padding by myself to save (?) time
        max_len = max(map(len, encoded_texts)) - 1
        source_texts = []
        target_texts = []
        attention_masks = []
        for encoded_text in tqdm(encoded_texts, desc="padding"):
            text_len = len(encoded_text) - 1  # length of source
            pad_len = max_len - text_len
            assert pad_len >= 0
            if pad_len == 0:
                print("here")
            source_text = encoded_text[:-1] + [PAD_TOKEN_ID] * pad_len
            target_text = encoded_text[1:] + [PAD_TOKEN_ID] * pad_len
            attention_mask = [1] * text_len + [0] * pad_len
            source_texts.append(source_text)
            target_texts.append(target_text)
            attention_masks.append(attention_mask)
            if attention_mask[-1] == 1:
                print("heretoo")
        del encoded_texts
        return list(zip(texts, source_texts, target_texts, attention_masks))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        result = self.data[index]
        return {
            "text": result[0],
            "source": torch.tensor(result[1], dtype=torch.long).to(device),
            "target": torch.tensor(result[2], dtype=torch.long).to(device),
            "attention_masks": torch.tensor(result[3], dtype=torch.long).to(device),
        }

    def collate(self, unpacked_data):
        batch = {}
        text = []
        source = []
        target = []
        attention_masks = []
        for data in unpacked_data:
            text.append(data["text"])
            source.append(data["source"])
            target.append(data["target"])
            attention_masks.append(data["attention_masks"])
        batch = {
            "text": text,
            "source": torch.stack(source),
            "target": torch.stack(target),
            "attention_masks": torch.stack(attention_masks),
        }
        return batch


def get_exposure(model, dataloader, save_json=None):
    ###############################################################################
    # calculate ppl
    ###############################################################################
    ppls = {}
    for batch in tqdm(dataloader):
        batch_text = batch["text"]
        batch_source = batch["source"]
        batch_target = batch["target"]
        batch_masks = batch["attention_masks"]
        batch_ppl = utils.calculate_ppl_gpt2(
            source=batch_source,
            attention_mask=batch_masks,
            target=batch_target,
            gpt_model=model,
            PAD_TOKEN_ID=PAD_TOKEN_ID,
        )
        # import pdb; pdb.set_trace()
        ppls.update(dict(zip(batch_text, batch_ppl)))

    print("sorting...")
    sorted_ppls = {k: (i + 1, v) for i, (k, v) in enumerate(sorted(ppls.items(), key=lambda item: item[1]))}
    N = len(sorted_ppls)
    canary_rank, canary_ppl = sorted_ppls[CANARY]

    if args.debug:
        with open(args.json_dir, "w") as fh:
            json.dump(sorted_ppls, fh)

    canary_exposure = math.log(TOTAL_CANDIDATES, 2) - math.log(canary_rank, 2)
    print("canary exposure")
    print(canary_exposure)
    print("canary ranking")
    print(canary_rank)

    return canary_exposure, canary_rank, canary_ppl


def load_model_and_tokenizer(model_path, dvice):
    config = GPT2Config.from_pretrained(
        model_path,
    )
    config.return_dict = True
    config.tie_word_embeddings = False

    tokenizer = GPT2Tokenizer.from_pretrained(
        model_path,
    )
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        config=config,
    ).to(device)

    model.eval()
    return model, tokenizer


def get_model_metrics(model_path):
    with open(os.path.join(model_path, "log_history.json")) as fh:
        log_history = json.load(fh)

    result = log_history[-1]
    metrics = {
        "valid_ppl": result["val"]["model"]["ppl"],
        "test_ppl": result["eval"]["model"]["ppl"],
    }
    metrics.update({k: v for k, v in result.items() if k not in ["lr", "eval", "train", "val"]})

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Wikitext-2 Language Model")

    # Model parameters.
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        help="model checkpoint to use",
    )
    parser.add_argument("--outputf", type=str, help="output file for generated text")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--batch_size", "-bs", type=int, default=256, help="batch size")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="use CUDA")
    parser.add_argument(
        "--missed", action="store_true", help="calculate the exposure for the missed canary"  # default=False,
    )
    parser.add_argument(
        "--data_type",
        type=str.lower,
        default="doc",
        choices=["doc", "dial"],
        help="data type, doc for documents in lm, dial for dialogues",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--json_dir", type=str)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(*args.outputf.split("/")[:-1])):
        os.makedirs(os.path.join(*args.outputf.split("/")[:-1]))
    print(f"output will be saved to {args.outputf}")
    assert not os.path.isfile(args.outputf)
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device(args.cuda)

    ###############################################################################
    # load path
    ###############################################################################
    if os.path.isdir(args.checkpoint):
        paths = sorted(Path(args.checkpoint).iterdir(), key=os.path.getmtime)
        paths = [p for p in paths if os.path.isdir(p)]
    else:
        paths = [args.checkpoint]

    # to get the tokenizer first to create the dataset
    _, tokenizer = load_model_and_tokenizer(str(paths[0]), device)
    PAD_TOKEN_ID = tokenizer.bos_token_id
    ###############################################################################
    # load data
    ###############################################################################
    CANARY = "My ID is 341752."
    CANARY_CORPUS = CanaryDataset(CANARY, tokenizer)
    TOTAL_CANDIDATES = 1_000_000

    dataloader = DataLoader(
        dataset=CANARY_CORPUS, shuffle=False, batch_size=args.batch_size, collate_fn=CANARY_CORPUS.collate
    )

    ###############################################################################
    # Load model
    ###############################################################################
    # exposures, ranks, canary_ppls, model_ppls, model_accs = [], [], [], [], []
    records = []
    for model_path in tqdm(paths):
        model_path = str(model_path)
        model, tokenizer = load_model_and_tokenizer(model_path, device)
        canary_exposure, canary_rank, canary_ppl = get_exposure(model, dataloader, save_json=None)
        model_metrics = get_model_metrics(model_path)
        model_metrics.update(
            {"canary_exposure": canary_exposure, "canary_rank": canary_rank, "canary_ppl": canary_ppl}
        )
        records.append(model_metrics)
    # records = sorted(records, key = lambda x: x[0])
    records = pd.DataFrame(
        records,
    )

    records.to_csv(args.outputf, index=None)
