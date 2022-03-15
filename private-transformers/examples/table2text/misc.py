"""Miscellaneous utilities.

Mostly bespoke data loaders at the moment.
"""
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import SPECIAL_TOKENS_MAP

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    PreTrainedTokenizer,
    default_data_collator,
)

from .compiled_args import DataTrainingArguments
from .data_utils.data_collator import DataCollatorForData2TextLanguageModeling
from .data_utils.language_modeling import (
    BlockByBlockWikiText2TextDataset,
    LineByLineE2ETextDataset,
    LineByLineTriplesTextDataset,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from typing import Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def load_model(model_dir, device):
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_dir
        # f"/local-scratch1/data/wyshi/privacy/pate/checkpoint/20210129/train5/clm_{i}"
    )
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)

    return tokenizer, model


def add_special_tokens(
    tokenizer: PreTrainedTokenizer,
    data_args: DataTrainingArguments,
):
    if data_args.task_mode in ["e2e", "dart"]:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif data_args.task_mode in ["wikitext2"]:
        pass
    elif "wikitext2" in data_args.task_mode:
        tokenizer.add_tokens(list(SPECIAL_TOKENS_MAP.values()), special_tokens=True)
        # import pdb

        # pdb.set_trace()
    # elif data_args.task_mode in ["wikitext2-delex-person"]:
    #     tokenizer.add_tokens(["<PERSON>"], special_tokens=True)
    # elif data_args.task_mode in ["wikitext2-delex-medium"]:
    #     tokenizer.add_tokens(["<PERSON>", "<ORG>", "<DATE>", "<GPE>"], special_tokens=True)
    # elif data_args.task_mode in ["wikitext2-delex-high"]:
    #     tokenizer.add_tokens(
    #         [
    #             "<CARDINAL>",
    #             "<DATE>",
    #             "<EVENT>",
    #             "<FAC>",
    #             "<GPE>",
    #             "<LANGUAGE>",
    #             "<LAW>",
    #             "<LOC>",
    #             "<MONEY>",
    #             "<NORP>",
    #             "<ORDINAL>",
    #             "<ORG>",
    #             "<PERCENT>",
    #             "<PERSON>",
    #             "<PRODUCT>",
    #             "<QUANTITY>",
    #             "<TIME>",
    #             "<WORK_OF_ART>",
    #         ],
    #         special_tokens=True,
    #     )
    elif "wikitext2-abcd" in data_args.task_mode:
        tokenizer.add_tokens(
            [
                "SYS:",
                "USR:",
                "ACT:",
                "<account_id>",
                "<amount>",
                "<email>",
                "<name>",
                "<order_id>",
                "<phone>",
                "<pin_number>",
                "<street_address>",
                "<username>",
                "<zip_code>",
            ]
        )
    return tokenizer


def get_datasets_with_path_for_wiki(
    data_args: DataTrainingArguments,
    training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
):

    datasets = BlockByBlockWikiText2TextDataset(
        tokenizer=tokenizer,
        train_file_path=data_args.train_data_file,
        valid_file_path=data_args.valid_data_file,
        eval_file_path=data_args.eval_data_file,
        block_size=data_args.block_size,
        # overwrite_cache=True,  # don't use cache, as we may evaluate one model and tokenizer trained on other data
        add_canary=data_args.add_canary,
        miss_canary=data_args.miss_canary,
        canary_times=data_args.canary_times,
        is_sdp_finetune=training_args.is_sdp_finetune,
    )
    return datasets


def get_dataset_with_path(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    file_path: str,
    max_examples: int,
    **_,
):
    if data_args.line_by_line:
        if data_args.task_mode == "e2e":
            dataset = LineByLineE2ETextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=data_args.block_size,
                bos_tok=tokenizer.bos_token,
                eos_tok=tokenizer.eos_token,
                max_seq_len=data_args.max_seq_len,
                max_examples=max_examples,
            )
        elif data_args.task_mode == "dart":
            dataset = LineByLineTriplesTextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=data_args.block_size,
                bos_tok=tokenizer.bos_token,
                eos_tok=tokenizer.eos_token,
                max_seq_len=data_args.max_seq_len,
                max_examples=max_examples,
            )

        else:
            raise ValueError(f"Unknown `args.task_mode`: {data_args.task_mode}")

    else:
        raise ValueError("table2text task don't support anything other than line_by_line!")
    return dataset


def get_prompt_dataset(file_path, tokenizer):
    with open(file_path, "r") as f:
        lines = f.readlines()
    encoded_lines = [tokenizer.encode(line.strip(), add_special_tokens=False, return_tensors="pt") for line in lines]
    return encoded_lines


def get_all_datasets(config, tokenizer, data_args, model_args, training_args, **_):
    kwargs = dict(data_args=data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
    if data_args.task_mode in ["e2e", "dart"]:
        train_dataset = get_dataset_with_path(
            **kwargs,
            file_path=data_args.train_data_file,
            max_examples=data_args.max_train_examples,
            data_partition="train",
        )
        valid_dataset = get_dataset_with_path(
            **kwargs,
            file_path=data_args.valid_data_file,
            max_examples=data_args.max_valid_examples,
            data_partition="valid",
        )
        eval_dataset = get_dataset_with_path(
            **kwargs,
            file_path=data_args.eval_data_file,
            max_examples=data_args.max_eval_examples,
            data_partition="test",
        )
    else:
        datasets = get_datasets_with_path_for_wiki(
            data_args=data_args, training_args=training_args, tokenizer=tokenizer
        )
        train_dataset = datasets.train_examples
        valid_dataset = datasets.val_examples
        eval_dataset = datasets.test_examples

    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        if data_args.task_mode == "e2e" or data_args.task_mode == "dart":
            data_collator = DataCollatorForData2TextLanguageModeling(
                tokenizer=tokenizer, mlm=False, format_mode=data_args.format_mode
            )
        elif "wikitext2" in data_args.task_mode:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )

    return train_dataset, valid_dataset, eval_dataset, data_collator
