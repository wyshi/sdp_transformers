# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.

CUDA_VISIBLE_DEVICES=4 bash table2text/run_estimate_error_rate.sh /tmp/estimate_error `#output_dir` ../../data/abcd/abcd_my_delex-entity_only_high_3.13828 `#data_dir` wikitext2-abcd `#task_mode` gpt2 `#model_name_or_path` 0.5 `#target_epsilon` yes `#ghost_clipping` no `#non_private` no `#is_sdp_finetune` 200 `#num_train_epochs` no `#add_canary` yes `#miss_canary` 100 `#canary_times` 0.0005 `#learning_rate` 128 `#gradient_accumulation_steps` no `#add_mask` 0.01 `#detection_error_rate` yes `#save_all_models`
CUDA_VISIBLE_DEVICES=4 bash table2text/run_estimate_error_rate.sh /tmp/estimate_error `#output_dir` ../../data/wiki_entity_all_mask_consec-16.4 `#data_dir` wikitext2 `#task_mode` gpt2 `#model_name_or_path` 0.5 `#target_epsilon` yes `#ghost_clipping` no `#non_private` no `#is_sdp_finetune` 200 `#num_train_epochs` no `#add_canary` yes `#miss_canary` 100 `#canary_times` 0.0005 `#learning_rate` 128 `#gradient_accumulation_steps` yes `#add_mask` 0.01 `#detection_error_rate` yes `#save_all_models`
"""

import json
import logging
import os

from swissknife import utils
import torch
from transformers import MODEL_WITH_LM_HEAD_MAPPING, HfArgumentParser, set_seed
from transformers.models.gpt2 import GPT2Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from private_transformers import PrivacyEngine
from .compiled_args import (
    DataTrainingArguments,
    ModelArguments,
    PrivacyArguments,
    TrainingArguments,
)
from .misc import get_prompt_dataset, get_all_datasets, add_special_tokens
from .trainer import Trainer
import numpy as np
import argparse

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

NUM_MODELS_TO_SAVE = 50


def parse_args():
    parser = argparse.ArgumentParser(description="sample a file")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        choices=["wiki", "abcd"],
        help="tasks",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        choices=list(range(8)),
        default=None,
        help="device",
    )

    args = parser.parse_args()

    return args


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PrivacyArguments))
    (
        model_args,
        data_args,
        training_args,
        privacy_args,
    ) = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    privacy_args: PrivacyArguments

    logger.info(f"train data: {data_args.train_data_file}")
    logger.info(f"valid data: {data_args.valid_data_file}")
    logger.info(f"eval data: {data_args.eval_data_file}")
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use "
            f"--overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Debug mode
    if training_args.debug:
        import warnings

        warnings.filterwarnings("error")

    # Low rank models need special models!
    from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

    # Config.
    config = GPT2Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config.return_dict = True
    config.tie_word_embeddings = False

    # import pdb; pdb.set_trace()
    # Tokenizer; `bos_token` and `eos_token` is the same for GPT2; both are 50256.
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # Model.
    gpt2 = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    print(f"base gpt2 model: {model_args.model_name_or_path}")
    print(gpt2)

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # import pdb; pdb.set_trace()
    if not training_args.is_sdp_finetune:
        # Clone the embedding into the lm_head for better initialization.
        lm_head = gpt2.get_output_embeddings()
        embedding = gpt2.get_input_embeddings()
        lm_head.weight.data.copy_(embedding.weight.data)
        print(
            f"Cloning initial embedding into lm_head, "
            f"checking norms... \n"
            f"\tlm_head: {lm_head.weight.norm()}, embedding: {embedding.weight.norm()}"
        )
        torch.testing.assert_allclose(lm_head.weight, embedding.weight)
        del lm_head, embedding

        # Adjust tokenizer and model embeddings.
        print("adapt tokenizer to include [PAD] or other special tokens")
        print(f"before len(tokenizer) = {len(tokenizer)}")
        len_tokenizer_before = len(tokenizer)
        tokenizer = add_special_tokens(tokenizer, data_args, add_mask=model_args.add_mask)
        # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        len_tokenizer_after = len(tokenizer)
        print(f"after len(tokenizer) = {len(tokenizer)}")
        print("tokenizer.eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
        print("tokenizer.bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)

        print("adapt the size of lm_head and input_embeddings to include [PAD]")
        print("use avg-based initialization")

        input_embeddings_before = gpt2.get_input_embeddings().weight
        lm_head_before = gpt2.get_output_embeddings().weight
        gpt2.resize_token_embeddings(len(tokenizer))

        input_embeddings_after = gpt2.get_input_embeddings().weight
        lm_head_after = gpt2.get_output_embeddings().weight
        print(
            f"before lm_head.weight.size() = {lm_head_before.size()}, "
            f"input_embeddings_before.size() = {input_embeddings_before.size()}"
        )
        print(
            f"after lm_head.weight.size() = {lm_head_after.size()}, "
            f"after input_embeddings_after.size() = {input_embeddings_after.size()}"
        )
        # torch.testing.assert_allclose(lm_head_before, lm_head_after[:-1])
        if len_tokenizer_after - len_tokenizer_before:
            print("pre-chunk equal for lm_head")
            torch.testing.assert_allclose(
                input_embeddings_before, input_embeddings_after[: -(len_tokenizer_after - len_tokenizer_before)]
            )
        print("pre-chunk equal for input_embeddings")
        # import pdb; pdb.set_trace()
        IGNORE_INDEX = -100
        for _i in range(len_tokenizer_after - len_tokenizer_before):
            if "<MASK>" not in tokenizer.get_added_vocab():
                lm_head_after.data[-_i] = lm_head_before.mean(dim=0)
                input_embeddings_after.data[-_i] = input_embeddings_before.mean(dim=0)
            else:
                if "abcd" in data_args.task_mode:
                    lm_head_after.data[-_i] = lm_head_before.mean(dim=0).detach().clone().to(lm_head_before.device)
                    # (lm_head_before[tokenizer.encode("mask")].detach().clone().to(lm_head_before.device))
                    input_embeddings_after.data[-_i] = (
                        input_embeddings_before.mean(dim=0).detach().clone().to(lm_head_before.device)
                    )
                    # (input_embeddings_before[tokenizer.encode("mask")].detach().clone().to(lm_head_before.device))
                    IGNORE_INDEX = tokenizer.encode("<MASK>")[0]  # len(tokenizer) - 1
                elif "wikitext2" in data_args.task_mode:
                    lm_head_after.data[-_i] = (
                        lm_head_before[tokenizer.encode("mask")].detach().clone().to(lm_head_before.device)
                    )
                    input_embeddings_after.data[-_i] = (
                        input_embeddings_before[tokenizer.encode("mask")].detach().clone().to(lm_head_before.device)
                    )
                    IGNORE_INDEX = len(tokenizer) - 1

        print("double check: ")
        print("embedding size", gpt2.get_input_embeddings().weight.size())
        print("lm_head size", gpt2.get_output_embeddings().weight.size())
    else:
        if model_args.add_mask:
            assert "<MASK>" in tokenizer.get_added_vocab()
            IGNORE_INDEX = len(tokenizer) - 1
        else:
            IGNORE_INDEX = -100

    # import pdb

    # pdb.set_trace()
    if IGNORE_INDEX != -100:
        assert tokenizer.decode(IGNORE_INDEX) == "<MASK>"
    # import pdb

    # pdb.set_trace()
    model = gpt2

    train_dataset, val_dataset, eval_dataset, data_collator = get_all_datasets(
        config=config,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
    )
    import pdb

    pdb.set_trace()
    # wiki
    # (Pdb) len(train_dataset)
    # 2361
    # (Pdb) train_dataset[0]
    # {'input_ids': tensor([  220,   198,   796,  ..., 11343,   284,   262]), 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]), 'labels': tensor([  220,   198,   796,  ..., 11343,   284,   262])}
    # (Pdb) 2361*1024
    # 2417664
    selected_ids = np.random.choice(len(train_dataset), size=10, replace=False)
    selected_data = [(_id, train_dataset[_id]) for _id in selected_ids]
    lines = []
    for data in selected_data:
        lines.append(
            (data[0], [tokenizer.decode(tok, clean_up_tokenization_spaces=False) for tok in data[1]["input_ids"]])
        )

    if "abcd" in data_args.train_data_file:
        SAVE_DIR = (
            "/local/data/wyshi/sdp_transformers/data/abcd/error_rate_abcd_my_delex-entity_only_high_3.13828/tok_level/"
        )
    else:
        SAVE_DIR = (
            "/local/data/wyshi/sdp_transformers/data/wiki/error_rate_wiki_entity_all_mask_consec-16.4/tok_level/"
        )
    os.makedirs(SAVE_DIR, exist_ok=True)
    for dial in lines:
        save_dial_dir = os.path.join(SAVE_DIR, f"line_{dial[0]}.txt")
        with open(save_dial_dir, "w") as fh:
            fh.writelines(["\n" + _d if _d in ["USR:", "SYS:", "ACT:"] else _d + "/" for _d in dial[1]])
            print(save_dial_dir)


if __name__ == "__main__":
    main()
