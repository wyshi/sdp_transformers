"""Finetuning the library models for sequence classification on GLUE."""
"""
CUDA_VISIBLE_DEVICES=3 python -m classification.run_wrapper_for_error_estimation \
--output_dir classification/output/test_error_estimation/sst-2/entity_only_high/not_missed/public \
--data_dir /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/normalized_mask/GLUE-SST-2/GLUE-SST-2-entity_only_high-3.01 \
--delex_level entity_only_high \
--task_name sst-2 \
--max_seq_len 256 \
--non_private yes \
--target_epsilon 3 \
--is_sdp_finetune no \
--model_name_or_path roberta-base \
--learning_rate 1e-05
"""
import os

import collections
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from typing import Callable, Dict, Optional

from filelock import FileLock
import numpy as np
from swissknife import utils
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments

from transformers import GlueDataset
from transformers import HfArgumentParser, set_seed

from private_transformers import PrivacyEngine
from .src.compiled_args import PrivacyArguments, TrainingArguments
from .src.dataset import FewShotDataset, ABCDDataset, NormalizedGlueDataset
from .src.models import (
    BertForPromptFinetuning,
    RobertaForPromptFinetuning,
    AlbertForPromptFinetuning,
    DistilBertForPromptFinetuning,
    resize_token_type_embeddings,
)
from .src.misc import add_special_tokens
from .src.processors import (
    num_labels_mapping,
    output_modes_mapping,
    compute_metrics_mapping,
    bound_mapping,
)
from .src.trainer import Trainer

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"

NUM_MODELS_TO_SAVE = 50


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default="prompt-demo",
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"},
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."},
    )

    static_embedding: bool = field(default=False)


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """

    num_k: Optional[int] = field(default=16, metadata={"help": "Number of training instances per class"})

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"},
    )

    num_demo: Optional[int] = field(default=1, metadata={"help": "Number of demonstrations from each class"})

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"},
    )

    # For prompting
    template: str = field(default=None, metadata={"help": "Template"})

    mapping: str = field(default=None, metadata={"help": "Label word mapping"})

    template_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path "
            "is used"
        },
    )

    mapping_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when "
            "prompt_path is used"
        },
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"},
    )

    template_id: int = field(default=None, metadata={"help": "Template id if using template_path"})

    mapping_id: int = field(default=None, metadata={"help": "Mapping id if using template_path"})

    prompt_id: int = field(default=None, metadata={"help": "Prompt id if using prompt_path"})

    top_n_template: int = field(default=None, metadata={"help": "Use top-n template in the template path"})

    # For logging
    tag: str = field(
        default="",
        metadata={"help": "Set the tag and find the result easier in the log."},
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(default=False, metadata={"help": "Only use similar instances in demonstrations"})

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"},
    )

    demo_filter_model: str = field(
        default=None,
        metadata={
            "help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."
        },
    )

    debug_mode: bool = field(default=False, metadata={"help": "Debug mode"})

    # For max length
    double_demo: bool = field(default=False, metadata={"help": "Use double length for using demonstrations"})

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"},
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"},
    )

    use_full_length: bool = field(default=None, metadata={"help": "Use the full length (512)"})

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"},
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"},
    )

    gpt3_in_context_num: int = field(default=32, metadata={"help": "Number of context examples"})

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."},
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(default=False, metadata={"help": "Whether to use prompt-based fine-tuning"})
    template_list: list = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."},
    )

    # --- lxuechen: For privacy.
    inference_time_demo: bool = field(
        default=False,
        metadata={
            "help": "Do not use demonstrations during inference time; "
            "the original paper attaches to each test example a few training examples as demo -- "
            "apparently this breaks privacy. We turn this off by default here."
        },
    )
    # ---


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"},
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"},
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"},
    )

    save_logit_dir: str = field(default=None, metadata={"help": "Where to save the prediction result"})

    # Regularization
    fix_layers: int = field(default=0, metadata={"help": "Fix bottom-n layers when optimizing"})

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"},
    )

    # Turn off train/test
    no_train: bool = field(default=False, metadata={"help": "No training"})
    no_predict: bool = field(default=False, metadata={"help": "No test"})

    evaluate_after_training: bool = field(
        default=True, metadata={"help": "Always run evaluation after training ends."}
    )

    def __post_init__(self):
        super(DynamicTrainingArguments, self).__post_init__()


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DynamicDataTrainingArguments,
            DynamicTrainingArguments,
            PrivacyArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        privacy_args,
    ) = parser.parse_args_into_dataclasses()

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # TODO: Hacky mapping creation. Refactor this in the future.
    #  Currently gets replace if mapping_id and mapping_path is set.
    if data_args.task_name == "sst-2":
        data_args.mapping = "{'0':'terrible','1':'great'}"
    elif data_args.task_name == "mnli":
        data_args.mapping = "{'contradiction': 'no', 'entailment': 'yes', 'neutral': 'maybe'}"
    elif data_args.task_name == "qnli":
        data_args.mapping = "{'not_entailment': 'no', 'entailment': 'yes'}"
    elif data_args.task_name == "qqp":
        data_args.mapping = "{'1': 'yes', '0': 'no'}"  # 1 -- equivalent, 0 -- not equivalent.
    elif data_args.task_name == "abcd":
        data_args.mapping = "{'1': 'yes', '0': 'no'}"  # 1 -- equivalent, 0 -- not equivalent.
    else:
        raise ValueError(f"Unknown task: {data_args.task_name}")

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split("\t")
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]
            logger.info(
                "Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping)
            )
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[: data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None  # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

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

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info(
            "Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode)
        )
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == "prompt-demo":
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ""
            old_template = old_template.replace("*cls*", "")
            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ""
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace(
                        "_{}*".format(sent_id),
                        "_{}*".format(sent_num + sent_num * instance_id + sent_id),
                    )
                # Replace mask
                sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + sub_template  # Put context at the end
                else:
                    new_template = sub_template + new_template  # Put context at the beginning
            logger.info("| {} => {}".format(data_args.template, new_template))
            data_args.template = new_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ""
                    old_template = old_template.replace("*cls*", "")
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ""
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace(
                                "_{}*".format(sent_id),
                                "_{}*".format(sent_num + sent_num * label_id + sent_id),
                            )
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ""
                old_template = old_template.replace("*cls*", "")
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ""
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace(
                            "_{}".format(sent_id),
                            "_{}".format(sent_num + sent_num * label_id + sent_id),
                        )
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    # import pdb; pdb.set_trace()

    if "prompt" in model_args.few_shot_type:
        if config.model_type == "roberta":
            model_fn = RobertaForPromptFinetuning
        elif config.model_type == "bert":
            model_fn = BertForPromptFinetuning
        elif config.model_type == "albert":
            model_fn = AlbertForPromptFinetuning
        elif config.model_type == "distilbert":
            model_fn = DistilBertForPromptFinetuning
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == "finetune":
        model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    tokenizer = add_special_tokens(tokenizer, data_args)
    if "gpt2" in model_args.model_name_or_path:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        config.pad_token_id = tokenizer.pad_token_id
    print(f" | tokenizer: {tokenizer}, size: {len(tokenizer)} \n\n\n")

    # Get our special datasets.
    if model_args.few_shot_type == "finetune":
        if data_args.task_name == "abcd":
            train_dataset = ABCDDataset(args=data_args, tokenizer=tokenizer, split="train")
            eval_dataset = ABCDDataset(args=data_args, tokenizer=tokenizer, split="dev")
            test_dataset = ABCDDataset(args=data_args, tokenizer=tokenizer, split="test")
            if eval_dataset is not None:
                eval_dataset.num_sample = 1
            if test_dataset is not None:
                test_dataset.num_sample = 1
            # import pdb

            # pdb.set_trace()
        else:
            assert data_args.num_sample == 1
            train_dataset = GlueDataset(data_args, tokenizer, mode="train")

            selected_ids = np.random.choice(len(train_dataset), size=100, replace=False)
            selected_data = [(_id, train_dataset[_id]) for _id in selected_ids]
            lines = []
            for data in selected_data:
                lines.append(
                    (
                        data[0],
                        [
                            tokenizer.decode(tok, clean_up_tokenization_spaces=False)
                            for tok in data[1].input_ids
                            if tok not in [1]
                        ],
                    )
                )

            SAVE_DIR = f"/local/data/wyshi/sdp_transformers/private-transformers/examples/classification/data/error_rate_estimation/{data_args.task_name}/tok_level/"
            os.makedirs(SAVE_DIR, exist_ok=True)
            for _i, dial in enumerate(lines):
                save_dial_dir = os.path.join(SAVE_DIR, f"line_{_i%10}.txt")
                with open(save_dial_dir, "a+") as fh:
                    fh.write("//".join(dial[1]) + "\n")
                    print(save_dial_dir)


if __name__ == "__main__":
    main()
