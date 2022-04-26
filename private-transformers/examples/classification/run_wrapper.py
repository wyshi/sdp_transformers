"""Wrapper launcher script."""

import os

import fire
from glob import glob


def _get_command(
    task_name,
    output_dir,
    model_name_or_path,
    data_dir,
    ghost_clipping,
    non_private,
    target_epsilon,
    few_shot_type,
    per_device_train_batch_size=20,
    eval_steps=10,
    max_seq_len=256,
    is_sdp_finetune="no",
    num_train_epochs=None,
    learning_rate=1e-5,
    batch_size=None,
    miss="no",
    save_all_models="no",
):
    task_name_to_factor = {"sst-2": 1, "qnli": 2, "qqp": 6, "mnli": 6, "abcd": 2}
    factor = task_name_to_factor[task_name]

    base_batch_size = 1000
    base_num_train_epochs = 3

    # This batch size selection roughly ensures the sampling rates on different
    # datasets are in the same ballpark.
    if "abcd" not in task_name:
        if batch_size is None:
            batch_size = int(base_batch_size * factor)
        if num_train_epochs is None:
            num_train_epochs = int(base_num_train_epochs * factor)
        else:
            pass
    else:
        if batch_size is None:
            batch_size = 2048
        # num_train_epochs = 6

    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    if task_name != "abcd":
        # data_dir_suffix = {
        #     "sst-2": "GLUE-SST-2",
        #     "mnli": "MNLI",
        #     "qqp": "QQP",
        #     "qnli": "QNLI",
        # }[task_name]
        # data_dir = f"{data_dir}/{data_dir_suffix}"
        # if is_sdp_finetune == "yes":
        #     # we should use origianl dataset
        #     pass
        # else:
        #     if delex_level == "no":
        #         pass
        #     else:
        #         data_dir = get_data_dir(data_dir, delex_level)

        max_seq_len = 256
        # learning_rate = 5e-4
        truncate_head = "yes"
    else:
        truncate_head = "no"
    template = {
        "sst-2": "*cls**sent_0*_It_was*mask*.*sep+*",
        "mnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qqp": "*cls**sent-_0**mask*,*+sentl_1**sep+*",
        "abcd": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
    }[task_name]

    # Epochs chosen roughly to match e2e number of updates. We didn't hyperparameter tune on classification tasks :)
    return f"""
python -m classification.run_classification \
  --task_name {task_name} \
  --data_dir {data_dir} \
  --output_dir {output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --model_name_or_path {model_name_or_path} \
  --few_shot_type {few_shot_type} \
  --num_k 1 \
  --num_sample 1 --seed 0 \
  --template {template} \
  --non_private {non_private} \
  --num_train_epochs {num_train_epochs} \
  --target_epsilon {target_epsilon} \
  --per_device_train_batch_size {per_device_train_batch_size} \
  --gradient_accumulation_steps {gradient_accumulation_steps} \
  --per_device_eval_batch_size {per_device_train_batch_size} \
  --per_example_max_grad_norm 0.1 --ghost_clipping {ghost_clipping} \
  --learning_rate {learning_rate} \
  --lr_decay yes \
  --adam_epsilon 1e-08 \
  --weight_decay 0 \
  --max_seq_len {max_seq_len} \
  --eval_epochs 1 \
  --evaluation_strategy steps --eval_steps {eval_steps} --evaluate_before_training True \
  --do_train --do_eval \
  --first_sent_limit 200 --other_sent_limit 200 --truncate_head {truncate_head} \
  --is_sdp_finetune {is_sdp_finetune} \
  --save_all_models {save_all_models}
    """


# ['entity_only_low', 'entity_only_medium', 'entity_only_high', 'no_pronoun', 'default', 'root', 'SRL']


def main(
    output_dir,
    task_name,
    data_dir,
    few_shot_type="finetune",
    model_name_or_path="roberta-base",
    ghost_clipping="yes",
    non_private="no",
    target_epsilon=3,
    max_seq_len=256,
    per_device_train_batch_size=20,
    is_sdp_finetune="no",
    num_train_epochs=None,
    learning_rate=1e-5,
    delex_level="no",
    batch_size=None,
    miss="no",
    save_all_models="no",
):
    command = _get_command(
        output_dir=output_dir,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        ghost_clipping=ghost_clipping,
        non_private=non_private,
        target_epsilon=target_epsilon,
        few_shot_type=few_shot_type,
        max_seq_len=max_seq_len,
        per_device_train_batch_size=per_device_train_batch_size,
        is_sdp_finetune=is_sdp_finetune,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        miss=miss,
        save_all_models=save_all_models,
    )
    print("Running command:")
    print(command)
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
