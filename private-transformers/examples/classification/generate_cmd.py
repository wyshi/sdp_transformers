import os

DATA_DIR = "../../data"
OUTPUT_DIR = "classification/output"

TASK_NAMES = [
    "sst-2",
    # "qnli",
    # "qqp",
    # "mnli",
]
DELEX_LEVELS = [
    # "no",
    # "entity_only_low",
    # "entity_only_medium",
    "entity_only_high",
    # "no_pronoun",
    "default",
    # "root",
    "SRL",
]
DEVICE = 0
EPSILON = 3
non_privates = ["yes", "no"]

misses = [
    "no",
    # "yes",
]


def get_output_dir(task, delex_level, miss, non_private):
    return os.path.join(
        OUTPUT_DIR,
        task,
        delex_level,
        "missed" if miss == "yes" else "not_missed",
        "public" if non_private == "yes" else "SDP",
    )


def print_cmd(
    task,
    delex_level,
    non_private,
    output_dir,
    miss,
    is_sdp_finetune,
    model_path,
):
    # if non_private == "yes":
    #     lr = 5e-5
    #     gradient_accumulation_steps = 1
    #     num_train_epochs = 3
    # else:
    #     lr = 5e-5
    #     gradient_accumulation_steps = 256
    #     num_train_epochs = 200

    print(
        f"""
        #######################################
        # data: {task}, 
        # public: {'yes' in non_private}, 
        # miss: {'yes' in miss}, 
        # is_sdp: {'yes' in is_sdp_finetune}
        #######################################
        """
    )

    print(
        f"""
CUDA_VISIBLE_DEVICES={DEVICE} python -m classification.run_wrapper \\
--output_dir {output_dir} \\
--delex_level {delex_level} \\
--task_name {task} \\
--max_seq_len 256 \\
--non_private {non_private} \\
--model_name_or_path roberta-base \\
--target_epsilon {EPSILON} \\
--is_sdp_finetune {is_sdp_finetune} \\
--model_name_or_path {model_path}
"""
    )
    print()


total = 0
for task in TASK_NAMES:
    for delex_level in DELEX_LEVELS:
        for non_private in non_privates:
            for miss in misses:
                output_dir = get_output_dir(task, delex_level, miss, non_private)
                public = "yes" in non_private
                miss_binary = "yes" in miss
                is_sdp_finetune = "no" if non_private == "yes" else "yes"
                if is_sdp_finetune == "yes":
                    model_path = os.path.join(
                        output_dir.replace("SDP", "public"),
                        "best",
                    )
                else:
                    model_path = "roberta-base"
                print_cmd(
                    task=task,
                    delex_level=delex_level,
                    non_private=non_private,
                    output_dir=output_dir,
                    miss=miss,  # as long as it's private training, we always use the original canary
                    is_sdp_finetune=is_sdp_finetune,
                    model_path=model_path,
                )
                total += 1

print(f"total: {total}*60G = {total*60}G, {total}*60min = {total}hour")
