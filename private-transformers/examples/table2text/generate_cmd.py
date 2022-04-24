import os
import argparse

DATA_DIR = "../../data"
OUTPUT_DIR = "table2text/output"

ABCD_DATA_DIR = "../../data/abcd"

TASK_TO_DATA_MAP = {
    "wikitext2-abcd": {
        "no": os.path.join(ABCD_DATA_DIR, "abcd_original"),
        "dpsgd": os.path.join(ABCD_DATA_DIR, "abcd_original"),
        "entity_only_person": os.path.join(ABCD_DATA_DIR, "abcd_my_delex-entity_only_low_1.47119"),
        "entity_only_medium": os.path.join(ABCD_DATA_DIR, "abcd_my_delex-entity_only_medium_2.71568"),
        "entity_only_high": os.path.join(ABCD_DATA_DIR, "abcd_my_delex-entity_only_high_3.13828"),
        "no_pronoun": os.path.join(ABCD_DATA_DIR, "abcd_my_delex-no_pronoun_19.22277"),
        "default": os.path.join(ABCD_DATA_DIR, "abcd_my_delex-default_22.26044"),
        "root": os.path.join(ABCD_DATA_DIR, "abcd_my_delex-root_25.20326"),
        "SRL": os.path.join(ABCD_DATA_DIR, "abcd_my_delex-SRL_28.6405"),
    },
    "wikitext2": {
        "no": os.path.join(DATA_DIR, "wikitext-2-raw"),
        "dpsgd": os.path.join(DATA_DIR, "wikitext-2-raw"),
        "entity_only_person": os.path.join(DATA_DIR, "wiki_entity_person_mask_consec-3.3"),
        "entity_only_medium": os.path.join(DATA_DIR, "wiki_entity_person_org_date_gpe_mask_consec-11.3"),
        "entity_only_high": os.path.join(DATA_DIR, "wiki_entity_all_mask_consec-16.4"),
        "no_pronoun": os.path.join(DATA_DIR, "wiki_contextual_no_pronoun-33.7"),
        "default": os.path.join(DATA_DIR, "wiki_contextual_default_mask_consec-34.8"),
        "root": os.path.join(DATA_DIR, "wiki_contextual_root_mask_consec-39.1"),
        "SRL": os.path.join(DATA_DIR, "wiki_contextual_SRL_mask_consec-45.0"),
    },
    ################ wiki ##################
    # raw data
    # "wikitext2": os.path.join(DATA_DIR, "wikitext-2-raw"),
    # # entity only
    # "wikitext2-delex-person": os.path.join(DATA_DIR, "wiki_entity_person_mask_consec-3.3"),
    # "wikitext2-delex-medium": os.path.join(DATA_DIR, "wiki_entity_person_org_date_gpe_mask_consec-11.3"),
    # "wikitext2-delex-high": os.path.join(DATA_DIR, "wiki_entity_all_mask_consec-16.4"),
    # # contextual
    # # "wikitext2-delex-no_pronoun": os.path.join(DATA_DIR, "wiki_contextual_no_pronoun-33.7"),
    # "wikitext2-delex-default": os.path.join(DATA_DIR, "wiki_contextual_default_mask_consec-34.8"),
    # "wikitext2-delex-root": os.path.join(DATA_DIR, "wiki_contextual_root_mask_consec-39.1"),
    # "wikitext2-delex-SRL": os.path.join(DATA_DIR, "wiki_contextual_SRL_mask_consec-45.0"),
    ################ abcd ##################
    # # public
    # "wikitext2-abcd": os.path.join(DATA_DIR, "abcd/abcd_original"),
    # # delex
    # "wikitext2-abcd-delex": os.path.join(DATA_DIR, "abcd/abcd_delex"),
}

# DEVICE = 1
EPSILON = 3
NON_PRIVATES = ["yes", "no"]


DELEX_LEVELS = [
    "no",
    "dpsgd",
    "entity_only_person",
    "entity_only_medium",
    "entity_only_high",
    "no_pronoun",
    "default",
    # "root",
    "SRL",
]


def parse_args():
    parser = argparse.ArgumentParser(description="delex a file")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        choices=["wikitext2-abcd", "wikitext2"],
        help="tasks",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        choices=list(range(8)),
        default=0,
        help="device",
    )
    parser.add_argument(
        "--contextual_level",
        "-cl",
        type=str,
        choices=DELEX_LEVELS,
        default=None,
        help="contextual level",
    )
    parser.add_argument("--canary_number", "-n", default=10, help="# of canaries")
    args = parser.parse_args()

    return args


def print_cmd(
    delex_level,
    device,
    task,
    non_private,
    data_dir,
    output_dir,
    miss,
    is_sdp_finetune,
    model_path,
    canary_number,
):
    if task == "wikitext2":
        if non_private == "yes":
            lr = 5e-5
            gradient_accumulation_steps = 1
            num_train_epochs = 3
        else:
            lr = 1e-4
            gradient_accumulation_steps = 256
            num_train_epochs = 200
    elif task == "wikitext2-abcd":
        if non_private == "yes":
            lr = 5e-5
            gradient_accumulation_steps = 1
            num_train_epochs = 3
        else:
            lr = 1e-4
            gradient_accumulation_steps = 128
            num_train_epochs = 200
    else:
        raise ValueError

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

    cmd = f"""
CUDA_VISIBLE_DEVICES={device} bash table2text/run.sh \\
{output_dir} `#output_dir` \\
{data_dir} `#data_dir` \\
{task} `#task_mode` \\
{model_path} `#model_name_or_path` \\
{EPSILON} `#target_epsilon` \\
yes `#ghost_clipping` \\
{non_private} `#non_private` \\
{is_sdp_finetune} `#is_sdp_finetune` \\
{num_train_epochs} `#num_train_epochs` \\
no `#add_canary` \\
{miss} `#miss_canary` \\
{canary_number} `#canary_times` \\
{lr} `#learning_rate` \\
{gradient_accumulation_steps} `#gradient_accumulation_steps`"""

    if delex_level in ["no", "dpsgd"]:
        cmd = f"""{cmd} \\
no `#add_mask`
"""
    else:
        cmd = f"""{cmd} \\
yes `#add_mask`
"""
    print(cmd)
    print()


def get_output_dir(task, delex_level, miss, non_private):
    return os.path.join(
        OUTPUT_DIR,
        "wiki" if task == "wikitext2" else "abcd",
        delex_level,
        "missed" if miss == "yes" else "not_missed",
        "public" if non_private == "yes" else "SDP",
    )


def main(args):
    if args.contextual_level == "default":
        MISSES = [
            "no",
            "yes",
        ]
    else:
        MISSES = [
            "no",
            # "yes",
        ]
    total = 0
    task_names = args.task.split(",") if args.task else TASK_TO_DATA_MAP.keys()
    delex_levels = args.contextual_level.split(",") if args.contextual_level else DELEX_LEVELS
    for task in task_names:
        for delex_level in delex_levels:
            data_dir = TASK_TO_DATA_MAP[task][delex_level]
            for non_private in NON_PRIVATES:
                if delex_level == "no" and non_private == "no":
                    continue
                elif delex_level == "dpsgd" and non_private == "yes":
                    continue

                for miss in MISSES:
                    output_dir = get_output_dir(task, delex_level, miss, non_private)
                    if non_private == "yes":
                        is_sdp_finetune = "no"
                    else:
                        if delex_level == "dpsgd":
                            is_sdp_finetune = "no"
                        else:
                            is_sdp_finetune = "yes"

                    if is_sdp_finetune == "yes":
                        model_path = os.path.join(
                            output_dir.replace("SDP", "public"),
                            "best",
                        )
                        # as long as it's private training, we always use the original data
                        data_dir = TASK_TO_DATA_MAP[task]["no"]
                    else:
                        model_path = "gpt2"

                    if non_private == "no":
                        # as long as it's private training, we always use the original canary
                        miss_override = "yes"
                    else:
                        if delex_level == "no":
                            miss_override = "yes"
                        else:
                            miss_override = miss

                    if delex_level in ["no", "dpsgd"]:
                        # public baseline and dpsgd baseline
                        output_dir = get_output_dir(task, delex_level, miss_override, non_private)

                    print_cmd(
                        delex_level=delex_level,
                        device=args.device,
                        task=task,
                        non_private=non_private,
                        data_dir=data_dir,  # TASK_TO_DATA_MAP["wikitext2"] if is_sdp_finetune == "yes" else data_dir,
                        output_dir=output_dir,
                        miss=miss_override,
                        is_sdp_finetune=is_sdp_finetune,
                        model_path=model_path,
                        canary_number=args.canary_number,
                    )
                    total += 1

    print(f"total: {total}*60G = {total*60}G, {total}*60min = {total}hour")


if __name__ == "__main__":
    args = parse_args()
    main(args)
