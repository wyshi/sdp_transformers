import os

DATA_DIR = "../../data"
OUTPUT_DIR = "table2text/output"

TASK_TO_DATA_MAP = {
    ################ wiki ##################
    # raw data
    "wikitext2": os.path.join(DATA_DIR, "wikitext-2-raw"),
    # entity only
    "wikitext2-delex-person": os.path.join(DATA_DIR, "wiki_entity_person-3.3"),
    "wikitext2-delex-medium": os.path.join(DATA_DIR, "wiki_entity_person_org_date_gpe-11.3"),
    "wikitext2-delex-high": os.path.join(DATA_DIR, "wiki_entity_all-16.4"),
    # contextual
    # "wikitext2-delex-no_pronoun": os.path.join(DATA_DIR, "wiki_contextual_no_pronoun-33.7"),
    "wikitext2-delex-default": os.path.join(DATA_DIR, "wiki_contextual_default-34.8"),
    "wikitext2-delex-root": os.path.join(DATA_DIR, "wiki_contextual_root-39.1"),
    "wikitext2-delex-SRL": os.path.join(DATA_DIR, "wiki_contextual_SRL-45.0"),
    ################ abcd ##################
    # # public
    # "wikitext2-abcd": os.path.join(DATA_DIR, "abcd/abcd_original"),
    # # delex
    # "wikitext2-abcd-delex": os.path.join(DATA_DIR, "abcd/abcd_delex"),
}

DEVICE = 1
EPSILON = 3
non_privates = [
    "yes",
    # "no"
]

misses = [
    "no",
    "yes",
]


def print_cmd(
    task,
    non_private,
    data_dir,
    output_dir,
    miss,
    is_sdp_finetune,
    model_path,
):
    if non_private == "yes":
        lr = 5e-5
        gradient_accumulation_steps = 1
        num_train_epochs = 3
    else:
        lr = 5e-4
        gradient_accumulation_steps = 512
        num_train_epochs = 200

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
CUDA_VISIBLE_DEVICES={DEVICE} bash table2text/run.sh \\
{output_dir} `#output_dir` \\
{data_dir} `#data_dir` \\
{task} `#task_mode` \\
{model_path} `#model_name_or_path` \\
{EPSILON} `#target_epsilon` \\
yes `#ghost_clipping` \\
{non_private} `#non_private` \\
{is_sdp_finetune} `#is_sdp_finetune` \\
{num_train_epochs} `#num_train_epochs` \\
yes `#add_canary` \\
{miss} `#miss_canary` \\
10 `#canary_times` \\
{lr} `#learning_rate` \\
{gradient_accumulation_steps} `#gradient_accumulation_steps` 
"""
    )
    print()


total = 0
for task in TASK_TO_DATA_MAP:
    for non_private in non_privates:
        data_dir = TASK_TO_DATA_MAP[task]
        last_part_data_dir = data_dir.split("/")[-1]
        output_dir = os.path.join(
            OUTPUT_DIR,
            "abcd" if "abcd" in task else "wiki",
        )
        if task in ["wikitext2", "wikitext2-abcd"]:
            # baselines
            final_output_dir = os.path.join(
                output_dir,
                f"{last_part_data_dir}_PUBLIC" if non_private == "yes" else f"{last_part_data_dir}_DPSGD",
            )
            print_cmd(
                task=task,
                non_private=non_private,
                data_dir=data_dir,
                output_dir=final_output_dir,
                miss="yes",
                is_sdp_finetune="no",
                model_path="gpt2",
            )
            total += 1
        else:
            for miss in misses:
                if miss == "yes" and task not in ["wikitext2-delex-high", "wikitext2-delex-no_pronoun"]:
                    continue
                public = "yes" in non_private
                miss_binary = "yes" in miss
                is_sdp_finetune = "no" if non_private == "yes" else "yes"
                if is_sdp_finetune == "yes":
                    model_path = os.path.join(
                        output_dir,
                        last_part_data_dir,
                        f"{'missed' if miss_binary else 'not_missed'}",
                        f"public",
                        "best",
                    )
                else:
                    model_path = "gpt2"
                final_output_dir = os.path.join(
                    output_dir,
                    last_part_data_dir,
                    f"{'missed' if miss_binary else 'not_missed'}",
                    f"{'public' if public else 'SDP'}",
                )
                print_cmd(
                    task=task,
                    non_private=non_private,
                    data_dir=data_dir,
                    output_dir=final_output_dir,
                    miss=miss,
                    is_sdp_finetune=is_sdp_finetune,
                    model_path=model_path,
                )
                total += 1

print(f"total: {total}*60G = {total*60}G, {total}*60min = {total}hour")
