"""
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t qnli -cl default -d 6
"""
import os, sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import NORMALIZE_MAP


def parse_args():
    parser = argparse.ArgumentParser(description="delex a file")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        choices=["mnli", "sst-2", "qqp", "qnli"],
        help="tasks",
    )
    parser.add_argument(
        "--contextual_level",
        "-cl",
        type=str,
        choices=NORMALIZE_MAP.keys(),
        default=None,
        help="contextual level",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        choices=list(range(8)),
        default=0,
        help="device",
    )

    args = parser.parse_args()

    return args


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
    device=0,
):
    if non_private == "yes":
        lr = 1e-5
    else:
        lr = 5e-4

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
CUDA_VISIBLE_DEVICES={device} python -m classification.run_wrapper \\
--output_dir {output_dir} \\
--delex_level {delex_level} \\
--task_name {task} \\
--max_seq_len 256 \\
--non_private {non_private} \\
--model_name_or_path roberta-base \\
--target_epsilon {EPSILON} \\
--is_sdp_finetune {is_sdp_finetune} \\
--model_name_or_path {model_path} \\
--learning_rate {lr}
"""
    )
    print()


def main(args):
    total = 0
    task_names = args.task.split(",") if args.task else TASK_NAMES
    delex_levels = args.contextual_level.split(",") if args.contextual_level else DELEX_LEVELS
    for task in task_names:
        for delex_level in delex_levels:
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
                        device=args.device,
                    )
                    total += 1

    print(f"total: {total}*60G = {total*60}G, {total}*60min = {total}hour")


if __name__ == "__main__":
    args = parse_args()
    main(args)
