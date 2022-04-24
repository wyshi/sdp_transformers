"""
# sst-2 and mnli
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t sst-2 -cl SRL -d 1
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t mnli -cl SRL -d 1

python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t sst-2 -cl entity_only_high -d 3
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t mnli -cl entity_only_high -d 3


python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t sst-2 -cl entity_only_medium -d 4
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t mnli -cl entity_only_medium -d 4

# qqp and qnli
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t qqp -cl SRL -d 5
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t qnli -cl SRL -d 5

python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t qqp -cl entity_only_high -d 6
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t qnli -cl entity_only_high -d 6


python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t qqp -cl entity_only_medium -d 7
python /local/data/wyshi/sdp_transformers/private-transformers/examples/classification/generate_cmd.py -t qnli -cl entity_only_medium -d 7


"""
import os
import argparse
from glob import glob
import pdb

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/original")


def get_data_dir(input_dir, contextual_level):
    if contextual_level != "no":
        prefix = input_dir.replace("original", "normalized_mask")
        _data_dir = os.path.join(prefix, input_dir.split("/")[-1] + f"-{contextual_level}")

        # pdb.set_trace()
        data_dir = [_dir for _dir in glob(os.path.join(prefix, "*")) if _dir.startswith(_data_dir)][0]
    else:
        data_dir = input_dir
    print(data_dir)
    return data_dir


def get_data_from_task(task_name, is_sdp_finetune, delex_level, crt_baseline=False):
    data_dir_suffix = {
        "sst-2": "GLUE-SST-2",
        "mnli": "MNLI",
        "qqp": "QQP",
        "qnli": "QNLI",
    }[task_name]
    data_dir = f"{DATA_DIR}/{data_dir_suffix}"
    if is_sdp_finetune == "yes":
        # we should use origianl dataset
        pass
    else:
        if delex_level == "no":
            pass
        else:
            data_dir = get_data_dir(data_dir, delex_level)

    # if crt_baseline:
    #     data_dir = get_data_dir(data_dir, delex_level)

    return data_dir


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
        choices=[
            "no",
            "entity_only_low",
            "entity_only_medium",
            "entity_only_high",
            "no_pronoun",
            "default",
            "root",
            "SRL",
        ],
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
    parser.add_argument("--crt_baseline", "-crt", action="store_true", default=False)

    args = parser.parse_args()

    return args


OUTPUT_DIR = "classification/output"

TASK_NAMES = [
    "sst-2",
    "qnli",
    "qqp",
    "mnli",
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
    crt_baseline=False,
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

    data_dir = get_data_from_task(
        task_name=task, is_sdp_finetune=is_sdp_finetune, delex_level=delex_level, crt_baseline=crt_baseline
    )
    print(
        f"""
CUDA_VISIBLE_DEVICES={device} python -m classification.run_wrapper \\
--output_dir {output_dir} \\
--data_dir {data_dir} \\
--delex_level {delex_level} \\
--task_name {task} \\
--max_seq_len 256 \\
--non_private {non_private} \\
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
                    if args.contextual_level == "no" and non_private == "no":
                        is_sdp_finetune = "no"
                    if is_sdp_finetune == "yes":
                        model_path = os.path.join(
                            output_dir.replace("SDP", "public"),
                            "best",
                        )
                    else:
                        model_path = "roberta-base"
                    if args.crt_baseline:
                        if non_private == "yes":
                            continue
                        output_dir = output_dir.replace("SDP", "CRT")
                        is_sdp_finetune = "no"
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
                        crt_baseline=args.crt_baseline,
                    )
                    total += 1

    print(f"total: {total}*60G = {total*60}G, {total}*60min = {total}hour")


if __name__ == "__main__":
    args = parse_args()
    main(args)
