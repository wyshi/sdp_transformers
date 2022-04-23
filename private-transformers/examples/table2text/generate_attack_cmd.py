import os
import json
import pandas as pd
import argparse

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TASKS = ["wiki", "abcd"]

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))
FILE_NAME = os.path.join(ROOT_DIR, "attacks/canary_insertion/canary_insertion.py")


def return_cmd(task, context_level, miss_or_not, public_or_not, abs_model_path):
    csv_path = os.path.join(
        FILE_DIR, "attack", "canary_insertion", task, context_level, miss_or_not, public_or_not, "exposure.csv"
    )
    FILE_NAME
    cmd = f"""
python {FILE_NAME} \\
-bs 128 \\
--checkpoint {abs_model_path} \\
--outputf {csv_path}    """

    return cmd


def round_float(num):
    return round(num, 5)


def pick_best_perf_log(log_history):
    best_ppl = 1_000_000
    for log in log_history:
        val_ppl = round_float(log["val"]["model"]["ppl"])
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_log = log

    return best_log


def get_model_metrics(model_path, task, context_level, miss_or_not=None, public_or_not=None):
    with open(os.path.join(model_path, "log_history.json")) as fh:
        log_history = json.load(fh)

    if len(log_history) == 1:
        ckpts = [
            int(ckpt.split("-")[1]) for ckpt in os.listdir(model_path.replace("best", "")) if "checkpoint" in ckpt
        ]
        if len(ckpts) != 0:
            # there are /checkpoint-
            last_ckpt = sorted(ckpts)[-1]
            ckpt_path = model_path.replace("best", f"checkpoint-{last_ckpt}")
            with open(os.path.join(ckpt_path, "log_history.json")) as fh:
                log_history = json.load(fh)
            # pick the best one, instead of the last one
            result = pick_best_perf_log(log_history[1:])
        else:
            # there are NO /checkpoint-, only /best
            result = log_history[-1]
    else:
        result = log_history[-1]
    metrics = {
        "task": task,
        "context_level": context_level,
        "miss_or_not": miss_or_not,
        "public_or_not": public_or_not,
    }
    metrics.update(
        {
            "valid_ppl": round_float(result["val"]["model"]["ppl"]),
            "test_ppl": round_float(result["eval"]["model"]["ppl"]),
        }
    )
    metrics.update(
        {k: round_float(v) if type(v) is float else v for k, v in result.items() if k not in ["train", "eval", "val"]}
    )

    return metrics


def main(args):
    total = 0
    cmds = []
    for task in TASKS:
        for context_level in os.listdir(os.path.join(OUTPUT_DIR, task)):
            if context_level in [
                "param_tune",
                "pre_mask_model",
            ]:
                continue
            potential_ckpts = os.listdir(os.path.join(OUTPUT_DIR, task, context_level))
            is_dir_cnt = len(
                [_c for _c in potential_ckpts if os.path.isdir(os.path.join(OUTPUT_DIR, task, context_level, _c))]
            )
            if is_dir_cnt == 0:
                continue
            elif "best" in potential_ckpts or len([_c for _c in potential_ckpts if _c.startswith("checkpoint")]):
                abs_model_path = os.path.join(
                    OUTPUT_DIR,
                    task,
                    context_level,
                )

                cmd = return_cmd(
                    task,
                    context_level,
                    miss_or_not,
                    public_or_not,
                    abs_model_path,
                )
                cmds.append(cmd)
                total += 1
            else:
                for miss_or_not in os.listdir(os.path.join(OUTPUT_DIR, task, context_level)):
                    for public_or_not in os.listdir(os.path.join(OUTPUT_DIR, task, context_level, miss_or_not)):
                        abs_model_path = os.path.join(
                            OUTPUT_DIR,
                            task,
                            context_level,
                            miss_or_not,
                            public_or_not,
                        )
                        cmd = return_cmd(
                            task,
                            context_level,
                            miss_or_not,
                            public_or_not,
                            abs_model_path,
                        )
                        cmds.append(cmd)
                        total += 1
    print(f"total {total} jobs")
    if args.device is None:
        devices = [i % 8 for i in range(total)]
        devices = sorted(devices)
    else:
        devices = [args.device] * total
    cmds = [cmd + f"--cuda cuda:{device}" for cmd, device in zip(cmds, devices)]

    for cmd in cmds:
        print(cmd)
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="delex a file")
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
