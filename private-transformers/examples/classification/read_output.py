import os
import json
from pyexpat import model
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TASKS = ["mnli", "qnli", "qqp", "sst-2"]


def round_float(num):
    return round(num, 5)


def pick_best_perf_log(log_history):
    best_acc = -100_000
    for log in log_history:
        eval_acc = log["dev"]["eval_acc"] if "eval_acc" in log["dev"] else log["dev"]["eval_mnli/acc"]
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_log = log

    return best_log


def get_model_metrics(model_path, task, context_level, miss_or_not, public_or_not):
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
            "eval_loss": round_float(result["dev"]["eval_loss"]),
            "eval_acc": round_float(result["dev"]["eval_acc"])
            if "eval_acc" in result["dev"]
            else round_float(result["dev"]["eval_mnli/acc"]),
        }
    )
    if "privacy_spent" in result:
        privacy_metrics = {k: round_float(v) if type(v) is float else v for k, v in result["privacy_spent"].items()}

        metrics.update(privacy_metrics)

    return metrics


records = []
for task in TASKS:
    for context_level in os.listdir(os.path.join(OUTPUT_DIR, task)):
        for miss_or_not in os.listdir(os.path.join(OUTPUT_DIR, task, context_level)):
            for public_or_not in os.listdir(os.path.join(OUTPUT_DIR, task, context_level, miss_or_not)):
                best_model_path = os.path.join(
                    OUTPUT_DIR,
                    task,
                    context_level,
                    miss_or_not,
                    public_or_not,
                    "best",
                )
                try:
                    metrics = get_model_metrics(best_model_path, task, context_level, miss_or_not, public_or_not)
                    records.append(metrics)
                except:
                    pass
records = pd.DataFrame(
    records,
)


save_to_path = os.path.join(OUTPUT_DIR, "summarized_perf.csv")
print(f"save to {save_to_path}")
records.to_csv(save_to_path, index=None)


def round_num(num, multiply_by_100):
    if multiply_by_100:
        return format(round(num * 100, 2), ".2f")
    else:
        return format(round(num, 2), ".2f")


CL_MAP = {
    "entity_only_medium": "low entity",
    "entity_only_high": "high entity",
    "default": "low contextual",
    "SRL": "high contextual",
}

# generate latex
lines = []
for public_or_not in ["public", "SDP", "CRT"]:
    for context_level in ["entity_only_medium", "entity_only_high", "default", "SRL"]:
        items = [public_or_not, CL_MAP[context_level]]
        for task in ["mnli", "qqp", "qnli", "sst-2"]:
            items.append("\%")
            selected_record = records[
                (records["task"] == task)
                & (records["context_level"] == context_level)
                & (records["public_or_not"] == public_or_not)
            ]
            if selected_record.shape[0] == 0:
                continue
            items.append(round_num(selected_record["eval_acc"].values.item(), True))
            if not np.isnan(selected_record["eps_estimate"].values.item()):
                items.append(round_num(selected_record["eps_estimate"].values.item(), False))
            else:
                items.append("-")
        try:
            lines.append(" & ".join([str(itm) for itm in items]))
        except:
            import pdb

            pdb.set_trace()

for line in lines:
    print(line + "\\\\")
