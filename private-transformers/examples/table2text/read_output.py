import os
import json
from pyexpat import model
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TASKS = ["wiki", "abcd"]


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
    with open(os.path.join(model_path, "../argparse.json")) as fh:
        argparse_record = json.load(fh)

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
        "lr": argparse_record["learning_rate"],
        "add_mask": argparse_record["add_mask"] if "add_mask" in argparse_record else None,
        "gradient_accumulation_steps": argparse_record["gradient_accumulation_steps"],
        "per_device_train_batch_size": argparse_record["per_device_train_batch_size"],
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


records = []
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
            best_model_path = os.path.join(
                OUTPUT_DIR,
                task,
                context_level,
                "best",
            )
            if not os.path.exists(best_model_path):
                continue
            metrics = get_model_metrics(
                best_model_path,
                task,
                context_level,
            )
            records.append(metrics)
        else:
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
                    if not os.path.exists(best_model_path):
                        continue
                    metrics = get_model_metrics(best_model_path, task, context_level, miss_or_not, public_or_not)
                    records.append(metrics)

records = pd.DataFrame(
    records,
)
records = records.sort_values(by=["task", "context_level", "miss_or_not", "public_or_not", "valid_ppl"])
save_to_path = os.path.join(OUTPUT_DIR, "summarized_perf.csv")
print(f"save to {save_to_path}")
records.to_csv(save_to_path, index=None)


# generate latex
def round_num(num, multiply_by_100):
    if multiply_by_100:
        return format(round(num, 2), ".2f")
    else:
        return format(round(num, 2), ".2f")


import numpy as np

PCT_MAP = {
    "wiki": {"CRT": "", "entity_only_medium": 11.3, "entity_only_high": 16.4, "default": 34.8, "SRL": 45.0},
    "abcd": {"CRT": "", "entity_only_medium": 2.7, "entity_only_high": 3.1, "default": 22.3, "SRL": 28.6},
}

CL_MAP = {
    "CRT": "CRT",
    "entity_only_medium": "low entity",
    "entity_only_high": "high entity",
    "default": "low contextual",
    "SRL": "high contextual",
}

PUBLIC_MAP = {"public": "redacted", "SDP": "SDP", "DP": "DPSGD"}
lines = []
for context_level in ["entity_only_medium", "entity_only_high", "default", "SRL"]:
    for public_or_not in [
        "public",
        "SDP",
    ]:

        items = [PUBLIC_MAP[public_or_not], CL_MAP[context_level]]
        for task in [
            "wiki",
            "abcd",
        ]:

            items.append(f"{PCT_MAP[task][context_level]}\%")
            selected_record = records[
                (records["task"] == task)
                & (records["context_level"] == context_level)
                & (records["public_or_not"] == public_or_not)
                & (records["miss_or_not"] == "not_missed")
            ]
            if selected_record.shape[0] == 0:
                continue
            try:
                items.append(round_num(selected_record["valid_ppl"].values.item(), True))
            except:
                import pdb

                pdb.set_trace()
            if not np.isnan(selected_record["eps_estimate"].values.item()):
                items.append(round_num(selected_record["eps_estimate"].values.item(), False))
            else:
                items.append("-")
        try:
            lines.append(" & ".join([str(itm) for itm in items]))
        except:
            import pdb

            pdb.set_trace()


for context_level in [
    "CRT",
]:
    for public_or_not in [
        "DP",
        "public",
        "SDP",
    ]:
        for miss_or_not in ["entity_only_medium", "entity_only_high", "default", "SRL"]:
            items = [context_level, CL_MAP[miss_or_not]]
            for task in [
                "wiki",
                "abcd",
            ]:

                items.append(f"{PCT_MAP[task][context_level]}\%")
                selected_record = records[
                    (records["task"] == task)
                    & (records["context_level"] == context_level)
                    & (records["public_or_not"] == public_or_not)
                    & (records["miss_or_not"] == miss_or_not)
                ]
                if selected_record.shape[0] == 0:
                    continue
                try:
                    items.append(round_num(selected_record["valid_ppl"].values.item(), True))
                except:
                    import pdb

                    pdb.set_trace()
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
