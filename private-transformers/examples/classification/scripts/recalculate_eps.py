# private_transformers
# <module 'private_transformers' from '/home/wyshi/anaconda3/envs/lm_privacy/lib/python3.9/site-packages/private_transformers/__init__.py'>
from private_transformers import PrivacyEngine
import private_transformers
from private_transformers.privacy_utils.privacy_engine import (
    _eps_from_rdp_cks,
    _eps_from_glw,
    get_sigma_from_glw,
    DEFAULT_ALPHAS,
)

kwargs_for_get_sigma = {
    "target_epsilon": 0.5,
    "target_delta": 1e-06,
    "sample_rate": 512 / 2361 * 0.01,
    "epochs": 5.0,
    "alphas": DEFAULT_ALPHAS,
    "eps_error": 0.05,
    "steps": 24,
}

TASK_CONFIG = {
    "mnli": {
        "sample_rate": 6000 / 392702,
        "low_miss_rate": 0.0028822402538490497,
        "high_miss_rate": 0.01076488972382562,
        "delta": 1.2732300828618139e-06,
        "accurate": {"step": 1081, "sigma": 0.6036865234375001},
        "conservative": {"step": 1170, "sigma": 0.8541015625000001, "miss_rate": 0.086},
    },
    "qqp": {
        "sample_rate": 6000 / 363846,
        "low_miss_rate": 0.003212509558241031,
        "high_miss_rate": 0.011670125552614308,
        "delta": 1.374207769221044e-06,
        "accurate": {"step": 777, "sigma": 0.61416015625},
        "conservative": {"step": 777, "sigma": 0.8560058593750002, "miss_rate": 0.083},
    },
    "qnli": {
        "sample_rate": 2000 / 104743,
        "low_miss_rate": 0.001244755937890738,
        "high_miss_rate": 0.006136074703850561,
        "delta": 4.773588688504244e-06,
        "accurate": {"step": 282, "sigma": 0.5103759765625},
        "conservative": {"step": 306, "sigma": 0.935986328125, "miss_rate": 0.172},
    },
    "sst-2": {
        "sample_rate": 1000 / 67349,
        "low_miss_rate": 1e-8,
        "high_miss_rate": 0.018063928918950645,
        "delta": 7.4240152043831385e-06,
        "accurate": {"step": 180, "sigma": 0.5541748046875001},
        "conservative": {"step": 176, "sigma": 0.600830078125, "miss_rate": 0.03},
    },
}


def print_high_low_bound(
    task, accurate, delta, sigma, step, sample_rate, low_miss_rate, high_miss_rate, actual_miss_rate=None
):
    if accurate == "accurate":
        print(f"{task}-{accurate}")
        print("low")
        results = _eps_from_glw(
            sample_rate=sample_rate * low_miss_rate,
            sigma=sigma,
            steps=step,
            alphas=kwargs_for_get_sigma["alphas"],
            delta=delta,
        )
        for k in results:
            results[k] = round(results[k], 3)
        print(results)
        print("high")
        results = _eps_from_glw(
            sample_rate=sample_rate * high_miss_rate,
            sigma=sigma,
            steps=step,
            alphas=kwargs_for_get_sigma["alphas"],
            delta=delta,
        )
        for k in results:
            results[k] = round(results[k], 3)
        print(results)
        print()

    else:
        print(f"{task}-{accurate}")
        print("actual miss rate")
        results = _eps_from_glw(
            sample_rate=sample_rate * actual_miss_rate,
            sigma=sigma,
            steps=step,
            alphas=kwargs_for_get_sigma["alphas"],
            delta=delta,
        )
        for k in results:
            results[k] = round(results[k], 3)
        print(results)
        print()


for task, values in TASK_CONFIG.items():
    for accurate in ["accurate", "conservative"]:
        print_high_low_bound(
            task=task,
            accurate=accurate,
            delta=values["delta"],
            sigma=values[accurate]["sigma"],
            step=values[accurate]["step"],
            sample_rate=values["sample_rate"],
            low_miss_rate=values["low_miss_rate"],
            high_miss_rate=values["high_miss_rate"],
            actual_miss_rate=values["conservative"]["miss_rate"],
        )
