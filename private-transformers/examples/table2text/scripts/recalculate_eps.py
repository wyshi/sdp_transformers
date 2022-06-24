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
DELTA = 1e-6

TASK_CONFIG = {
    "wiki": {
        "sample_rate": 512 / 2361,
        "low_miss_rate": 0.00356,
        "high_miss_rate": 0.0072,
        "delta": DELTA,
        "accurate": {"step": 1000, "sigma": 0.9445556640625},
        "conservative": {"step": 980, "sigma": 8.345989990234376, "miss_rate": 0.15},
    },
    "abcd": {
        # /local/data/wyshi/sdp_transformers/private-transformers/examples/table2text/output/abcd/entity_only_high/test_glw_amplification_1%_eps0.5_200epoches_grad_acc=128/public/argparse.json
        "sample_rate": 256 / 2547,
        "low_miss_rate": 0.0008067672103024004,
        "high_miss_rate": 0.0116932327896976,
        "delta": DELTA,
        "accurate": {"step": 1960, "sigma": 0.8302978515625},
        "conservative": {"step": 1840, "sigma": 2.09951171875, "miss_rate": 0.05},
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

# print("wiki")
# print("accurate")
# ACCURATE_SIGMA = 0.9445556640625
# ACCURATE_STEP = 1000
# CONSERVATIVE_SIGMA = 8.345989990234376
# CONSERVATIVE_STEP = 980
# print("low")
# print(
#     _eps_from_glw(
#         sample_rate=512 / 2361 * 0.00356,
#         sigma=ACCURATE_SIGMA,
#         steps=ACCURATE_STEP,
#         alphas=kwargs_for_get_sigma["alphas"],
#         delta=1e-6,
#     )
# )
# print("high")
# print(
#     _eps_from_glw(
#         sample_rate=512 / 2361 * 0.0072,
#         sigma=ACCURATE_SIGMA,
#         steps=ACCURATE_STEP,
#         alphas=kwargs_for_get_sigma["alphas"],
#         delta=1e-6,
#     )
# )

# print("low")
# print(
#     _eps_from_glw(
#         sample_rate=512 / 2361 * 0.00356,
#         sigma=CONSERVATIVE_SIGMA,
#         steps=940,
#         alphas=kwargs_for_get_sigma["alphas"],
#         delta=1e-6,
#     )
# )
# # (0.14027361307764652, 63.0)
# print("high")
# {"eps_low": -0.04496990818914154, "eps_estimate": 0.005030292292405734, "eps_upper": 0.05503049277391299}
# print(
#     _eps_from_glw(
#         sample_rate=512 / 2361 * 0.0072,
#         sigma=CONSERVATIVE_SIGMA,
#         steps=940,
#         alphas=kwargs_for_get_sigma["alphas"],
#         delta=1e-6,
#     )
# )
# # (0.1411023107261905, 63.0)
# {"eps_low": -0.028094594643937647, "eps_estimate": 0.021906759439840967, "eps_upper": 0.07190811352178604}
# 0.17


# print("abcd")
# # abcd
# # Training set size: 2547,
# _eps_from_glw(
#     sample_rate=256 / 2547 * 0.0116932327896976,
#     sigma=1.2654296875,
#     steps=1840,
#     alphas=kwargs_for_get_sigma["alphas"],
#     delta=1e-6,
# )
# # (0.5161440565400476, 21.0)
# {"eps_low": 0.13834318988952682, "eps_estimate": 0.1883555757666907, "eps_upper": 0.2383679614904465}
# _eps_from_glw(
#     sample_rate=256 / 2547 * 0.0008067672103024004,
#     sigma=1.2654296875,
#     steps=1840,
#     alphas=kwargs_for_get_sigma["alphas"],
#     delta=1e-6,
# )
# # (0.32649364341339515, 30.0)
# {"eps_low": -0.04045315174609455, "eps_estimate": 0.009547613661083016, "eps_upper": 0.059548379067674566}
# _eps_from_glw(
#     sample_rate=256 / 2547 * 0.0008067672103024004,
#     sigma=4.7472290039062495,
#     steps=1400,
#     alphas=kwargs_for_get_sigma["alphas"],
#     delta=1e-6,
# )
# # (0.14001888017821548, 63.0)
# {"eps_low": -0.050000792164511275, "eps_estimate": -7.911645102979916e-07, "eps_upper": 0.04999920983549079}
# _eps_from_glw(
#     sample_rate=256 / 2547 * 0.0116932327896976,
#     sigma=4.7472290039062495,
#     steps=1400,
#     alphas=kwargs_for_get_sigma["alphas"],
#     delta=1e-6,
# )
# # (0.3441841386090626, 55.0)
# {"eps_low": -0.014742304105024657, "eps_estimate": 0.03526008365115967, "eps_upper": 0.08526247140164277}
