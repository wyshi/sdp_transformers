from private_transformers import PrivacyEngine
import private_transformers
private_transformers
<module 'private_transformers' from '/home/wyshi/anaconda3/envs/lm_privacy/lib/python3.9/site-packages/private_transformers/__init__.py'>
kwargs_for_get_sigma={'target_epsilon': 0.5, 'target_delta': 1e-06, 'sample_rate': 512/2361*0.01, 'epochs': 5.0, 'alphas': (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63), 'eps_error': 0.05, 'steps': 24}
from private_transformers.privacy_utils.privacy_engine import _eps_from_rdp_cks, _eps_from_glw, get_sigma_from_glw
_eps_from_glw(sample_rate=512/2361*0.01, sigma=1.3435058593750002, steps=980, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.5272485426905034, 21.0)
{'eps_low': 0.19060990904896216, 'eps_estimate': 0.24062540784627387, 'eps_upper': 0.2906409064033766}
_eps_from_glw(sample_rate=512/2361*0.0072, sigma=1.3435058593750002, steps=980, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.4721194922865186, 23.0)
{'eps_low': 0.11956280092353658, 'eps_estimate': 0.1695738024233137, 'eps_upper': 0.2195848038020592}
_eps_from_glw(sample_rate=512/2361*0.00356, sigma=1.3435058593750002, steps=980, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.40630140896215927, 25.0)
{'eps_low': 0.030041957272570415, 'eps_estimate': 0.08004757581871409, 'eps_upper': 0.13005319433328974}
_eps_from_glw(sample_rate=512/2361*0.00356, sigma=8.149957275390628, steps=940, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.14027361307764652, 63.0)
{'eps_low': -0.04496990818914154, 'eps_estimate': 0.005030292292405734, 'eps_upper': 0.05503049277391299}
_eps_from_glw(sample_rate=512/2361*0.0072, sigma=8.149957275390628, steps=940, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.1411023107261905, 63.0)
{'eps_low': -0.028094594643937647, 'eps_estimate': 0.021906759439840967, 'eps_upper': 0.07190811352178604}
0.17
# abcd
# Training set size: 2547, 
_eps_from_glw(sample_rate=256/2547*0.0116932327896976, sigma=1.2654296875, steps=1840, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.5161440565400476, 21.0)
{'eps_low': 0.13834318988952682, 'eps_estimate': 0.1883555757666907, 'eps_upper': 0.2383679614904465}
_eps_from_glw(sample_rate=256/2547*0.0008067672103024004, sigma=1.2654296875, steps=1840, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.32649364341339515, 30.0)
{'eps_low': -0.04045315174609455, 'eps_estimate': 0.009547613661083016, 'eps_upper': 0.059548379067674566}
_eps_from_glw(sample_rate=256/2547*0.0008067672103024004, sigma=4.7472290039062495, steps=1400, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.14001888017821548, 63.0)
{'eps_low': -0.050000792164511275, 'eps_estimate': -7.911645102979916e-07, 'eps_upper': 0.04999920983549079}
_eps_from_glw(sample_rate=256/2547*0.0116932327896976, sigma=4.7472290039062495, steps=1400, alphas=kwargs_for_get_sigma['alphas'], delta=1e-6)
# (0.3441841386090626, 55.0)
{'eps_low': -0.014742304105024657, 'eps_estimate': 0.03526008365115967, 'eps_upper': 0.08526247140164277}