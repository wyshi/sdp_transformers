import torch
import numpy as np
from tqdm import tqdm

F_public = torch.load("found_transition/SimpleSampleConvNet_1.0_1.0_1.0_0.1_200.pt")["fc1.weight"].cpu().numpy()

PUBLIC_GRADS = torch.load("grads_normalize=True_dp=False_sample-rate=0.001_epoch=5.pt")
TRUE_GRADS = torch.load("grads_normalize=False_dp=False_sample-rate=0.001_epoch=5.pt")
NOISE_GRADS = torch.load("grads_normalize=False_dp=True_sample-rate=0.001_epoch=5.pt")


residual_public_norms = []
for i in tqdm(range(len(TRUE_GRADS) - 1)):
    residual_public_norms.append(
        np.linalg.norm(TRUE_GRADS[i + 1] - np.matmul(F_public, TRUE_GRADS[i].reshape(-1)).reshape(TRUE_GRADS[0].shape))
    )
np.mean(residual_public_norms)

residual_public = []
for i in tqdm(range(len(TRUE_GRADS) - 1)):
    residual_public.append(PUBLIC_GRADS[i + 1].reshape(-1) - np.matmul(F_public, PUBLIC_GRADS[i].reshape(-1)))

estimated_transition_covariance = np.cov(np.stack(residual_public).T)


cov_residual_public = [np.cov(res) for res in tqdm(residual_public)]
for i in tqdm(range(len(TRUE_GRADS) - 1)):
    cov_residual_public.append(TRUE_GRADS[i + 1].reshape(-1) - np.matmul(F_public, TRUE_GRADS[i].reshape(-1)))


residual_noise = []
for i in tqdm(range(len(TRUE_GRADS) - 1)):
    residual_noise.append(np.linalg.norm(TRUE_GRADS[i + 1] - NOISE_GRADS[i + 1]))
np.mean(residual_noise)
