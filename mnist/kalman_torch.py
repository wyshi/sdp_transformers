import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from KalmanFilterTorch import KalmanFilterTorch

# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
import numpy as np
import torch
from tqdm import tqdm

device = "cuda:0"

F_public = torch.load("found_transition/SimpleSampleConvNet_1.0_1.0_1.0_0.1_200.pt")["fc1.weight"].to(device)

# my_filter = KalmanFilter(dim_x=2, dim_z=1)
# my_filter.x = np.array([[2.0], [0.0]])  # initial state (location and velocity)

# my_filter.F = np.load("/local/data/wyshi/sdp_transformers/mnist/tm.npy")  # state transition matrix

# my_filter.H = np.array([[1.0, 0.0]])  # Measurement function
# my_filter.P *= 1000.0  # covariance matrix
# my_filter.R = np.identity(N) * SIGMA  # state uncertainty
# dt = 1
# my_filter.Q = Q_discrete_white_noise(2, dt, 0.1)  # process uncertainty


PUBLIC_GRADS = torch.stack(
    [torch.from_numpy(_grad) for _grad in torch.load("grads_normalize=True_dp=False_sample-rate=0.001_epoch=5.pt")]
).to(device)
TRUE_GRADS = torch.stack(
    [torch.from_numpy(_grad) for _grad in torch.load("grads_normalize=False_dp=False_sample-rate=0.001_epoch=5.pt")]
).to(device)
NOISE_GRADS = torch.stack(
    [torch.from_numpy(_grad) for _grad in torch.load("grads_normalize=False_dp=True_sample-rate=0.001_epoch=5.pt")]
).to(device)

m = 10
n = 28 * 28

N = m * n  # * 784
SIGMA = 1
TIMESTEP = 10

#################################################################
# estimate transition covariance
#################################################################
residual_public = []
for i in tqdm(range(len(PUBLIC_GRADS) - 1)):
    residual_public.append(PUBLIC_GRADS[i + 1].reshape(-1) - F_public @ PUBLIC_GRADS[i].reshape(-1))

estimated_transition_covariance = torch.cov(torch.stack(residual_public).T)

estimated_initial_state_covariance = torch.cov(torch.stack([grad.reshape(-1) for grad in PUBLIC_GRADS]).T)

# tm = np.load("/local/data/wyshi/sdp_transformers/mnist/tm.npy")
# tc = np.load("/local/data/wyshi/sdp_transformers/mnist/tc.npy")
kf = KalmanFilterTorch(
    n_dim_obs=N,
    # initial
    initial_state_mean=PUBLIC_GRADS[0].reshape(-1),
    initial_state_covariance=estimated_initial_state_covariance,
    # transition
    transition_matrices=F_public,
    transition_covariance=estimated_transition_covariance,  # we don't have to use this estimate, since kf will estimate for us
    transition_offsets=torch.zeros(N).to(device),
    # observation
    observation_matrices=torch.eye(N).to(device),
    observation_covariance=torch.eye(N).to(device) * SIGMA,
    observation_offsets=torch.zeros(N).to(device),
    device=device,
    # em_vars=[
    #     "transition_matrices",
    #     "transition_covariance",  # "initial_state_mean", "initial_state_covariance"
    # ],
)

np.random.seed(1)

# measurements = [np.random.randn(m, n).reshape(-1) for _ in range(TIMESTEP)]
# measurements = [noise_grads[i].reshape(-1) for i in range(TIMESTEP)]
# # kf.em(measurements, n_iter=2)
# p1, c1 = kf.filter_update([noise_grads[0].reshape(-1), noise_grads[1].reshape(-1).tolist()])

n_timesteps = 10
n_dim_state = N
filtered_state_means = []  # [kf.initial_state_mean]  # torch.zeros((n_timesteps, n_dim_state)).to(device)
filtered_state_covariances = []  # [kf.initial_state_covariance]
# torch.zeros((n_timesteps, n_dim_state, n_dim_state)).to(device)
# 6min/ 4 steps
for t in tqdm(range(n_timesteps - 1)):
    if t == 0:
        prev_state_means = kf.initial_state_mean
        prev_state_covariances = kf.initial_state_covariance
        # filtered_state_means[t] = kf.initial_state_mean
        # filtered_state_covariances[t] = kf.initial_state_covariance
    tmp_filtered_state_means, tmp_filtered_state_covariances = kf.filter_update(
        filtered_state_mean=prev_state_means,  # filtered_state_means[t],
        filtered_state_covariance=prev_state_covariances,  # filtered_state_covariances[t],
        observation=NOISE_GRADS[t].reshape(-1),
        # transition_offset=data.transition_offsets[t],
    )
    filtered_state_means.append(tmp_filtered_state_means)
    filtered_state_covariances.append(tmp_filtered_state_covariances)
    prev_state_means = tmp_filtered_state_means
    prev_state_covariances = tmp_filtered_state_covariances
import pdb

pdb.set_trace()

start = 0
residual_after_filter = [
    TRUE_GRADS[i] - filtered_state_means[i].reshape(TRUE_GRADS[i].shape)
    for i in range(start, len(filtered_state_means))
]
print([torch.linalg.norm(r) for r in residual_after_filter])
[2.5610855624632327, 2.9230360122852415, 2.989765793629204, 5.320349435377508]
print([torch.linalg.norm(r, ord=1) for r in residual_after_filter])
[0.6586174010993008, 0.6047016387887195, 0.6670769819859377, 1.17939277232758]

residual_of_noise = [TRUE_GRADS[i] - NOISE_GRADS[i] for i in range(start, len(filtered_state_means))]
print([torch.linalg.norm(r) for r in residual_of_noise])
[2.8012345, 3.09025, 3.482447, 5.4846616]
print([torch.linalg.norm(r, ord=1) for r in residual_of_noise])
[0.52681935, 0.6953145, 0.7249874, 1.2834848]

residual_of_public = [TRUE_GRADS[i] - PUBLIC_GRADS[i] for i in range(start, len(filtered_state_means))]
print([torch.linalg.norm(r) for r in residual_of_public])
[2.2761939, 3.5512056, 3.4418054, 4.921996]
print([torch.linalg.norm(r, ord=1) for r in residual_of_public])
[0.5435084, 0.7558049, 0.80391026, 1.1028893]


residual_of_noise_and_filter = [
    NOISE_GRADS[i] - filtered_state_means[i].reshape(TRUE_GRADS[i].shape)
    for i in range(start, len(filtered_state_means))
]
[
    torch.linalg.norm(
        r,
    )
    for r in residual_of_noise_and_filter
]
print([torch.linalg.norm(r) for r in residual_of_noise_and_filter])

[2.080535735027285, 1.1758106012814022, 1.7002046752308193, 1.4850198541705035]
print([torch.linalg.norm(r, ord=1) for r in residual_of_noise_and_filter])
[0.42474727591445927, 0.19729312857206366, 0.2786663047301083, 0.25073068703558954]
