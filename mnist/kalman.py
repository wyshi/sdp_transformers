from pykalman import KalmanFilter
import numpy as np
import torch

noise_grads = torch.load("/local/data/wyshi/sdp_transformers/mnist/grads_normalize=False_dp=True_sample-rate=0.001.pt")
m = 10
n = 28 * 28

N = m * n  # * 784
SIGMA = 1
TIMESTEP = 10
kf = KalmanFilter(
    # initial_state_mean=0,
    n_dim_obs=N,
    transition_offsets=np.zeros(N),
    observation_offsets=np.zeros(N),
    observation_matrices=np.identity(N),
    observation_covariance=np.identity(N) * SIGMA,
    em_vars=[
        "transition_matrices",
        "transition_covariance",  # "initial_state_mean", "initial_state_covariance"
    ],
)

np.random.seed(1)

# measurements = [np.random.randn(m, n).reshape(-1) for _ in range(TIMESTEP)]
measurements = [noise_grads[i].reshape(-1) for i in range(TIMESTEP)]
kf.em(measurements, n_iter=2)
kf.smooth([[2, 0], [2, 1], [2, 2]])
