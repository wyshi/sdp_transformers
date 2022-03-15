from pykalman import KalmanFilter

# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
import numpy as np
import torch

# my_filter = KalmanFilter(dim_x=2, dim_z=1)
# my_filter.x = np.array([[2.0], [0.0]])  # initial state (location and velocity)

# my_filter.F = np.load("/local/data/wyshi/sdp_transformers/mnist/tm.npy")  # state transition matrix

# my_filter.H = np.array([[1.0, 0.0]])  # Measurement function
# my_filter.P *= 1000.0  # covariance matrix
# my_filter.R = np.identity(N) * SIGMA  # state uncertainty
# dt = 1
# my_filter.Q = Q_discrete_white_noise(2, dt, 0.1)  # process uncertainty


noise_grads = torch.load("/local/data/wyshi/sdp_transformers/mnist/grads_normalize=False_dp=True_sample-rate=0.001.pt")
true_grads = torch.load("/local/data/wyshi/sdp_transformers/mnist/grads_normalize=False_dp=False_sample-rate=0.001.pt")
m = 10
n = 28 * 28

N = m * n  # * 784
SIGMA = 1
TIMESTEP = 10
tm = np.load("/local/data/wyshi/sdp_transformers/mnist/tm.npy")
tc = np.load("/local/data/wyshi/sdp_transformers/mnist/tc.npy")
kf = KalmanFilter(
    # initial_state_mean=0,
    transition_matrices=tm,
    transition_covariance=tc,
    n_dim_obs=N,
    transition_offsets=np.zeros(N),
    observation_offsets=np.zeros(N),
    observation_matrices=np.identity(N),
    observation_covariance=np.identity(N) * SIGMA,
    # em_vars=[
    #     "transition_matrices",
    #     "transition_covariance",  # "initial_state_mean", "initial_state_covariance"
    # ],
)

np.random.seed(1)

# measurements = [np.random.randn(m, n).reshape(-1) for _ in range(TIMESTEP)]
measurements = [noise_grads[i].reshape(-1) for i in range(TIMESTEP)]
# kf.em(measurements, n_iter=2)
p1, c1 = kf.filter_update([noise_grads[0].reshape(-1), noise_grads[1].reshape(-1).tolist()])

n_timesteps = 3
n_dim_state = N
filtered_state_means = np.zeros((n_timesteps, n_dim_state))
filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
for t in range(n_timesteps - 1):
    if t == 0:
        filtered_state_means[t] = noise_grads[0].reshape(-1)
        filtered_state_covariances[t] = np.identity(N)
    filtered_state_means[t + 1], filtered_state_covariances[t + 1] = kf.filter_update(
        filtered_state_means[t],
        filtered_state_covariances[t],
        noise_grads[t + 1].reshape(-1),
        # transition_offset=data.transition_offsets[t],
    )
