import cvxpy as cp
import numpy as np
import torch

public_grad = torch.load("grads_normalize=True_dp=False.pt")
true_grad = torch.load("grads_normalize=False_dp=False.pt")
noise_grad = torch.load("grads_normalize=False_dp=True.pt")
noise_grad_final_0.8
noise_grad_final_0.3

# Problem data.
m = 10
n = 28 * 28
np.random.seed(1)
N_STEPS = 5
matrix_tuples = [(public_grad[i], public_grad[i - 1]) for i in range(N_STEPS)]

# Construct the problem.
F = cp.Variable((m, m))
objective = cp.Minimize(cp.sum([cp.norm2(A_t1 - F @ A_t0) for A_t1, A_t0 in matrix_tuples]))
constraints = None
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(F.value)

sum([np.linalg.norm(A_t1 - np.matmul(F.value, A_t0)) for A_t1, A_t0 in matrix_tuples])


#####################
#  true
#####################
true_matrix_tuples = [(true_grad[i], true_grad[i - 1]) for i in range(N_STEPS)]
# Construct the problem.
F_true = cp.Variable((m, m))
objective_true = cp.Minimize(cp.sum([cp.norm2(A_t1 - F_true @ A_t0) for A_t1, A_t0 in true_matrix_tuples]))
constraints = None
prob_true = cp.Problem(objective_true, constraints)

# The optimal objective value is returned by `prob.solve()`.
result_true = prob_true.solve()
# The optimal value for x is stored in `x.value`.
print(F_true.value)


#####################
#  noise
#####################
noise_matrix_tuples = [(noise_grad[i], noise_grad[i - 1]) for i in range(N_STEPS)]
# Construct the problem.
F_noise = cp.Variable((m, m))
objective_noise = cp.Minimize(cp.sum([cp.norm2(A_t1 - F_noise @ A_t0) for A_t1, A_t0 in noise_matrix_tuples]))
constraints = None
prob_noise = cp.Problem(objective_noise, constraints)

# The optimal objective value is returned by `prob.solve()`.
result_noise = prob_noise.solve()
# The optimal value for x is stored in `x.value`.
print(F_noise.value)


sum([np.linalg.norm(A_t1 - np.matmul(F_true.value, A_t0)) for A_t1, A_t0 in true_matrix_tuples])

sum([np.linalg.norm(A_t1 - np.matmul(F.value, A_t0)) for A_t1, A_t0 in true_matrix_tuples])

# norm between True transition and estimated transition
np.linalg.norm(F_true.value - F.value)
5.415394073344793


# norm between True transition and random matrix
np.linalg.norm(F_true.value - np.random.randn(*F_true.value.shape))
9.459912598354743
10.09387853890853
10.430716855050552
11.64129230637961
11.337725362661496
11.291050562575393
10.032289422335339

# estimate during steps
np.mean([np.linalg.norm(true_grad[i + 1] - np.matmul(F.value, public_grad[i])) for i in range(len(true_grad) - 1)])
# mean = 1.054808385112666

np.mean([np.linalg.norm(true_grad[i + 1] - np.matmul(F_true.value, true_grad[i])) for i in range(len(true_grad) - 1)])
# 0.9518897963402736

np.mean([np.linalg.norm(true_grad[i + 1] - np.matmul(F.value, true_grad[i])) for i in range(len(true_grad) - 1)])
# 1.8440215921331211

np.mean([np.linalg.norm(true_grad[i + 1] - np.matmul(F_noise.value, true_grad[i])) for i in range(len(true_grad) - 1)])
# 1.0414392351221649

np.mean([np.linalg.norm(true_grad[i] - noise_grad[i]) for i in range(len(true_grad))])
# 1.5111395

# if use estimated transition, with true gradients
sum([np.linalg.norm(A_t1 - np.matmul(F.value, A_t0)) for A_t1, A_t0 in true_matrix_tuples])
17.14006110408391

# norm between noise and true
sum([np.linalg.norm(A_t1 - np.matmul(F.value, A_t0)) for A_t1, A_t0 in true_matrix_tuples])

# norm between estimated and true
