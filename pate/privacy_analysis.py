from autodp.mechanism_zoo import GaussianMechanism, LaplaceMechanism
import numpy as np


class PATE(GaussianMechanism):
    def __init__(self, sigma, m, Binary, name="PATE"):
        # sigma is the std of the Gaussian noise added to the voting scores
        if Binary:
            # This is a binary classification task
            sensitivity = 1
        else:  # for the multiclass case, the L2 sensitivity is sqrt(2)
            sensitivity = np.sqrt(2)
        GaussianMechanism.__init__(
            self, sigma=sigma / sensitivity / np.sqrt(m), name=name
        )

        self.params = {"sigma": sigma}


class PATELaplace(LaplaceMechanism):
    def __init__(self, sigma, m, Binary, name="PATE"):
        # sigma is the std of the Gaussian noise added to the voting scores
        if Binary:
            # This is a binary classification task
            sensitivity = 1
        else:  # for the multiclass case, the L2 sensitivity is sqrt(2)
            sensitivity = np.sqrt(2)
        LaplaceMechanism.__init__(self, b=sigma / sensitivity / np.sqrt(m), name=name)

        self.params = {"sigma": sigma}


# Computing the privacy loss after running the algorithm

# let's say after running PATE-PSQ or ASQ with Gaussian mechanism to release m labels
# the noise added to the voted histogram (in the multi-class case),
# or the noise added to the # of teachers who voted positive is sigma
#
m = 10
sigma = 4

# let's say it is a binary classification task


pate_mech = PATE(sigma=sigma, m=m, Binary=False, name="PATE")

delta = 1e-5
eps = pate_mech.get_approxDP(delta)

print(eps, delta)


# pate_mech = PATE(sigma=sigma, m=m, Binary=True, name="PATE")
# delta = 1e-6
# eps = pate_mech.get_approxDP(delta)

# print(eps, delta)


# # Privacy calibtation:  given m,  eps,  choose sigma

# from autodp.calibrator_zoo import eps_delta_calibrator

# calibrate = eps_delta_calibrator()


# class PATE_nonbinary_m(PATE):
#     def __init__(self, sigma, name="PATE_m"):
#         PATE.__init__(self, sigma=sigma, m=10, Binary=False, name=name)


# # Find the \sigma parameter that gives the following privacy guarantee
# eps = 2.0
# delta = 1e-6

# mech1 = calibrate(PATE_nonbinary_m, eps, delta, [0, 10], name="PATE_eps=2")
# print(mech1.name, mech1.params, mech1.get_approxDP(delta))


# # Find the \sigma parameter that gives the following privacy guarantee
# eps = 0.5
# delta = 1e-6

# mech2 = calibrate(PATE_nonbinary_m, eps, delta, [0, 10], name="PATE_eps=0.5")

# print(mech2.name, mech2.params, mech2.get_approxDP(delta))
