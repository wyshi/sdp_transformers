import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm
from opacus import PrivacyEngine
from KalmanFilterTorch import KalmanFilterTorch


class PublicEngine(PrivacyEngine):
    def step(self, is_empty: bool = False):
        """
        Takes a step for the privacy engine.

        Args:
            is_empty: Whether the step is taken on an empty batch
                In this case, we do not call clip_and_accumulate since there are no
                per sample gradients.

        Notes:
            You should not call this method directly. Rather, by attaching your
            ``PrivacyEngine`` to the optimizer, the ``PrivacyEngine`` would have
            the optimizer call this method for you.

        Raises:
            ValueError: If the last batch of training epoch is greater than others.
                This ensures the clipper consumed the right amount of gradients.
                In the last batch of a training epoch, we might get a batch that is
                smaller than others but we should never get a batch that is too large

        """
        self.steps += 1
        if not is_empty:
            self.clipper.clip_and_accumulate()
            clip_values, batch_size = self.clipper.pre_step()
        else:
            clip_values = (
                self.max_grad_norm
                if type(self.max_grad_norm) is list
                else [self.max_grad_norm for p in self.module.parameters() if p.requires_grad]
            )
            batch_size = self.avg_batch_size

        # doesn't add any noise, only clip to get the true grads
        # params = (p for p in self.module.parameters() if p.requires_grad)
        # for p, clip_value in zip(params, clip_values):
        #     noise = self._generate_noise(clip_value, p)
        #     if self.loss_reduction == "mean":
        #         noise /= batch_size

        #     if self.rank == 0:
        #         # Noise only gets added on first worker
        #         # This is easy to reason about for loss_reduction=sum
        #         # For loss_reduction=mean, noise will get further divided by
        #         # world_size as gradients are averaged.
        #         p.grad += noise

        #     # For poisson, we are not supposed to know the batch size
        #     # We have to divide by avg_batch_size instead of batch_size
        #     if self.poisson and self.loss_reduction == "mean":
        #         p.grad *= batch_size / self.avg_batch_size


class PrivacyEngineWithFilter(PrivacyEngine):
    def setup_filter(self, PUBLIC_GRADS=None, prev_state_means=None, prev_state_covariances=None):
        with torch.no_grad():
            device = self.device
            N = PUBLIC_GRADS[-1].shape[0] * PUBLIC_GRADS[-1].shape[1]
            # F_public = torch.load("found_transition/SimpleSampleConvNet_1.0_1.0_1.0_0.1_200.pt")["fc1.weight"].to(
            #     device
            # )
            F_public = torch.eye(N).to(device)
            # PUBLIC_GRADS = torch.stack(
            #     [
            #         torch.from_numpy(_grad)
            #         for _grad in torch.load("grads_normalize=True_dp=False_sample-rate=0.001_epoch=5.pt")
            #     ]
            # ).to(device)

            # H_public = torch.load("found_transition/SimpleSampleConvNet_1.0_1.0_1.0_0.1_50_mse_obs=True.pt")[
            #     "fc1.weight"
            # ].to(device)
            # H_public = torch.load(
            #     "found_transition/SimpleSampleConvNet_1.0_1.0_1.0_0.1_50_mse_obs=True_USE_RUNNING=True.pt"
            # )["fc1.weight"].to(device)
            H_public = torch.eye(N).to(device)
            #################################################################
            # estimate transition covariance
            #################################################################
            residual_public = []
            for i in tqdm(range(len(PUBLIC_GRADS) - 1)):
                residual_public.append(PUBLIC_GRADS[i + 1].reshape(-1) - F_public @ PUBLIC_GRADS[i].reshape(-1))

            estimated_transition_covariance = torch.cov(torch.stack(residual_public).T)

            estimated_initial_state_covariance = torch.cov(torch.stack([grad.reshape(-1) for grad in PUBLIC_GRADS]).T)

            self.kf = KalmanFilterTorch(
                n_dim_obs=N,
                # initial
                initial_state_mean=PUBLIC_GRADS[0].reshape(-1),
                initial_state_covariance=estimated_initial_state_covariance,
                # transition
                transition_matrices=F_public,
                transition_covariance=estimated_transition_covariance,  # we don't have to use this estimate, since kf will estimate for us
                transition_offsets=torch.zeros(N).to(device),
                # observation
                observation_matrices=H_public,  # torch.eye(N).to(device),
                # observation_covariance=torch.eye(N).to(device) * SIGMA,
                observation_offsets=torch.zeros(N).to(device),
                device=device,
                # em_vars=[
                #     "transition_matrices",
                #     "transition_covariance",  # "initial_state_mean", "initial_state_covariance"
                # ],
            )
            self.prev_state_means = prev_state_means
            self.prev_state_covariances = prev_state_covariances

    def denoise(self, noisy_grad, sigma):
        with torch.no_grad():
            assert (self.prev_state_means is not None) and (self.prev_state_covariances is not None)

            # import pdb

            # pdb.set_trace()
            cur_filtered_state_means, cur_filtered_state_covariances = self.kf.filter_update(
                filtered_state_mean=self.prev_state_means,
                filtered_state_covariance=self.prev_state_covariances,
                observation=noisy_grad.reshape(-1),
                observation_covariance=torch.eye(self.kf.n_dim_obs).to(self.device) * sigma,
            )
            # update
            self.prev_state_means = cur_filtered_state_means
            self.prev_state_covariances = cur_filtered_state_covariances
            return cur_filtered_state_means, cur_filtered_state_covariances

    def step(self, is_empty: bool = False):
        """
        Takes a step for the privacy engine.

        Args:
            is_empty: Whether the step is taken on an empty batch
                In this case, we do not call clip_and_accumulate since there are no
                per sample gradients.

        Notes:
            You should not call this method directly. Rather, by attaching your
            ``PrivacyEngine`` to the optimizer, the ``PrivacyEngine`` would have
            the optimizer call this method for you.

        Raises:
            ValueError: If the last batch of training epoch is greater than others.
                This ensures the clipper consumed the right amount of gradients.
                In the last batch of a training epoch, we might get a batch that is
                smaller than others but we should never get a batch that is too large

        """
        self.steps += 1
        # import pdb

        # pdb.set_trace()

        if self.steps % self.interval_T == 0:
            self.setup_filter()
        original_params = (p for p in self.module.parameters() if p.requires_grad)
        original_grad_copys = []
        for p in original_params:
            original_grad_copy = p.grad.data.clone().detach()
            original_grad_copys.append(original_grad_copy)
        if not is_empty:
            self.clipper.clip_and_accumulate()
            clip_values, batch_size = self.clipper.pre_step()
            # import pdb

            # pdb.set_trace()
        else:
            clip_values = (
                self.max_grad_norm
                if type(self.max_grad_norm) is list
                else [self.max_grad_norm for p in self.module.parameters() if p.requires_grad]
            )
            batch_size = self.avg_batch_size

        params = (p for p in self.module.parameters() if p.requires_grad)
        for i, (p, clip_value) in enumerate(zip(params, clip_values)):
            noise = self._generate_noise(clip_value, p)
            if self.loss_reduction == "mean":
                noise /= batch_size

            if self.rank == 0:
                # Noise only gets added on first worker
                # This is easy to reason about for loss_reduction=sum
                # For loss_reduction=mean, noise will get further divided by
                # world_size as gradients are averaged.
                # p.grad += noise
                # now let's denoise
                grad_copy = p.grad.data.clone().detach()
                grad_copy += noise
                # import pdb

                # pdb.set_trace()
                if self.prev_state_means is None:
                    assert self.prev_state_covariances is None
                    self.prev_state_means = self.kf.initial_state_mean
                    self.prev_state_covariances = self.kf.initial_state_covariance

                    denoised_grad = grad_copy
                else:
                    denoised_grad, _ = self.denoise(noisy_grad=grad_copy, sigma=clip_value * self.noise_multiplier)
                    denoised_grad = denoised_grad.reshape(p.grad.shape)
                # import pdb

                # pdb.set_trace()
                print("l2-norm: noised, ", torch.linalg.norm(grad_copy - original_grad_copys[i]))
                print("l2-norm: denoised,", torch.linalg.norm(denoised_grad - original_grad_copys[i]))
                print("l1-norm: noised,", torch.linalg.norm(grad_copy - original_grad_copys[i], ord=1))
                print("l1-norm: denoised,", torch.linalg.norm(denoised_grad - original_grad_copys[i], ord=1))

                p.grad = denoised_grad

            # For poisson, we are not supposed to know the batch size
            # We have to divide by avg_batch_size instead of batch_size
            if self.poisson and self.loss_reduction == "mean":
                p.grad *= batch_size / self.avg_batch_size
