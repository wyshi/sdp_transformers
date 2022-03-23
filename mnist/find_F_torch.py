#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
optimize to get transition matrix
python find_F_torch.py --lr 1 -n 200 --save-model
"""

import argparse
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from torch.utils.data.dataset import Dataset
import random


SEED = 1111
USE_RUNNING = True
if USE_RUNNING:
    PUBLIC_GRADS = torch.load("grads_normalize=True_dp=False_sample-rate=0.001_epoch=5.pt")
else:
    PUBLIC_GRADS = torch.load(
        "grads_normalize=True_dp=True_sample-rate=0.001_epoch=5_pubengine=False_filter=False_seed=123456.pt"
    )
TRUE_GRADS = torch.load("grads_normalize=False_dp=False_sample-rate=0.001_epoch=5.pt")
NOISE_GRADS = torch.load("grads_normalize=False_dp=True_sample-rate=0.001_epoch=5.pt")
N = PUBLIC_GRADS[0].shape[0] * PUBLIC_GRADS[0].shape[1]
CRITERION = "mse"

if CRITERION == "mse":
    criterion = nn.MSELoss(reduction="sum")
else:
    criterion = nn.CosineSimilarity(eps=1e-08)


class GradDataset(Dataset):
    def __init__(self, grads):
        self.examples = [(grads[i - 1].reshape(-1), grads[i].reshape(-1)) for i in range(1, len(grads))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class ObservationDataset(Dataset):
    def __init__(self, x_grads, y_grads):
        self.examples = [(x_grad.reshape(-1), y_grad.reshape(-1)) for x_grad, y_grad in zip(x_grads, y_grads)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SimpleSampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        # self.conv2 = nn.Conv2d(16, 32, 4, 2)
        # self.fc1 = nn.Linear(32 * 4 * 4, 32)
        # self.fc2 = nn.Linear(32, 10)
        self.fc1 = nn.Linear(N, N, bias=False)

    def forward(self, x):
        # x of shape [B, N]
        x = self.fc1(x)
        return x

    def name(self):
        return "SimpleSampleConvNet"


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # criterion = nn.MSELoss(reduction="sum")
    losses = []
    for _batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # pdb.set_trace()
        loss = criterion(output, target)
        if CRITERION == "mse":
            loss = loss / target.shape[0]
        else:
            loss = -loss.mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(args, model, device, test_loader):
    model.eval()
    # criterion = nn.MSELoss(reduction="sum")
    if CRITERION == "mse":
        test_loss = 0
        test_loss_random = 0
    else:
        test_loss = []
        test_loss_random = []

    random_model = nn.Linear(N, N, bias=False).to(device)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if CRITERION == "mse":
                test_loss += criterion(output, target).item()  # sum up batch loss
            else:
                test_loss.append(-criterion(output, target).mean().item())
            # random
            output_random = random_model(data)
            if CRITERION == "mse":
                test_loss_random += criterion(output_random, target).item()  # sum up batch loss
            else:
                test_loss_random.append(-criterion(output_random, target).mean().item())

    if CRITERION == "mse":
        test_loss /= len(test_loader.dataset)
        test_loss_random /= len(test_loader.dataset)
    else:
        test_loss = np.mean(test_loss)
        test_loss_random = np.mean(test_loss_random)

    print(f"\nTest set: Average loss: {test_loss:.4f}, random loss: {test_loss_random:.4f}\n")
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "-sr",
        "--sample-rate",
        type=float,
        default=0.1,
        metavar="SR",
        help="sample rate used for batch construction (default: 0.001)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: .1)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        choices={"noised", "public"},
        help="if to use noised grads or public grads",
    )
    parser.add_argument(
        "--normalize",
        "-nor",
        action="store_true",
        default=False,
        help="normalize the dataset",
    )
    parser.add_argument(
        "--train_obs",
        "-obs",
        action="store_true",
        default=False,
        help="train observation H",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    kwargs = {"num_workers": 1, "pin_memory": True}

    # make_deterministic(SEED)

    publicgrads_dataset = GradDataset(PUBLIC_GRADS)
    noisegrads_dataset = GradDataset(NOISE_GRADS)
    truegrads_dataset = GradDataset(TRUE_GRADS)
    observatoin_dataset = ObservationDataset(x_grads=PUBLIC_GRADS, y_grads=NOISE_GRADS)

    if not args.train_obs:
        if args.train_data == "noised":
            train_dataset = noisegrads_dataset
        else:
            train_dataset = publicgrads_dataset
        test_dataset = truegrads_dataset
    else:
        split = int(len(observatoin_dataset) * 0.8)
        train_dataset = observatoin_dataset[:split]
        test_dataset = observatoin_dataset[split:]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs,
    )
    run_results = []
    for _ in range(args.n_runs):
        model = SimpleSampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        for epoch in tqdm(range(1, args.epochs + 1)):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
        run_results.append(test(args, model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"{model.name()}_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{CRITERION}_obs={args.train_obs}_USE_RUNNING={USE_RUNNING}"
    )
    torch.save(run_results, f"found_transition/run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"found_transition/{repro_str}.pt")
        print(f"found_transition/{repro_str}.pt")


if __name__ == "__main__":
    main()
