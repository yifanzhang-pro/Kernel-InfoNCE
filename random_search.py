import torch
from pathlib import Path
from argparse import ArgumentParser
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse
from torch import nn, optim
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import simclr_module


def random_search():
    parser = ArgumentParser()
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--max_t", default=200, type=int, help="max epoch to report")
    parser.add_argument("--num_samples", default=100, type=int, help="number of samples")
    parser.add_argument("--search_gammas", default=[0.5, 1.0], type=float, nargs='+', help="number of samples")
    parser.add_argument("--search_mus", default=[1.0], type=float, nargs='+', help="projection mu")
    parser.add_argument("--loss_type", default="origin", type=str, help="search type, origin, sum or product")
    parser.add_argument("--search_acos_orders", default=[0], type=int, nargs='+', help="number of samples")
    # specify flags to store false
    parser.add_argument("--first_conv", action="store_false")
    parser.add_argument("--maxpool1", action="store_false")
    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--norm_p", default=2., type=float, help="norm p, -1 for inf")
    parser.add_argument("--distance_p", default=2., type=float, help="distance p, -1 for inf")
    parser.add_argument("--acos_order", default=0, type=int, help="order of acos, 0 for not use acos kernel")
    parser.add_argument("--gamma", default=2., type=float, help="gamma")
    parser.add_argument("--online_ft", action="store_true")
    parser.add_argument("--fp32", action="store_true")

    # transform params
    parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
    parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
    parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
    parser.add_argument("--data_dir", type=str, default="/home/yjq/graph", help="path to download data")

    # training params
    parser.add_argument("--fast_dev_run", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
    parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
    parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

    parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
    parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

    args = parser.parse_args()

    max_t = args.max_t
    num_samples = args.num_samples

    search_params = {
        "learning_rate": tune.loguniform(1e-2, 10),
        "temperature": tune.loguniform(1e-2, 1),
        "gamma": tune.choice(args.search_gammas),
        "projection_mu": tune.choice(args.search_mus),
        "gamma_lambd": tune.uniform(0, 1),
        "acos_order": tune.choice(args.search_acos_orders)
    }
    scheduler = ASHAScheduler(
        max_t=max_t,
        grace_period=20,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(simclr_module.cli_main, args=args, isTune=True),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="online_val_acc",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=ray.air.RunConfig(
          local_dir="~/ray_results"
        ),
        param_space=search_params,
    )

    results = tuner.fit()
    best_result = results.get_best_result("online_val_acc", "max")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["online_val_acc"]))


if __name__ == "__main__":
    random_search()