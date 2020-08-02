import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(local_world_size, gpu_rank):
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}"
    )

    model = ToyModel().cuda(gpu_rank)
    ddp_model = DistributedDataParallel(model, [gpu_rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(gpu_rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    print("my_parameters", list(model.parameters()))


def spmd_main(local_world_size, gpu_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    demo_basic(local_world_size, gpu_rank)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This is passed in via launch.py
    parser.add_argument("--local_rank", type=int, default=0)
    # This needs to be explicitly passed in
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    print(args)
    # The main entry point is called directly without using subprocess
    spmd_main(args.local_world_size, args.local_rank)