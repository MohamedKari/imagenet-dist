import argparse
import os
import sys
import time

from torch import tensor
from torch.nn import Module, Parameter, MSELoss
from torch.optim import SGD

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class ToyModel(Module):
    def __init__(self, param_init: float):
        super(ToyModel, self).__init__()
        print("param_init", param_init)
        # self.param = Parameter(tensor([param_init])) # pylint: disable=not-callable
        self.param = Parameter(tensor([param_init])) # pylint: disable=not-callable
    
    def forward(self, x: tensor):
        return self.param * (x ** 2)
    
    def log_param(self):
        print("self.param", self.param)
        print("self.param.grad", self.param.grad)


def train(local_rank):
    x = tensor([float(os.environ["RANK"])]).to(local_rank) # pylint: disable=not-callable
    y_true = (3. * (x ** 2.)).to(local_rank)

    print(f"local_rank {local_rank}: ", "x", x)
    print(f"local_rank {local_rank}: ", "y_true", y_true)

    model = ToyModel(2.).to(local_rank)
    print("Creating model now...")
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    print("model now...")
    loss_fn = MSELoss()
    lr = .001
    optimizer = SGD(model.parameters(), lr=lr)

    # Local only
    #with model.no_sync():
    #    y_pred = model(x)
    #    print(f"local_rank {local_rank}: ", "y_pred", y_pred)
    #    loss = loss_fn(y_pred, y_true)
    #    print(f"local_rank {local_rank}: ", "loss", loss)
    #    loss.backward()
    #    print(f"local_rank {local_rank}: ", "pre-optimization w", list(model.parameters())[0].data)
    #    print(f"local_rank {local_rank}: ", "pre-optimization non-synchronized grad_w", list(model.parameters())[0].grad)

    # Now again, but synchronized
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
    loss.backward()
    print(f"local_rank {local_rank}: ", "pre-optimization synchronized grad_w", list(model.parameters())[0].grad)

    optimizer.step()
    print(f"local_rank {local_rank}: ", "post-optimization w", list(model.parameters())[0].data)

    optimizer.zero_grad()


def setup(local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl", init_method="env://")
    
    print(
        f"[PID {os.getpid()}]:"
        f" world_size = {dist.get_world_size()}"
        f" rank = {dist.get_rank()}"
        f" backend={dist.get_backend()}"
    )

    train(local_rank)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    print(sys.argv, os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    setup(args.local_rank)