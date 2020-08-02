import os
import time

from torch import tensor
from torch.nn import Module, Parameter, MSELoss
from torch.optim import SGD


class QuadraticModel(Module):
    def __init__(self, param_init: float):
        super(QuadraticModel, self).__init__()
        print("param_init", param_init)
        # self.param = Parameter(tensor([param_init])) # pylint: disable=not-callable
        self.param = Parameter(tensor([param_init])) # pylint: disable=not-callable
    
    def forward(self, x: tensor):
        return self.param * (x ** 2)
    
    def log_param(self):
        print("self.param", self.param)
        print("self.param.grad", self.param.grad)

def train(x, y_true):
    print("x", x)
    print("y_true", y_true)

    model = QuadraticModel(2.)
    loss_fn = MSELoss()
    lr = .001
    steps = 1
    optimizer = SGD(model.parameters(), lr=lr)

    for step in range(steps):
        y_pred = model(x)
        print("y_pred", y_pred)
        loss = loss_fn(y_pred, y_true)
        print("loss", loss)
        loss.backward()
        model.log_param()

        optimizer.step()
        model.log_param()
        optimizer.zero_grad()

    print("\n\n")

x = tensor([0., 1., 2., 3.]).reshape((4, 1))
train(x,  3 * (x ** 2))

import sys
sys.exit(0)

x_all = tensor([0., 1., 2., 3., 4., 5., 6., 7.]).reshape((8, 1)) # pylint: disable=not-callable

for x in x_all:
    train(x,  3 * (x ** 2))