import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
import time
import pickle
import random
plt.ion()
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)


# Function to keep the random seed
def seed_everything(seed=14):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()


def create_model(li):
    mod = nn.Sequential()
    for i in range(len(li)):
        out_put = li[i+1] if i != len(li)-1 else 1
        mod.append(nn.Linear(li[i], out_put))
        mod.append(nn.Tanh())
    return mod


class PINN(nn.Module):
    def __init__(self, d, algorithm):
        super(PINN, self).__init__()
        self.d = d  # dimension
        # model structure selection
        self.nets = {1: create_model([1, 10, 10, 1]),
                     2: create_model([2, 10, 15, 1]),
                     3: create_model([3, 15, 15, 1]),
                     4: create_model([4, 20, 20, 1]),}
        self.net = self.nets[d]
        self.length = self.d * 1000    # samples m
        self.device = 'cuda'  # If you don't have cuda devices use "cpu" insdead
        self.net.to(self.device)
        self.learning_rate = 0.001    # learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  # Optimizer
        self.mse = torch.nn.MSELoss(reduction='mean')  # MSE
        self.loss = None
        self.algorithm = algorithm    # mode selection
        x = lhs(self.d, self.length)
        self.x = []
        for i in range(self.d):
            xi = x[:, i:i+1]
            x_fin = self.load_data(xi)
            self.x.append(x_fin)
        u = np.random.rand(self.length, 1)
        u = u / np.linalg.norm(u)
        self.u = self.load_data(u, requires_grad=False)
        self.lambda_k1 = 1    # lambda_{k-1}
        self.lambda_k = None    # lambda_{k}
        self.lambdas = []
        self.losses = []

    def load_data(self, x, requires_grad=True):
        data64 = torch.tensor(x, requires_grad=requires_grad, dtype=torch.float64)
        return data64.to(self.device)

    def read_data(self, data):
        tmp_data = data.detach().cpu().numpy()
        if np.isnan(tmp_data).any():
            raise Exception
        return tmp_data

    def forward(self, x):
        if isinstance(x, list):
            x_fin = torch.cat(x, 1)
        else:
            x_fin = x
        return self.net(x_fin)

    def net_forward(self, x):
        v = self.forward(x)
        # DirichletBoundary
        phi = 1
        for i in range(self.d):
            phi *= (torch.exp(x[i]) - 1) * (torch.exp(1 - x[i]) - 1)
        u = v * phi
        return u

    def update(self):
        # 初始化更新准备
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()
        u = self.net_forward(self.x)
        ux, uxx = None, None
        for i in range(self.d):
            u_xi = torch.autograd.grad(u.sum(), self.x[i], create_graph=True)[0]
            if ux is None:
                ux = u_xi
            else:
                ux += u_xi
            u_xixi = torch.autograd.grad(u_xi.sum(), self.x[i], create_graph=True)[0]
            if uxx is None:
                uxx = u_xixi
            else:
                uxx += u_xixi

        # edit your operator here
        lu = - uxx
        # mode selection
        if self.algorithm == "DRQI":
            omega = 0.5    # relaxation factor
            self.loss = self.mse(lu - (self.lambda_k1 + 1) * (omega * u + (1 - omega) * self.u),
                                 torch.zeros_like(u).to(self.device))
        elif self.algorithm == "IPMNN":
            self.loss = self.mse(lu / torch.norm(lu) - self.u, torch.zeros_like(u).to(self.device))
        elif self.algorithm == "DRM":
            gamma = 0
            if self.d == 4:
                gamma = 100
            self.loss = torch.sum(lu * u) / torch.sum(u * u) + gamma * (torch.sum(u * u) - 1) ** 2

        self.loss.backward()
        lambda_k = self.read_data(torch.sum(lu * u)/torch.sum(u * u)).max()
        self.lambda_k1 = lambda_k
        self.lambdas.append(self.lambda_k1)
        self.losses.append(self.loss.item())
        normalized_u = u / torch.norm(u)
        self.u = self.load_data(self.read_data(normalized_u), requires_grad=False)
        self.optimizer.step()


def train(model, epochs):
    standard = model.d * np.pi**2    # edit the theoretical solution here
    for epoch in range(epochs):
        model.update()
        print(f'Dimension{model.d} Epoch {epoch + 1}/{epochs}, Loss: {model.loss.item()}, '
              f'Lambda_Error: {abs(model.lambda_k1-standard)}, '
              f'Eigen Value: {model.lambda_k1}')


if __name__ == "__main__":
    epochs = 10000   # maximum iteration K
    d = 1    # dimension
    model = PINN(d, "DRM")  # create model
    train(model, epochs=epochs)
    # save model
    with open(f'DRM_LP_1D_GAMMA0_SEED30_Adam.pkl', 'wb') as fi:
        pickle.dump(model, fi)
