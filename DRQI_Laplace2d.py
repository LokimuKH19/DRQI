import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
import time
import pickle
import random
import os
from datetime import datetime
plt.ion()
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)


# print the configuration and save to log
def print_config(model, seed, epochs, real_epoch, epsilon, save_log=True, log_dir="logs"):
    config_lines = []
    config_lines.append("========== Configuration ==========")
    config_lines.append(f"Seed: {seed}")
    config_lines.append(f"Device: {model.device}")
    config_lines.append(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        config_lines.append(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    config_lines.append("-----------------------------------")
    config_lines.append(f"Model Dimension (d): {model.d}")
    config_lines.append(f"Model Architecture:\n{model.net}")
    config_lines.append(f"Number of Samples: {model.length}")
    config_lines.append(f"Input Sampling Method: Latin Hypercube Sampling (LHS)")
    config_lines.append("-----------------------------------")
    config_lines.append(f"Optimizer: {model.optimizer.__class__.__name__}")
    config_lines.append(f"Learning Rate: {model.learning_rate}")
    config_lines.append(f"Loss Function: {model.mse.__class__.__name__}")
    config_lines.append("-----------------------------------")
    config_lines.append(f"Algorithm: {model.algorithm}")
    config_lines.append(f"Unique Parameters: {model.unique_parameter}")
    config_lines.append("-----------------------------------")
    config_lines.append(f"Max Epochs: {epochs}")
    config_lines.append(f"Real Epochs: {real_epoch}")
    config_lines.append(f"Early Stopping Threshold (eq_loss < epsilon): {epsilon}")
    standard = model.d * np.pi**2
    config_lines.append(f"Theoretical Eigenvalue: {standard}")
    config_lines.append("===================================\n")

    # 打印信息
    for line in config_lines:
        print(line)

    # 写入日志
    if save_log:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model.algorithm}_LP_{model.d}D_SEED{seed}_{timestamp}.log"
        path = os.path.join(log_dir, filename)
        with open(path, "w") as f:
            for line in config_lines:
                f.write(line + "\n")
        print(f"[Log Saved] Configuration saved to: {path}\n")


SEED = 30
# Function to keep the random seed
def seed_everything(seed=SEED):
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
        self.unique_parameter = {
            "omega": 0.5,
            "gamma": 0,
        }   # to hold the unique parameter in the algorithm
        # to hold the equation loss (PINN)
        self.eq_losses = []
        self.eq_loss = None
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
        # Initialization
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
            omega = self.unique_parameter["omega"]    # relaxation factor and it could be changed then
            self.loss = self.mse(lu - (self.lambda_k1 + 1) * (omega * u + (1 - omega) * self.u),
                                 torch.zeros_like(u).to(self.device))
        elif self.algorithm == "IPMNN":
            self.loss = self.mse(lu / torch.norm(lu) - self.u, torch.zeros_like(u).to(self.device))
        elif self.algorithm == "DRM":
            gamma = self.unique_parameter["gamma"]
            if self.d == 4:
                gamma = 100
            self.loss = torch.sum(lu * u) / torch.sum(u * u) + gamma * (torch.sum(u * u) - 1) ** 2

        self.loss.backward()
        lambda_k = self.read_data(torch.sum(lu * u)/torch.sum(u * u)).max()
        # calculate the equation loss
        self.eq_loss = self.mse(lu-lambda_k*u, torch.zeros_like(u).to(self.device))
        self.eq_losses.append(self.eq_loss)
        self.lambda_k1 = lambda_k
        self.lambdas.append(self.lambda_k1)
        self.losses.append(self.loss.item())
        normalized_u = u / torch.norm(u)
        self.u = self.load_data(self.read_data(normalized_u), requires_grad=False)

        self.optimizer.step()


def train(model, epochs, epsilon):
    standard = model.d * np.pi**2    # edit the theoretical solution here
    for epoch in range(epochs):
        model.update()
        print(f'Dimension{model.d} Epoch {epoch + 1}/{epochs}, Loss: {model.loss.item()}, '
              f'Lambda_Error: {abs(model.lambda_k1-standard)}, '
              f'Eigen Value: {model.lambda_k1}, Equation Loss:{model.eq_loss}')
        # reflects the
        if model.eq_loss < epsilon:
            return epoch+1
    return epochs


if __name__ == "__main__":
    epochs = 10000   # maximum iteration K
    d = 1    # dimension
    epsilon = 0.0001  # for different methods the epsilon cannot be consistent
    algorithm = "DRQI"
    model = PINN(d, algorithm)  # create model
    real_epoch = train(model, epochs=epochs, epsilon=epsilon)
    # save model
    with open(f'models/{algorithm}_LP_{d}D_OMEGA{model.unique_parameter["omega"]}_SEED{SEED}_Adam.pkl', 'wb') as fi:
        pickle.dump(model, fi)
    # print info for reproduction
    print_config(model, SEED, epochs, real_epoch, epsilon)

