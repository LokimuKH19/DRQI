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
import json
from datetime import datetime


def load_config():
    # load config from JSON
    with open("CONFIG_dynamic.json", "r") as f:
        config = json.load(f)
    return config


    # assign config values
    ALGORITHM = config["algorithm"]
    D = config["dimension"]
    SEED = config["seed"]
    EPOCHS = config["epochs"]
    EPSILON1 = config["epsilon1"]
    EPSILON2 = EPSILON1 * config["epsilon2_ratio"]
    LEARNING_RATE = config["learning_rate"]
    OMEGA = config["omega"]
    GAMMA = config["gamma"]
    NET_STRUCTURE = config["network_structure"]
    POINTS_PER_DIM = config["points_per_dimension"]
    OPTIMIZER = config["optimizer"]


# print the configuration and save to log
def print_config(model, seed, epochs, real_epoch, epsilon1, epsilon2, avg_time, timestamp, save_log=True, log_dir="logs"):
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
    config_lines.append(f"Early Stopping Threshold (eq_loss < epsilon1): {epsilon1}")
    config_lines.append(f"Early Stopping Threshold (eq_loss_k - ep_loss_k-1 < epsilon2): {epsilon2}")
    standard = model.d * np.pi**2
    config_lines.append(f"Theoretical Eigenvalue: {standard}")
    config_lines.append("=================Results==================\n")
    config_lines.append(f'Loss: {model.loss.item()}\n'
                      f'Lambda_Error: {abs(model.lambda_k1-standard)}\n'
                      f'Eigen Value: {model.lambda_k1}\nEquation Loss:{model.eq_loss}')
    config_lines.append(f"Average time per step: {avg_time:.6f} seconds")
    for line in config_lines:
        print(line)

    if save_log:
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{model.algorithm}_LP_{model.d}D_SEED{seed}_{timestamp}.log"
        path = os.path.join(log_dir, filename)
        with open(path, "w") as f:
            for line in config_lines:
                f.write(line + "\n")
        print(f"[Log Saved] Configuration saved to: {path}\n")


def seed_everything(seed):
    plt.ion()
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=16)
    np.set_printoptions(precision=16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        self.d = d
        self.net = create_model(NET_STRUCTURE[str(d)])
        self.length = int(self.d * POINTS_PER_DIM)
        self.device = 'cuda'
        self.net.to(self.device)
        self.learning_rate = LEARNING_RATE
        if OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.loss = None
        self.algorithm = algorithm
        x = lhs(self.d, self.length)
        self.x = [self.load_data(x[:, i:i+1]) for i in range(self.d)]
        u = np.random.rand(self.length, 1)
        u = u / np.linalg.norm(u)
        self.u = self.load_data(u, requires_grad=False)
        self.lambda_k1 = 1
        self.lambda_k = None
        self.lambdas = []
        self.losses = []
        self.unique_parameter = {
            "omega": OMEGA,
            "gamma": GAMMA,
        }
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
        phi = 1
        for i in range(self.d):
            phi *= (torch.exp(x[i]) - 1) * (torch.exp(1 - x[i]) - 1)
        return v * phi

    def update(self):
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()
        u = self.net_forward(self.x)
        ux, uxx = None, None
        for i in range(self.d):
            u_xi = torch.autograd.grad(u.sum(), self.x[i], create_graph=True)[0]
            ux = u_xi if ux is None else ux + u_xi
            u_xixi = torch.autograd.grad(u_xi.sum(), self.x[i], create_graph=True)[0]
            uxx = u_xixi if uxx is None else uxx + u_xixi

        lu = -uxx
        if self.algorithm == "DRQI":
            omega = self.unique_parameter["omega"]
            self.loss = self.mse(lu - (self.lambda_k1 + 1) * (omega * u + (1 - omega) * self.u), torch.zeros_like(u))
        elif self.algorithm == "IPMNN":
            self.loss = self.mse(lu / torch.norm(lu) - self.u, torch.zeros_like(u))
        elif self.algorithm == "DRM":
            gamma = self.unique_parameter["gamma"]
            if self.d == 4:
                gamma = 100
            self.loss = torch.sum(lu * u) / torch.sum(u * u) + gamma * (torch.sum(u * u) - 1) ** 2

        self.loss.backward()
        lambda_k = self.read_data(torch.sum(lu * u)/torch.sum(u * u)).max()
        self.eq_loss = self.mse(lu - lambda_k * u, torch.zeros_like(u))
        self.eq_losses.append(self.eq_loss)
        self.lambda_k1 = lambda_k
        self.lambdas.append(lambda_k)
        self.losses.append(self.loss.item())
        normalized_u = u / torch.norm(u)
        self.u = self.load_data(self.read_data(normalized_u), requires_grad=False)
        self.optimizer.step()


def train(model, epochs, epsilon1, epsilon2):
    standard = model.d * np.pi**2
    start = time.time()
    for epoch in range(epochs):
        model.update()
        print(f'Dimension{model.d} Epoch {epoch + 1}/{epochs}, Loss: {model.loss.item()}, '
              f'Lambda_Error: {abs(model.lambda_k1-standard)}, '
              f'Eigen Value: {model.lambda_k1}, Equation Loss:{model.eq_loss}')
        if model.eq_loss < epsilon1 and abs(model.eq_losses[-2]-model.eq_loss) < epsilon2:
            end = time.time()
            return epoch+1, (end-start)/epoch
    end = time.time()
    return epochs, (end - start)/epochs


# Main Process
def main():
    global ALGORITHM, D, SEED, EPOCHS, EPSILON1, EPSILON2, LEARNING_RATE, OMEGA, GAMMA, NET_STRUCTURE, POINTS_PER_DIM, OPTIMIZER
    config = load_config()
    ALGORITHM = config["algorithm"]
    D = config["dimension"]
    SEED = config["seed"]
    EPOCHS = config["epochs"]
    EPSILON1 = config["epsilon1"]
    EPSILON2 = EPSILON1 * config["epsilon2_ratio"]
    LEARNING_RATE = config["learning_rate"]
    OMEGA = config["omega"]
    GAMMA = config["gamma"]
    NET_STRUCTURE = config["network_structure"]
    POINTS_PER_DIM = config["points_per_dimension"]
    OPTIMIZER = config["optimizer"]
    seed_everything(SEED)

    os.makedirs("models", exist_ok=True)
    model = PINN(D, ALGORITHM)
    real_epoch, avg_time = train(model, epochs=EPOCHS, epsilon1=EPSILON1, epsilon2=EPSILON2)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'models/{ALGORITHM}_LP_{D}D_OMEGA{model.unique_parameter["omega"]}_SEED{SEED}_Adam_{time_stamp}.pkl', 'wb') as fi:
        pickle.dump(model, fi)
    print_config(model, SEED, EPOCHS, real_epoch, EPSILON1, EPSILON2, avg_time, time_stamp)


if __name__ == '__main__':
    main()
