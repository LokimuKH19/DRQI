import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
import pickle
import random
plt.ion()
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)


# Function to keep the random seed
def seed_everything(seed=11):
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
        self.algorithm = algorithm  # Mode Selection
        self.d = d  # Dimensions
        self.M = 3  # Fourier Features
        self.net = create_model([2*self.d*self.M, 40, 40, 1])  # network structure
        self.length = self.d * 1000    # control points m
        self.device = 'cuda'  # If you don't have cuda devices use "cpu" insdead
        self.net.to(self.device)
        self.learning_rate = 0.001  # learning rate
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)  # Optimizer Selection
        self.mse = torch.nn.MSELoss(reduction='mean')  # MSE
        self.loss = None
        self.lambdas = []

        x = lhs(self.d, self.length)*2*np.pi
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

    def load_data(self, x, requires_grad=True):
        data64 = torch.tensor(x, requires_grad=requires_grad, dtype=torch.float64)
        return data64.to(self.device)

    def potential(self, x):
        if isinstance(x, list):
            X = torch.cat((x), 1)
        else:
            X = x
        res = torch.zeros_like(X)
        for i in range(self.d):
            res += 0.1 * torch.cos(x[i])
        return torch.sin(res)

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
        x1 = []
        for i in range(self.d):
            x1 += [torch.sin(j * x[i]) for j in range(self.M)] + [torch.cos(j * x[i]) for j in range(self.M)]
        if isinstance(x1, list):
            X = torch.cat((x1), 1)
        else:
            X = x1
        result = self.forward(X)
        return result

    def update(self):
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()
        x = self.x
        u = self.net_forward(x)
        v = self.potential(x)
        u_xx = None
        v_xx = None
        uxvx = None
        for i in range(self.d):
            xi = x[i]
            u_xi = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
            v_xi = torch.autograd.grad(v.sum(), xi, create_graph=True)[0]
            u_xixi = torch.autograd.grad(u_xi.sum(), xi, create_graph=True)[0]
            v_xixi = torch.autograd.grad(v_xi.sum(), xi, create_graph=True)[0]
            if u_xx is None:
                u_xx = u_xixi
            else:
                u_xx += u_xixi
            if v_xx is None:
                v_xx = v_xixi
            else:
                v_xx += v_xixi
            if uxvx is None:
                uxvx = u_xi * v_xi
            else:
                uxvx += u_xi * v_xi

        # Edit your operator here
        lu = - u_xx - uxvx - u*v_xx + 5*u
        # Mode selection
        if self.algorithm == "DRQI":
            omega = 0.72    # Relaxation Factor
            self.loss = self.mse(lu-(self.lambda_k1+1)*(omega*u+(1-omega)*self.u), torch.zeros_like(u).to(self.device))
        elif self.algorithm == "IPMNN":
            self.loss = self.mse(lu / torch.norm(lu) - self.u, torch.zeros_like(u).to(self.device))
        elif self.algorithm == "DRM":
            gamma = 0    # d>3 use gamma=100
            self.loss = torch.sum(lu * u)/torch.sum(u * u) + gamma*(torch.sum(u * u)-1)**2
        self.loss.backward()
        lambda_k = self.read_data(torch.sum(lu * u)/torch.sum(u * u))
        lambda_k = lambda_k.max()
        self.lambda_k1 = lambda_k
        self.lambdas.append(self.lambda_k1)
        normalized_u = u / torch.norm(u, p=2)
        self.u = self.load_data(self.read_data(normalized_u), requires_grad=False)
        self.optimizer.step()


def train(model, epochs):
    standard = 5
    for epoch in range(epochs):
        model.update()
        print(f'Dimension{model.d} Epoch {epoch + 1}/{epochs}, Loss: {model.loss.item()}, '
              f'Lambda_Error: {abs(model.lambda_k1-standard)}, '
              f'Eigen Value: {model.lambda_k1}')
        if abs(model.lambda_k1-standard) <= 1e-8 and epoch > 10000:
            break


if __name__ == "__main__":
    epochs = 20000   # maximum iteration K
    d = 1    # dimension
    model = PINN(d, "DRM")  # create model
    train(model, epochs=epochs)
    # save model
    with open(f'DRM_FP_1D_SEED11.pkl', 'wb') as fi:
        pickle.dump(model, fi)
