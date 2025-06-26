import numpy as np
from scipy.linalg import eigh
from matplotlib import pyplot as plt

# parameter set
N = 32  # fourier cut-off frequency, controls the final solution
L = 2 * np.pi  # period
x = np.linspace(0, L, 2 * N + 1, endpoint=False)  # spatial mesh
dx = x[1] - x[0]

# constant for normalization
c = 3.784581467

# Fourier's Base
k = np.fft.fftfreq(2 * N + 1, d=dx) * 2 * np.pi


# potential energy
def V(x):
    return - (1 / c**2) * np.exp(2 * np.cos(x)) + np.sin(x)**2 - np.cos(x) - 3


# Construct a Hamilton matrix
H = np.zeros((2 * N + 1, 2 * N + 1), dtype=complex)

# Kinetic energy
for n in range(2 * N + 1):
    H[n, n] += k[n]**2

# Update Potential Energy
V_x = V(x)
for n in range(2 * N + 1):
    for m in range(2 * N + 1):
        H[n, m] += V_x[m] * np.fft.ifft(np.fft.fft(np.exp(1j * k[n] * x)) * np.exp(1j * k[m] * x))[n]

# Nonlinear terms
psi_x = np.exp(np.cos(x)) / c
psi_x_cubed = psi_x**3
for n in range(2 * N + 1):
    for m in range(2 * N + 1):
        H[n, m] += psi_x_cubed[m] * np.fft.ifft(np.fft.fft(np.exp(1j * k[n] * x)) * np.exp(1j * k[m] * x))[n]

# Diagonalization the Hamilton
eigenvalues, eigenvectors = eigh(H)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# output the minimum non-trivial eigenvalue
lambda_min = eigenvalues[4]
psi_min = np.fft.ifft(eigenvectors[:, 4])

# Normalization
psi_min = np.real(psi_min / np.sqrt(np.sum(np.abs(psi_min)**2) * dx))
psi_min = (psi_min - np.min(psi_min))/(np.max(psi_min)-np.min(psi_min))


# save results
print("Smallest modulus eigenvalue:", lambda_min)
print("Corresponding eigenfunction:", psi_min)
np.savetxt(f'GP{N}.csv', psi_min, fmt='%f')

plt.plot(x, psi_min)
plt.show()
