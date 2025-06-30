# DRQI-Based Eigenvalue Solver

This repository provides a fully reproducible implementation of the DRQI-based eigenvalue solver used in our recent submission. This method is used to solve eigenvalue problem in practical engineering with a relatively large learning rate and realize a rapid convergence. In particular, the algorithm is used to find the non-trivial minimum modulus eigenvalue and its corresponding eigenfunction of the following problem with a relatively large learning rate and reach convergence with a certain stability.

# Problem Setup: Eigenvalue Formulation

This project addresses a general eigenvalue problem for a second-order differential operator $L$, defined over a connected domain  
$\Omega\subset\mathbb{R}^d$ with a Lipschitz continuous boundary $\partial\Omega$.

The goal is to find eigenpairs $u, \lambda$ such that:

$$
\begin{cases}
L u = \lambda u, & \text{for } x \in \Omega \\\\
B u = 0, & \text{for } x \in \partial\Omega
\end{cases}
\tag{1}
$$

Where:
- $L$ is a linear differential operator (typically involves second-order partial derivatives),
- $\lambda\in\mathbb{R}$ is the eigenvalue,
- $u(x)$ is the corresponding eigenfunction,
- $B$ encodes boundary conditions (e.g., Dirichlet or Neumann).

---

## Function Spaces

To ensure a valid solution, the eigenfunction $u(x)$ is assumed to lie in the Sobolev space:

$$
u(x) \in H^2(\Omega) \cap V
$$

Here, $V$ depends on the type of boundary condition $B$.  
For example, with homogeneous Dirichlet boundary conditions (i.e., $u = 0$ on $\partial\Omega$), we set:

$$
V = H_0^1(\Omega)
$$

In this context, the operator $L$ acts as a mapping:

$$
L : H^2(\Omega) \cap V \rightarrow L^2(\Omega)
$$

---

## Residuals and Error Measures

Throughout this project, all residuals and error metrics are computed using the standard $L^2$ norm unless otherwise specified. That is:

$$
\Vert f\Vert_{L^2(\Omega)} = \left( \int_\Omega |f(x)|^2 dx \right)^{1/2}
$$

This norm provides a measure of the mean-squared difference over the domain $\Omega$ and is commonly used in PDE-related numerical and learning-based methods. (I mean, PINN...)
In fact, we have to use discreted sampling points to approximate the norm.


## 🔍 Features
- Deterministic runs via fixed random seeds and recorded training parameters.
- Supports multiple algorithms: **DRQI**, **IPMNN**, and **DRM**.
- Logs include training loss, eigenvalue estimates, and equation residuals.
- Models and logs are saved for reproducibility.
- To ensure consistent results, please follow the exact dependency versions listed in `requirements.txt`.

## 🚀 How to Use

- The main entry points are `para_search.py` and `seed_search.py`.  
  Choose the target problem type and modify relevant parameters directly in these files.

- Basic configuration can be changed in `CONFIG_DIRICHLET.json` or `CONFIG_PERIOD.json`.

- Run `seed_search.py` to train models and generate logs.  
  *(Note: `para_search.py` is currently experimental.)*

- The core implementation is located in:
  - `DRQI_Laplace2d.py` for Dirichlet problems  
  - `DRQI_FokkerPlank2d.py` for periodic/Fokker-Planck problems  
  Modify the differential operators as needed — detailed annotations are provided.

- Use `view.py` to analyze logs **after training is complete**.  
  ⚠️ Always set the theoretical eigenvalue (`lambda`) manually before launching the UI.

---

## 📅 Development Log

### 2025-06-19
- DRQI with Adam optimizer performs best with a larger learning rate (0.01).
- IPMNN and DRM are more stable with a smaller learning rate (0.001).
- Further experiments are planned.

### 2025-06-22
- Additional Laplace test results completed.
- QMC integration improves DRM's high-dimensional accuracy only in the Laplace case.
- The Definition of MSEE still needs further refinement.

### 2025-06-26 — Major Update
- Support for Fokker-Planck problems added.
- New visualization tools: **Density plots** and **Omega plots**.

### 2025-06-30
- The DRQI was validated by neuron transportation equation, showing its potential for nuclear engineering usage.

---

## ⚠️ Known Issues & Notes

- Some log files contain incorrect "Lambda Error" values.  
  However, this is handled correctly in the UI through manual input of the theoretical solution.

- DO NOT use seeds longer than 8 digits or the search for timestamp will be bugged in `view.py`

- MSEE behavior was improved after June 23.  
  Still, **Type-B convergence** (discussed in the paper) occasionally appears in periodic cases — likely due to eigenfunction phase shifts.  
  We suggest using **Pearson correlation** as an alternative metric to monitor convergence.

- Pretraining (as mentioned in the manuscript) must currently be performed manually by modifying the loss function in the main program files.  
  *Note: current logs do not include pretraining runs.*

- Rarely, models without pretraining may converge to higher-order eigenpairs. This phenomenon is actually quite interesting and could be a direction for future work. The general idea would be to **intentionally guide the model to first converge toward an approximate eigenfunction**, possibly by using a simplified or auxiliary loss function. Once the model has "locked onto" that mode, we can then **switch back to the original loss function** and continue normal training.  In principle, this kind of staged training might allow us to selectively target higher-order eigenpairs. However, in practice, **we haven't yet found a stable or repeatable way** to make this work consistently. The results seem to depend on a mix of initialization, seed values, and architecture details — all of which need more systematic exploration.

- The figure-saving function has some issues with filename formatting.
