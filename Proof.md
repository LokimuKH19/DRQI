# Theoretical Analysis of DRQI
> To response the concerns of DRQI's theoretical issues, we conducted the proof of the algorithm as below.

## Notation & Setup (variables definition)

* Let $\Omega \subset \mathbb{R}^d$ be a bounded domain and $\mathcal H$ denote a Hilbert space (e.g. $L^2(\Omega)$ or $H^2(\Omega)$ depending on context). Inner product $\langle \cdot,\cdot\rangle$ and norm $| \cdot|$ denote the Hilbert-space inner product and induced norm (or Euclidean norm $|\cdot|_2$ for finite-dimensional vectors when we consider discrete/matrix case).

* Continuous operator: (\mathcal L: \mathcal D(\mathcal L)\subset\mathcal H \to \mathcal H). Assume (\mathcal L) is linear and (for main theorems) self-adjoint (Hermitian) and elliptic so that spectral theory applies; denote its eigenpairs by ((\lambda_j,u_j)) with (|u_j|=1) and eigenvalues ordered ( \lambda_1 < \lambda_2 \le \lambda_3 \le \dots ). We focus on the principal eigenpair ((\lambda^*,u^*):=(\lambda_1,u_1)) and assume it is simple.

* Discrete operator / matrix: for a given sampling size (m) (point evaluation collocation), denote the discrete operator / matrix by (\mathbf L_m \in \mathbb R^{m\times m}). Its eigenpairs are ((\lambda_{m,j}, \mathbf u_{m,j})). We use (|\cdot|) also for Euclidean norm on (\mathbb R^m).

* Rayleigh quotient for a vector/function (v) (nonzero): (R(v) = \dfrac{\langle \mathcal L v, v\rangle}{\langle v,v\rangle}) (continuous) or (R(\mathbf v)=\dfrac{\mathbf v^\top \mathbf L_m \mathbf v}{\mathbf v^\top\mathbf v}) (discrete).

* Classical (matrix) Rayleigh Quotient Iteration (RQI): given (\mathbf u_0) normalized, define
  [
  \mathbf u_k^{\rm ideal} = (\mathbf L_m - \lambda_{k-1} \mathbf I)^{-1} \mathbf u_{k-1},\quad
  \mathbf u_k^{\rm ideal} \leftarrow \frac{\mathbf u_k^{\rm ideal}}{|\mathbf u_k^{\rm ideal}|},\quad
  \lambda_k = R(\mathbf u_k^{\rm ideal}).
  ]
  (We will compress notation and call the exact mapping (F(\mathbf u) := \mathrm{Norm}\big((\mathbf L_m - R(\mathbf u)\mathbf I)^{-1}\mathbf u\big)).)

* DRQI step (algorithm used in the paper): instead of computing exact inverse, a neural network parameterization (u(x;\theta)) is trained to minimize a semi-implicit loss approximating the RQI step; the resulting discrete vector at sampled points is (\tilde{\mathbf u}_k). We model the DRQI update as an **inexact / perturbed RQI**:
  [
  |\tilde{\mathbf u}*k - F(\tilde{\mathbf u}*{k-1})| \le \varepsilon_k,
  ]
  where (\varepsilon_k) models the optimization / approximation error in step (k).

* Spectral gap on discrete level: assume (\mathbf L_m) has eigenvalues (\lambda_{m,1} < \lambda_{m,2} \le \cdots). Define gap (\gamma_{m} := \lambda_{m,2}-\lambda_{m,1} > 0). For continuous operator, let gap (\gamma := \lambda_2 - \lambda_1>0).

* All constants (C_i) below will be functions of (\mathbf L_m) or (\mathcal L) norms, the spectral gap, and/or norms of resolvents at (\lambda^*); we indicate dependence when relevant.
