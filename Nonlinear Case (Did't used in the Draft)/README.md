## Potential Extension to Nonlinear Problems

Although the main body of our paper focuses on linear differential operators, we believe that our method is capable of solving **nonlinear operator eigenvalue problems** as well. Due to the lack of a rigorous theoretical framework at this stage, we have not included these results in the formal submission. However, preliminary numerical experiments indicate promising behavior.

A classical example in this context is the **Gross–Pitaevskii equation**, which models quantum fluids and Bose–Einstein condensates (BECs) in low-temperature physics:

$$
  -u'' + u^3 + \left( -\frac{1}{c^2} \exp(2\cos x) + \sin^2 x - \cos x - 3 \right) u = \lambda u
$$

Here, $x \in [0, 2\pi]$ and periodic boundary conditions are assumed. The constant $c$ ensures normalization:

$$
c = \left[ \int_0^{2\pi} \exp(2 \cos x) \, dx \right]^{1/2} \approx 3.7846
$$

We experimented with a **spectral method** based on Fourier series and computed the minimum-modulus eigenvalues for different orders:

| N (Fourier Order) | Eigenvalue (λ) |
|-------------------|----------------|
| 8                 | 3.5323         |
| 16                | 2.1479         |
| 32                | 1.3957         |
| 64                | 1.2656         |

However, due to the presence of the **nonlinear cubic term** $u^3$, the spectral method struggles with frequency coupling and shows poor numerical stability as $N$ increases.

By contrast, when we apply our DRQI-based method (see `DRQI_FokkerPlank2D.py`) with appropriate modifications to the operator and potential, the method stably converges to an eigenvalue close to **1.4**. The MSR (Mean Squared Residual) of the PDE loss function is observed to converge across multiple relaxation settings.

> You can reproduce the spectral method results by changing the parameter `N` in `spec1.py`. The corresponding sampled eigenfunction values will be saved to files named like `f'GP{N}.csv'`.

---

### Informal Commentary

We didn’t formally include nonlinear problems in the paper, mainly because we’re still figuring out the theory. But honestly, the method **does seem to work** even when the operator includes nonlinear terms like $u^3$ We took a Gross–Pitaevskii-type equation and tried both spectral and PINN-based methods. Spectral methods get unstable quickly as the resolution increases (due to frequency interactions), but our DRQI framework **stably locked onto an eigenvalue around 1.4**, and the MSR loss even converged nicely.

So, while it's not fully ready for publication, it’s definitely something worth exploring if you’re interested in nonlinear eigenvalue problems.
