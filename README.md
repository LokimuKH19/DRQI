# DRQI-Based Eigenvalue Solver

This repository provides a fully reproducible implementation of the DRQI-based eigenvalue solver used in our recent submission. This method is used to solve the eigenvalue problem in practical engineering with a relatively large learning rate and realize a rapid convergence.

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
- QMC integration improves 2D DRM accuracy.
- MSEE still needs further refinement.

### 2025-06-26 — Major Update
- Support for Fokker-Planck problems added.
- New visualization tools: **Density plots** and **Omega plots**.

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

- Rarely, models without pretraining may converge to higher-order eigenpairs.

- The figure-saving function has some issues with filename formatting.

- Some comments in code files are written in Chinese.
