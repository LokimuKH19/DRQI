# DRQI-Based Eigenvalue Solver

This repository provides a reproducible implementation of the DRQI-based eigenvalue solver used in our submission.

## Features
- Fully reproducible with fixed seed and logged training parameters.
- Supports multiple algorithms: DRQI, IPMNN, DRM.
- Logs training loss, eigenvalue estimates, and equation loss.
- Model and logs are saved for reproducibility.
- To ensure the reproductivity, follow the versions of all denpendencies in requirments.txt.

## How to Use?
- Main Process is in para_search.py. Select the type of problem in this file (as well as the parameters you want to change)
- Change the basic config in CONFIG_DIRICHLET.json and CONFIG_PERIOD.json
- Run para_search/seed_search.py to generate trained models and relative logs
- Detialed program is located in DRQI_Laplace2d.py
- Log files can be analyzed through view.py. ATTENTION: Always set the theoritical lambda manually when start the UI

## Exceptions
- In random initialized cases sometimes the model might convergence to high ordered eigenpairs
- We will continue to update the DRQI to improve its performance

## 2025-06-19
- After several series of experiments we found: 
- DRQI under different seeds, Adam optimizer: Suitable for larger lr (0.01). In contrast, IPMNN and DRM are advised to use 0.001.
- We would carry out more experiments tommorrow.
