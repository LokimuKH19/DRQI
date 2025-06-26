# DRQI-Based Eigenvalue Solver

This repository provides a reproducible implementation of the DRQI-based eigenvalue solver used in our submission.

## Features
- Fully reproducible with fixed seed and logged training parameters.
- Supports multiple algorithms: DRQI, IPMNN, DRM.
- Logs training loss, eigenvalue estimates, and equation loss.
- Model and logs are saved for reproducibility.
- To ensure the reproductivity, follow the versions of all denpendencies in requirments.txt.

## How to Use?
- Main Process is in para_search.py/seed_search.py. Select the type of problem in this file (as well as the parameters you want to change)
- Change the basic config in CONFIG_DIRICHLET.json and CONFIG_PERIOD.json
- Run seed_search.py to generate trained models and relative logs (While para_search.py is still an experimental function)
- Detialed program is located in DRQI_Laplace2d.py/DRQI_FokkerPlank2d.py. You can change the operators in Dirichlet and periodic boundary  respectivelly. Follow the annotations.
- Log files can be analyzed through view.py AFTER THE MODEL IS CREATED. ATTENTION: Always set the theoritical lambda manually when start the UI

## 2025-06-19
- After several series of experiments we found: 
- DRQI under different seeds, Adam optimizer: Suitable for larger lr (0.01). In contrast, IPMNN and DRM are advised to use 0.001.
- We would carry out more experiments tommorrow.

## 2025-06-22
- More Laplace results finished
- QMC is useful for imporve 2D results of DRM
- The MSEE needs to be improved

## 2025-6-26 Major Update
- Fokker-Plank Problems can be solved through these package now
- Density Plot supported
- Omega Plot supported

## Some known issues has not been completely solved 
- In some log files "The Lambda Error" is not correct. However, in the UI we managed to avoid this problem by manually update the theoretical solution.
- the MSEE was imporved after June 23. However, the type-B convergences in related paper are still observed especially in periodic problems. We belive that this issue is mainly caused by the phase shift of the eigenfunction. Maybe the Pearson correlation coefficient is more suitable for monitoring the convergence of the trail function during training.
- The pre-training mentioned in the original manuscript has to be done manually, by switching the loss in the 2 main programs. (The current log files are not involved)
- In extremely rare cases, sometimes the model without pre-training still might convergence to high ordered eigenpairs.
- The save figure function has problems on nomenclature of the figures.
- Some minor annotations were written in Chinese in the program files.
