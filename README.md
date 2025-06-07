# DRQI-Based Eigenvalue Solver

This repository provides a reproducible implementation of the DRQI-based eigenvalue solver used in our submission.

## Features
- Fully reproducible with fixed seed and logged training parameters.
- Supports multiple algorithms: DRQI, IPMNN, DRM.
- Logs training loss, eigenvalue estimates, and equation loss.
- Model and logs are saved for reproducibility.

## Reproduction Instructions

```bash
pip install -r requirements.txt
python DRQI_Laplace2d.py
