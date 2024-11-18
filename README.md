# JAX Densities

This repository contains a set of differentiable target probability densities coded in JAX, along with any necessary data, and in some cases, a sequence of (roughly) independent reference draws.

## Package `densjax`

This repository defines a Python package `densjax`.

## Basic directory structure

#### Utilities

General utilities are in the top-level directory, e.g., `densjax/readers.py`.


#### Target densities

Each target density has its own subdirectory, e.g., `densjax/uni_linear_regression` for the univariate linear regression example.

Each target density directory has a top level density definition in `density.py`, e.g., `densjac/uni_linear_regression/density.py`.  Then there are subdirectories containing pairs of data required for the model and roughly independent reference draws.
