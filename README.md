# JAX Densities

This repository contains a set of differentiable target probability densities coded in JAX, along with any necessary data, and in some cases, a sequence of (roughly) independent reference draws.

## Package `densjax`

This repository defines a Python package `densjax`.  To install the package in local editable mode (so that updates to the files are picked up without reinstall), install the package from the top-level of this repository (i.e., the directory in which this `README.md` file resides).

```
$ cd jax-densities
$ pip install -e .
```

If you don't want editable mode, drop the `-e`. You can make sure the install happens for the Python you want it to by executing `pip` from within Python as follows.

```
$ python3 -m pip install -e .
```

## Basic directory structure

#### Package helpers

In the top level directory of the repository:

* `pyproject.toml` tells Python's `pip` installer to use `setuptools` for the installation with a recent enough version to support the Python enhancement proposal [PEP 517](https://peps.python.org/pep-0517/).
* `setup.py` defines the package and version using `setuptools`. 

#### Utilities

General utilities are in the top-level package directory.

* `densjax/readers.py` is the code for deserializing and constraining parameters.
* `densjax/BaysianModel.py` is the base class for Bayesian models (using `@classmethod` inheritance)


#### Target density directory structure

Each target density has its own subdirectory, which contains at least one file

* `<density-dir>/density.py`: defines the model's log density up to an additive constant (using `@classmethod` inheritance).

In addition, there may be subdirectories that contain pairs of:

* `<density-dir>/<test-dir>/data.json`: data for the models created by [`json.dump`](https://docs.python.org/3/library/json.html#json.dump); can be read in using [`json.load`](https://docs.python.org/3/library/json.html#json.load).
* `<density-dir>/<test-dir>/reference.npz`: reference draws created by [`numpy.savez_compressed`](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html); can be loaded using [`numpy.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load).


## List of target densities

| Class              | Directory             | Description                     |
|:-------------------|:----------------------|:--------------------------------|
| `UniLinearRegression` | `uni_linear_regression` | one-dimensional linear regression |
