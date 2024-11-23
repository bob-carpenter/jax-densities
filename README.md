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

General utilities are in the `utils` directory.

* `densjax/utils/readers.py` is the code for deserializing and constraining parameters.

#### Target density directory structure

The models are in the `models` directory.  There is a top-level utility file with base classes for models.

* `densjax/models/base.py`

Each target density has its own subdirectory `<model-dir>` with a file defining the density.

* `densjax/models/<model-dir>/density.py`: defines the model's log density up to an additive constant.

In addition, there may be subdirectories `<test-dir>` that contain pairs of:

* `densjax/models/<model-dir>/<test-dir>/data.json`: data for the models created by [`json.dump`](https://docs.python.org/3/library/json.html#json.dump); can be read in using [`json.load`](https://docs.python.org/3/library/json.html#json.load).
* `densjax/models/<model-dir>/<test-dir>/reference.npz`: reference draws created by [`numpy.savez_compressed`](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html); can be loaded using [`numpy.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load).


## Model base class: `Model.py`

Each model is coded as a class in Python that extends the base class `Model.py`.

#### Dimensionality

Each model must implement a method to return the (fixed) number of dimensions of the arguments to the log density function.  The argument `data` is the a dictionary of data variables.  There is no default implementation.

```python
def num_unconstrained_parameters(self, data) 
```

#### Variable transforms 

Each model must provide an unconstraining transform from a dictionary of parameters to a vector of unconstrained parameters and a matching constraining inverse transfrom from a vector of unconstrained parameters to a dictionary of parameters. The constraining transform also returns a log Jacobian for the change-of-variables adjustment involved in the inverse transform.

By default, the constraining transform maps a vector `v` of unconstrained parameters to the dictionary `{'params': v}` with a zero log Jacobian.  By default, the unconstraining transform maps a dictionary `{'params': v}` to the vector `v`.  Subclasses should override this for meaningful variable transforms.  These must both be defined.

```python 
def constrain_parameters(self, data, params_unc) 
```

```python
def unconstrain_parameters(self, data, params) 
```

#### Unnormalized log density---constrained and unconstrained

Each model must implement a method for a differentiable log density where `params_unconstrained` will be an array argument of size given by `num_unconstrained_parameters(self, data)` and `data` will be a dictionary of data variables.

```python
def log_density_unconstrained(self, data, params_unconstrained) 
```

By default, this method applies the constraining transform to the unconstrained parameters and returns the sum of the resulting log Jacobian and the return of the log density function, where the argument is a dictionary of data and a dictionary of constrained parameters.  There is no default implementation.

```python
def log_density(self, data, params)
```


#### Initializer

Each model must provide a method to initialize that returns a vector of the size given by `num_unconstrained_parameters`. The `Model.py` class provides a default implementation that returns a standard normal draw, which may be overridden by a subclass.

```python 
def initialize(self, data) 
```

#### Generated quantities

Each model must provide a method to define further quantities generated from parameters and data using a random number generator and return them in the form o a dictionary.  The default is to return an empty dictionary.

Models can provide additional quantities that are (random) functions of the parameters and data.  These will often be used for posterior predictive quantities of interest in Bayesian models in a similar manner to the Stan block `generated quantities`.

```python
def generated_quantities(self, data, params, rng) 
```


## Bayesian base model

A base class `BayesianModel` is provided in the package which can be used for Bayesian models.  It implements the `log_density` method over constrained parameters and data as the sum of a prior and a likelihood, given the following signatures, where `data` is a dictionary of data and `params` a dictionary of constrained parameters.  The log Jacobian is handled by the superclass's `log_density` method.

```python
def log_prior(self, data, params)

def log_likelihood(self, data, params)
```

By default, the `log_prior` method returns `0.0`.  There is no defualt implementation of the `log_likelihood` method.


## List of target densities

| Class              | Directory             | Description                     |
|:-------------------|:----------------------|:--------------------------------|
| `UniLinearRegression` | `uni_linear_regression` | one-dimensional linear regression |
