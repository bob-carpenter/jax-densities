import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, expon

from ..BayesianModel import BayesianModel
from ..readers import read_real, read_real_lb

class UniLinearRegression(BayesianModel):
    @classmethod
    def constrain_parameters(cls, data, params_unc):
        pos = 0
        alpha, pos, _ = read_real(params_unc, pos)
        beta, pos, _ = read_real(params_unc, pos)
        sigma, pos, log_jac = read_real_lb(0.0, params_unc, pos)
        params = {'alpha': alpha, 'beta': beta, 'sigma': sigma}
        return params, log_jac

    @classmethod
    def unconstrain_parameters(cls, data, params):
        alpha = params['alpha']
        beta = params['beta']
        sigma_unc = jnp.log(params['sigma'])
        params_unconstrained = jnp.array([alpha, beta, sigma_unc])
        return params_unconstrained

    @classmethod
    def log_prior(cls, data, params):
        lp_alpha = norm.logpdf(params['alpha'], loc=0.0, scale=1.0)
        lp_beta = norm.logpdf(params['beta'], loc=0.0, scale=1.0)
        # "scale" is rate of exponential distribution (bad SciPy)
        lp_sigma = expon.logpdf(params['sigma'], scale=1.0)  
        return lp_alpha + lp_beta + lp_sigma

    @classmethod
    def log_likelihood(cls, data, params):
        x = jnp.asarray(data['x'])
        y = jnp.asarray(data['y'])
        alpha = params['alpha']
        beta = params['beta']
        sigma = params['sigma']
        mu = alpha + x * beta
        log_lik = jnp.sum(norm.logpdf(y, loc=mu, scale=sigma))
        return log_lik

    @classmethod
    def prior_sample(cls, rng, data):
        key_alpha, key_beta, key_sigma = jax.random.split(rng, 3)
        alpha = jax.random.normal(key_alpha, shape=()) * 1.0 + 0.0  # loc=0.0, scale=1.0
        beta = jax.random.normal(key_beta, shape=()) * 1.0 + 0.0
        sigma = jax.random.exponential(key_sigma, shape=()) * 1.0  # scale=1.0
        draw = {'alpha': alpha, 'beta': beta, 'sigma': sigma}
        return draw

    @classmethod
    def num_params_unc(cls, data):
        return 3


