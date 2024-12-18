import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, expon

from ..base import BayesianModel
from ...utils.readers import read_real, read_real_lb

class UniLinearRegression(BayesianModel):
    def num_unconstrained_parameters(self, data):
        return 3

    def constrain_parameters(self, data, params_unc):
        pos = 0
        alpha, pos, _ = read_real(params_unc, pos)
        beta, pos, _ = read_real(params_unc, pos)
        sigma, pos, log_jac = read_real_lb(0.0, params_unc, pos)
        params = {'alpha': alpha, 'beta': beta, 'sigma': sigma}
        return params, log_jac

    def unconstrain_parameters(self, data, params):
        alpha = params['alpha']
        beta = params['beta']
        sigma_unc = jnp.log(params['sigma'])
        params_unconstrained = jnp.array([alpha, beta, sigma_unc])
        return params_unconstrained

    def log_prior(self, data, params):
        lp_alpha = norm.logpdf(params['alpha'], loc=0.0, scale=1.0)
        lp_beta = norm.logpdf(params['beta'], loc=0.0, scale=1.0)
        # "scale" is rate of exponential distribution (bad SciPy)
        lp_sigma = expon.logpdf(params['sigma'], scale=1.0)  
        return lp_alpha + lp_beta + lp_sigma

    def log_likelihood(self, data, params):
        x = jnp.asarray(data['x'])
        y = jnp.asarray(data['y'])
        alpha = params['alpha']
        beta = params['beta']
        sigma = params['sigma']
        mu = alpha + x * beta
        log_lik = jnp.sum(norm.logpdf(y, loc=mu, scale=sigma))
        return log_lik

    def generated_quantities(self, data, params, rng):
        x_new = jnp.asarray(data['x_new'])
        alpha = params['alpha']
        beta = params['beta']
        sigma = params['sigma']
        mu_new = alpha + beta * x_new
        y_new = mu_new + sigma * jax.random.normal(rng, shape=x_new.shape)
        return {'y_new': y_new}
        
    def initial_draw(self, rng, data):
        key_alpha, key_beta, key_sigma = jax.random.split(rng, 3)
        # normal(0, 1)
        alpha = jax.random.normal(key_alpha, shape=())
        beta = jax.random.normal(key_beta, shape=())
        # exponential(1)
        sigma = jax.random.exponential(key_sigma, shape=())
        draw = {'alpha': alpha, 'beta': beta, 'sigma': sigma}
        return draw


