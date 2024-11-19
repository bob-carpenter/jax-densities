import jax
import jax.numpy as jnp
from densjax import UniLinearRegression

data = {
    'x': jnp.array([1.0, 2.0, 3.0]),
    'y': jnp.array([2.1, 3.7, 6.5])
}

params_unc = jnp.array([0.5, 1.0, 0.1])

log_density_value = UniLinearRegression.log_density(data, params_unc)
print("Log Density Value (just value):", log_density_value)

log_density_v, log_density_g = jax.value_and_grad(UniLinearRegression.log_density, argnums=1)(data, params_unc)
print("Log Density Value (combo):", log_density_v)
print("Log Density Gradient (combo):", log_density_g)

log_density_and_grad_jit = jax.jit(jax.value_and_grad(UniLinearRegression.log_density, argnums=1))
log_density_value_jit, log_density_grad_jit = log_density_and_grad_jit(data, params_unc)
print("JIT Log Density Value:", log_density_value_jit)
print("JIT Log Density Gradient:", log_density_grad_jit)
