
def read_real_array(theta_unc, pos, shape):
    num_elements = jnp.prod(jnp.array(shape))
    x = jax.lax.dynamic_slice(theta_unc, (pos,), (num_elements,))
    x = x.reshape(shape)
    next_pos = pos + num_elements
    return x, next_pos, 0.0

def read_real(theta_unc, pos, shape=None):
    return (theta_unc[pos], pos + 1, 0.0) if not shape else read_real_array(theta_unc, pos, shape)

def read_real_lb(lb, theta_unc, pos, shape=None):
    x, next_pos, _ = read_real(theta_unc, pos, shape)
    value = lb + jnp.exp(x)
    log_jac = jnp.sum(x)
    return value, next_pos, log_jac

def read_real_ub(ub, theta_unc, pos, shape=None):
    x, next_pos, _ = read_real(theta_unc, pos, shape)
    value = ub - jnp.exp(x)
    log_jac = jnp.sum(x)
    return value, next_pos, log_jac

def read_real_lub(lb, ub, theta_unc, pos, shape=None):
    x, next_pos, _ = read_real(theta_unc, pos, shape)
    y = jax.nn.sigmoid(x)
    value = lb + (ub - lb) * y
    log_jac = jnp.sum(jnp.log(ub - lb) + jax.nn.log_sigmoid(x) + jax.nn.log_sigmoid(-x))
