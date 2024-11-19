class Model:
    @classmethod
    def num_unconstrained_parameters(cls, data):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def constrain_parameters(cls, data, params_unc):
        return {'params': params_unc}

    @classmethod
    def unconstrain_parameters(cls, data, params):
        return params['params']

    @classmethod
    def log_density_unconstrained(cls, data, params_unconstrained):
        params, log_jacobian = cls.constrain_parameters(data, params_unconstrained)
        log_density_params = cls.log_density(data, params)
        return log_jacobian + log_density_params

    @classmethod
    def log_density(cls, data, params):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def generated_quantities(cls, data):
        return { }

    @classmethod 
    def initial_draw(cls, data, rng):
        D = cls.num_unconstrained_parameters(data)
        params_unconstrained = jax.random.normal(rng, shape=D)
        return params_unconstrained


