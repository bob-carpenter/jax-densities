class Model:
    def num_unconstrained_parameters(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def constrain_parameters(self, data, params_unc):
        return {'params': params_unc}

    def unconstrain_parameters(self, data, params):
        return params['params']

    def log_density_unconstrained(self, data, params_unconstrained):
        params, log_jacobian = self.constrain_parameters(data, params_unconstrained)
        log_density_params = self.log_density(data, params)
        return log_jacobian + log_density_params

    def log_density(self, data, params):
        raise NotImplementedError("Subclasses should implement this method.")

    def generated_quantities(self, data):
        return { }

    def initial_draw(self, data, rng):
        D = self.num_unconstrained_parameters(data)
        params_unconstrained = jax.random.normal(rng, shape=D)
        return params_unconstrained


class BayesianModel(Model):
    def log_density(self, data, params):
        log_prior = self.log_prior(data, params)
        log_likelihood = self.log_likelihood(data, params)
        log_posterior = log_prior + log_likelihood
        return log_posterior

    def log_prior(self, data, params):
        return 0.0

    def log_likelihood(self, data, params):
        raise NotImplementedError("Subclasses should implement this method.")

