class BayesianModel:
    @classmethod
    def log_density(cls, data, params_unconstrained):
        params, log_jacobian = cls.constrain_parameters(data, params_unconstrained)
        log_prior = cls.log_prior(data, params)
        log_likelihood = cls.log_likelihood(data, params)
        log_posterior = log_jacobian + log_prior + log_likelihood
        return log_posterior

    @classmethod
    def constrain_parameters(cls, data, params_unc):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def log_prior(cls, data, params):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def log_likelihood(cls, data, params):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def unconstrain_parameters(cls, data, params):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def prior_sample(cls, rng, data):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def num_params_unc(cls, data):
        raise NotImplementedError("Subclasses should implement this method.")

