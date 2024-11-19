from .Model import Model

class BayesianModel(Model):
    @classmethod
    def log_density(cls, data, params):
        log_prior = cls.log_prior(data, params)
        log_likelihood = cls.log_likelihood(data, params)
        log_posterior = log_prior + log_likelihood
        return log_posterior

    @classmethod
    def log_prior(cls, data, params):
        return 0.0

    @classmethod
    def log_likelihood(cls, data, params):
        raise NotImplementedError("Subclasses should implement this method.")

