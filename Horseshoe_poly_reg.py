import numpy as np
import numpy.random as npr
import pymc3 as pm
import theano.tensor as tt
import pandas as pd

class Horseshoe_poly_reg():

    def __init__(self):
        self.dim = None
        self.poly = None
        self.coefs = None
        self.X = None
        self.y = None
        self.X_ob = None
        self.y_ob = None


    def create_poly_reg_problem(self, dim=12, sparse=True, n=40, sigma=1, random_state=42):
        """
        Create a polynomial regression problem (with sparse or non-sparce coefficients).
        """
        rng = np.random.default_rng(random_state)
        
        self.dim = dim
        self.n = n
        self.sigma = sigma
        
        coefs = rng.integers(5, 10) # Coefficients chosen between 5 and 10 in abs value
                                         # to make sure it is not sparse at this stage
        signs = rng.choice((-1, 1), size = self.dim)
        coefs = coefs * signs
        
        if sparse:
            nb_of_zero_coefs = np.ceil(0.75 * self.dim).astype(int) # Around 75a % of coefs set to 0
            zero_indices = rng.integers(0, self.dim, size=nb_of_zero_coefs)
            coefs[zero_indices] = 0
        
        self.coefs = coefs
        self.poly = lambda x : np.sum(np.array([self.coefs[i] * x ** i for i in range(len(coefs))]), axis=0)
        self.X = np.linspace(0, 1, self.n)
        self.y = self.poly(self.X)
        self.X_ob = rng.uniform(0, 1, size=self.n)
        self.y_ob = rng.normal(self.poly(self.X_ob), self.sigma)
        
        return self.X_ob, self.y_ob, self.X, self.y
        
        
    
    def get_mcmc_sample_lin_reg(self, X, y, sigma=None, tau=None, samples=4000):
        """
        This should return a pymc3 Trace object
        Solves a linear regression problem via the Horseshoe prior.
        """
        
        regression = pm.Model()
        
        with regression:
            dim = X.shape[1]
            if sigma is None:
                sigma = pm.HalfFlat("sigma")
            
            # Defining the horseshoe prior
            lmbda_vect = pm.HalfCauchy("lambda", 1, shape=dim)
            if tau is None:
                tau = pm.HalfCauchy("tau", 1)
            beta_vect = pm.Normal("Beta", mu=0, sigma=tt.dot(tau, lmbda_vect), shape=dim)
            
            # Defining the likelihood
            likelihood = pm.Normal('likelihood', mu=tt.dot(X, beta_vect), sigma=sigma, observed = y)
            
            
            trace = pm.sample(samples, target_accept=.9)
            
        return trace


if __name__ == '__main__':
    print("Hello Mr. Bardenet!")