"""
# ----- Future Work -----
- Direct implementation of multi-tart optimization for hyperparameters on GaussianProcess class
"""


from typing import Dict

import torch
import torch.optim as optim
from linear_operator.utils.cholesky import psd_safe_cholesky

from ..covfunc import CovarianceFunction


class GaussianProcess:
    def __init__(self, config, optimize=False, mprior=0.0):
        """
        Gaussian Process regressor class using PyTorch.

        Parameters
        ----------
        covfunc : instance of CovFunc
            Covariance function implementing K(X, Xstar).
        optimize : bool
            Whether to perform hyperparameter optimization.
        mprior : float
            Prior mean of the Gaussian Process.
        """
        self.config = config
        self.covfunc = CovarianceFunction(config)
        self.optimize = optimize
        self.mprior = mprior

    def get_params(self) -> Dict[str, float]:
        """
        Returns the covariance function hyperparameters.

        Returns
        -------
        dict
            Dictionary of hyperparameters.
        """
        params = self.covfunc.get_params()
        # params = {k: v.item() for k, v in params.items()}
        return params

    def fit(self, X, y, **kwargs):
        """
        Fits the Gaussian Process model to the data.
        """
        # self.X = torch.tensor(X, dtype=torch.float32)
        # self.y = torch.tensor(y, dtype=torch.float32)
        self.X = X.to(self.config.device.device, self.config.device.dtype)
        self.y = y.to(self.config.device.device, self.config.device.dtype)
        self.n_samples = X.shape[0]
        if self.optimize:
            self._optimize_hyperparameters(**kwargs)

        # Compute posterior parameters
        self._compute_posterior()

    def _compute_posterior(self):
        """
        Computes the posterior distribution of the Gaussian Process.
        """
        K = self.covfunc.K(self.X, self.X)

        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(1)

        self.L = psd_safe_cholesky(K)
        self.alpha = torch.cholesky_solve(self.y - self.mprior, self.L)

        # Compute log marginal likelihood
        y_centered = self.y - self.mprior
        self.logp = -0.5 * torch.matmul(y_centered.t(), self.alpha).item()
        self.logp -= torch.sum(torch.log(torch.diag(self.L))).item()
        self.logp -= self.n_samples / 2 * torch.log(torch.tensor(2 * torch.pi)).item()

    def _optimize_hyperparameters(self, n_steps=1000, lr=0.01):
        """
        Optimizes the hyperparameters of the covariance function.

        Parameters
        ----------
        n_steps : int
            Number of optimization steps.
        lr : float
            Learning rate for the optimizer.
        """
        params = self.get_params().values()
        optimizer = optim.Adam(params, lr=lr)

        # Optimization loop
        for _ in range(n_steps):
            optimizer.zero_grad()
            self._compute_posterior()
            loss = - self.logp
            loss.backward()
            optimizer.step()
            
            print(f"Iter: {_}, Log Marginal Likelihood: {self.logp.item()}")

    def predict(self, Xstar, var_min=1e-12, return_std=False):
        """
        Makes predictions using the Gaussian Process model.

        Parameters
        ----------
        Xstar : torch.Tensor, shape=(n_test_samples, n_features)
            Test inputs.
        var_min : float, default=1e-12
        return_std : bool
            If True, returns the standard deviation of the predictions.

        Returns
        -------
        fmean : torch.Tensor, shape=(n_test_samples,)
            Predictive mean at test inputs.
        fcov : torch.Tensor or None
            Predictive covariance matrix or standard deviations.
        """
        k_star = self.covfunc.K(self.X, Xstar)
        fmean = self.mprior + torch.matmul(k_star.t(), self.alpha)

        v = torch.cholesky_solve(k_star, self.L)
        K_star = self.covfunc.K(Xstar, Xstar)
        fcov = K_star - torch.matmul(k_star.t(), v)
        fcov = fcov.clamp_min(var_min)

        if return_std:
            fcov = torch.sqrt(torch.diag(fcov))

        return fmean, fcov

    def update(self, x_new, y_new):
        """
        Updates the model with new observations.

        Parameters
        ----------
        x_new : torch.Tensor, shape=(n_new_samples, n_features)
            New training inputs.
        y_new : torch.Tensor, shape=(n_new_samples,)
            New training targets.
        """
        self.X = torch.cat((self.X, x_new), dim=0)
        self.y = torch.cat((self.y, y_new), dim=0)
        self.n_samples = self.X.shape[0]
        if self.optimize:
            self._optimize_hyperparameters()
        else:
            self._compute_posterior()



###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
# Learning the model does not seem to work properly.
# Need to fix the code below.


# import pyro
# import pyro.distributions as dist
# from pyro.infer import MCMC, NUTS


# class GaussianProcessMCMC:
#     def __init__(self, covfunc, niter=2000, num_chains=1, warmup_steps=1000, mprior=0.0):
#         """
#         Gaussian Process class using MCMC sampling of covariance function hyperparameters.

#         Parameters
#         ----------
#         covfunc: instance of covariance function class
#             Covariance function to use. Currently supports `squaredExponential`, `matern32`, and `matern52`.
#         niter: int
#             Number of MCMC samples to draw.
#         num_chains: int
#             Number of MCMC chains to run in parallel.
#         warmup_steps: int
#             Number of warm-up (burn-in) steps before sampling.
#         mprior: float
#             Prior mean of the Gaussian Process.
#         """
#         self.covfunc = covfunc
#         self.niter = niter
#         self.num_chains = num_chains
#         self.warmup_steps = warmup_steps
#         self.mprior = mprior

#     def model(self, X, y):
#         """
#         Probabilistic model for the Gaussian Process with hyperparameter priors.
#         """
#         # Define priors for hyperparameters
#         lengthscale = pyro.sample('lengthscale', dist.Uniform(0.1, 10.0).expand([1]).to_event(1))
#         outputscale = pyro.sample('outputscale', dist.LogNormal(0.0, 1.0))
#         noise = pyro.sample('noise', dist.LogNormal(0.0, 1.0))

#         # Update covariance function hyperparameters
#         self.covfunc.kernel.base_kernel.lengthscale = lengthscale
#         self.covfunc.kernel.outputscale = outputscale
#         noise_matrix = noise * torch.eye(X.shape[0], device=X.device)

#         # Compute covariance matrix
#         K = self.covfunc.K(X, X) + noise_matrix

#         # Sample from multivariate normal
#         pyro.sample('y_obs', dist.MultivariateNormal(torch.zeros(X.shape[0], device=X.device), covariance_matrix=K), obs=y)

#     def fit(self, X, y):
#         """
#         Fits the Gaussian Process regressor using MCMC sampling of hyperparameters.

#         Parameters
#         ----------
#         X: torch.Tensor, shape=(n_samples, n_features)
#             Training instances.
#         y: torch.Tensor, shape=(n_samples,)
#             Target values.
#         """
#         self.X = X
#         self.y = y

#         # Use NUTS sampler
#         nuts_kernel = NUTS(self.model)
#         mcmc = MCMC(nuts_kernel, num_samples=self.niter, warmup_steps=self.warmup_steps, num_chains=self.num_chains)
#         mcmc.run(X, y)
#         self.trace = mcmc.get_samples()

#     def predict(self, Xstar, return_std=False, nsamples=10):
#         """
#         Predicts using the posterior samples of the Gaussian Process.

#         Parameters
#         ----------
#         Xstar: torch.Tensor, shape=(n_test_samples, n_features)
#             Test instances.
#         return_std: bool
#             If True, returns the standard deviation of the predictions.
#         nsamples: int
#             Number of posterior samples to use for prediction.

#         Returns
#         -------
#         torch.Tensor
#             Predictive means.
#         torch.Tensor
#             Predictive variances or standard deviations.
#         """
#         # Draw posterior samples of hyperparameters
#         samples = {key: self.trace[key][-nsamples:] for key in ['lengthscale', 'outputscale', 'noise']}
#         post_mean, post_var = [], []

#         for i in range(nsamples):
#             # Set sampled hyperparameters
#             self.covfunc.kernel.base_kernel.lengthscale = samples['lengthscale'][i]
#             self.covfunc.kernel.outputscale = samples['outputscale'][i]
#             noise = samples['noise'][i]

#             # Compute predictive mean and covariance
#             K_train = self.covfunc.K(self.X, self.X) + noise * torch.eye(self.X.size(0), device=self.X.device)
#             K_train_test = self.covfunc.K(self.X, Xstar)
#             K_test = self.covfunc.K(Xstar, Xstar)

#             L = torch.linalg.cholesky(K_train)
#             alpha = torch.cholesky_solve((self.y - self.mprior).unsqueeze(1), L).squeeze()

#             fmean = torch.matmul(K_train_test.t(), alpha) + self.mprior
#             v = torch.cholesky_solve(K_train_test, L)
#             fcov = K_test - torch.matmul(K_train_test.t(), v)

#             post_mean.append(fmean)
#             post_var.append(fcov if not return_std else torch.sqrt(torch.diag(fcov)))

#         mean = torch.stack(post_mean).mean(dim=0)
#         if return_std:
#             std = torch.stack(post_var).mean(dim=0)
#             return mean, std
#         else:
#             cov = torch.stack(post_var).mean(dim=0)
#             return mean, cov

#     def update(self, xnew, ynew):
#         """
#         Updates the model with new observations.

#         Parameters
#         ----------
#         xnew: torch.Tensor, shape=(n_new_samples, n_features)
#             New training instances.
#         ynew: torch.Tensor, shape=(n_new_samples,)
#             New target values.
#         """
#         self.X = torch.cat((self.X, xnew), dim=0)
#         self.y = torch.cat((self.y, ynew), dim=0)
#         self.fit(self.X, self.y)