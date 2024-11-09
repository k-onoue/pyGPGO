from typing import Dict

import torch
import torch.optim as optim

from ..covfunc import CovarianceFunction


class tStudentProcess:
    def __init__(self, config, optimize=False, mprior=0.0):
        """
        t-Student Process regressor class using PyTorch.

        Parameters
        ----------
        covfunc : instance of covariance function class
            Covariance function implementing K(X, Xstar).
        nu : float
            Degrees of freedom (>2.0).
        optimize : bool
            Whether to perform hyperparameter optimization.
        mprior : float
            Prior mean of the t-Student Process.
        """
        self.config = config
        self.covfunc = CovarianceFunction(self.config)

        if self.config.trained_params and "df" in self.config.trained_params:
            self.df = torch.nn.Parameter(torch.tensor(self.config.trained_params["df"], device=self.config.device.device, dtype=self.config.device.dtype))
        else:
            self.df = torch.nn.Parameter(torch.tensor(3.0, device=self.config.device.device, dtype=self.config.device.dtype))  # Default value

        self.optimize = optimize
        self.mprior = mprior

    def get_params(self) -> Dict[str, torch.Tensor]:
        """
        Returns the covariance function hyperparameters.

        Returns
        -------
        dict
            Dictionary of hyperparameters.
        """
        params = self.covfunc.get_params()
        params["df"] = self.df
        return params

    def fit(self, X, y):
        """
        Fits the t-Student Process model to the data.

        Parameters
        ----------
        X : torch.Tensor, shape=(n_samples, n_features)
            Training inputs.
        y : torch.Tensor, shape=(n_samples,)
            Training targets.
        """
        self.X = X.to(self.config.device.device, self.config.device.dtype)
        self.y = y.to(self.config.device.device, self.config.device.dtype)
        self.n_samples = X.shape[0]

        if self.optimize:
            self._optimize_hyperparameters()
        else:
            self._compute_posterior()

    def _compute_posterior(self):
        """
        Computes the posterior distribution of the t-Student Process.
        """
        K = self.covfunc.K(self.X, self.X)
        K += torch.eye(self.n_samples, device=self.X.device) * self.covfunc.noise ** 2  # Add noise term
        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve((self.y - self.mprior).unsqueeze(1), self.L).squeeze()

        # Compute log marginal likelihood
        self.logp = self._log_marginal_likelihood()

    def _log_marginal_likelihood(self):
        """
        Computes the log marginal likelihood of the t-Student Process.

        Returns
        -------
        float
            Log marginal likelihood.
        """
        n, df, y = self.n_samples, self.df, self.y - self.mprior
        
        df = torch.tensor(df, device=self.X.device, dtype=self.X.dtype)

        quad_form = torch.dot(y, self.alpha)

        logp = torch.lgamma((df + n) / 2) - torch.lgamma(df / 2)
        logp -= n / 2 * torch.log(df * torch.pi)
        logp -= torch.sum(torch.log(torch.diag(self.L)))
        logp -= (df + n) / 2 * torch.log(1 + quad_form / df)

        return logp

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
        for step in range(n_steps):
            optimizer.zero_grad()
            self._compute_posterior()
            loss = -self.logp
            loss.backward()
            optimizer.step()

            print(f"Iter: {step}, Log Marginal Likelihood: {self.logp.item()}")

    def predict(self, Xstar, return_std=False):
        """
        Makes predictions using the t-Student Process model.

        Parameters
        ----------
        Xstar : torch.Tensor, shape=(n_test_samples, n_features)
            Test inputs.
        return_std : bool
            If True, returns the standard deviation of the predictions.

        Returns
        -------
        fmean : torch.Tensor, shape=(n_test_samples,)
            Predictive mean at test inputs.
        fcov : torch.Tensor or None
            Predictive covariance matrix or standard deviations.
        """
        K_s = self.covfunc.K(self.X, Xstar)
        fmean = self.mprior + torch.matmul(K_s.t(), self.alpha)

        v = torch.cholesky_solve(K_s, self.L)
        K_ss = self.covfunc.K(Xstar, Xstar)
        fcov = ((self.df + self.alpha @ self.y - 2) / (self.df + self.n_samples - 2)) * (K_ss - K_s.t() @ v)

        if return_std:
            return fmean, torch.sqrt(torch.diag(fcov))
        else:
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



# class tStudentProcessMCMC:
#     def __init__(self, covfunc, nu=3.0, niter=2000, num_chains=1, warmup_steps=1000):
#         """
#         Student-t Process regressor class using MCMC sampling of hyperparameters.

#         Parameters
#         ----------
#         covfunc : instance of covariance function class
#             Covariance function implementing K(X, Xstar).
#         nu : float
#             Degrees of freedom (>2.0).
#         niter : int
#             Number of MCMC samples to draw.
#         num_chains : int
#             Number of MCMC chains to run in parallel.
#         warmup_steps : int
#             Number of warm-up (burn-in) steps before sampling.
#         """
#         self.covfunc = covfunc
#         self.nu = nu
#         self.niter = niter
#         self.num_chains = num_chains
#         self.warmup_steps = warmup_steps

#     def model(self, X, y):
#         """
#         Probabilistic model for the t-Student Process with hyperparameter priors.
#         """
#         # Define priors for hyperparameters
#         l = pyro.sample('l', dist.Uniform(0.1, 10.0))
#         sigmaf = pyro.sample('sigmaf', dist.LogNormal(0.0, 1.0))
#         sigman = pyro.sample('sigman', dist.LogNormal(0.0, 1.0))

#         # Update covariance function hyperparameters
#         self.covfunc.l = l
#         self.covfunc.sigmaf = sigmaf
#         self.covfunc.sigman = sigman

#         # Compute covariance matrix
#         K = self.covfunc.K(X, X)
#         K += torch.eye(X.shape[0]) * self.covfunc.sigman ** 2

#         # Student-t likelihood
#         y = y.squeeze()
#         zero_mean = torch.zeros_like(y)
#         scale_matrix = K * (self.nu - 2) / self.nu

#         pyro.sample('y_obs', dist.MultivariateStudentT(df=self.nu, loc=zero_mean, scale_tril=torch.linalg.cholesky(scale_matrix)), obs=y)

#     def fit(self, X, y):
#         """
#         Fits the t-Student Process regressor using MCMC sampling of hyperparameters.

#         Parameters
#         ----------
#         X : torch.Tensor, shape=(n_samples, n_features)
#             Training instances.
#         y : torch.Tensor, shape=(n_samples,)
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
#         Predicts using the posterior samples of the t-Student Process.

#         Parameters
#         ----------
#         Xstar : torch.Tensor, shape=(n_test_samples, n_features)
#             Test instances.
#         return_std : bool
#             If True, returns the standard deviation of the predictions.
#         nsamples : int
#             Number of posterior samples to use for prediction.

#         Returns
#         -------
#         fmean : torch.Tensor
#             Predictive means.
#         fcov : torch.Tensor
#             Predictive variances or standard deviations.
#         """
#         samples = {}
#         for key in ['l', 'sigmaf', 'sigman']:
#             samples[key] = self.trace[key][-nsamples:]

#         post_mean = []
#         post_var = []

#         for i in range(nsamples):
#             # Update covariance function with sampled hyperparameters
#             params = {key: samples[key][i].item() for key in samples}
#             covfunc = self.covfunc.__class__(**params)
#             tsp = tStudentProcess(covfunc, nu=self.nu)
#             tsp.fit(self.X, self.y)
#             m, v = tsp.predict(Xstar, return_std=return_std)
#             post_mean.append(m)
#             post_var.append(v)

#         mean = torch.stack(post_mean).mean(dim=0)
#         if return_std:
#             std = torch.stack(post_var).mean(dim=0)
#             return mean, std
#         else:
#             cov = torch.stack(post_var).mean(dim=0)
#             return mean, cov

#     def update(self, x_new, y_new):
#         """
#         Updates the model with new observations.

#         Parameters
#         ----------
#         x_new : torch.Tensor, shape=(n_new_samples, n_features)
#             New training instances.
#         y_new : torch.Tensor, shape=(n_new_samples,)
#             New target values.
#         """
#         self.X = torch.cat((self.X, x_new), dim=0)
#         self.y = torch.cat((self.y, y_new), dim=0)
#         self.fit(self.X, self.y)
