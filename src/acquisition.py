import math

import torch
from botorch.utils.constants import get_constants_like
from botorch.utils.probability.utils import log_ndtr as log_Phi
from botorch.utils.probability.utils import log_phi
from botorch.utils.probability.utils import ndtr as Phi
from botorch.utils.probability.utils import phi
from botorch.utils.safe_math import log1mexp
from torch import Tensor


class AnalyticAcquisitionFucntion:
    def __init__(self, model, maximize: bool = True):
        self.model = model
        self.maximize = maximize

        self.device = model.X.device
        self.dtype = model.X.dtype

    def __call__(self, X):
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        mean, std = self._compute_mean_std(X)
        return self._compute_acquisition(mean, std)

    def _compute_mean_std(self, X):
        mean, std = self.model.predict(X, return_std=True)
        return mean, std

    def _compute_acquisition(self, mean, std):
        raise NotImplementedError


class ProbabilityOfImprovement(AnalyticAcquisitionFucntion):
    def __init__(self, model, best_f: float | Tensor, maximize: bool = True):
        """
        Single-outcome Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, maximize=maximize)
        self.best_f = torch.as_tensor(best_f)

    def _compute_acquisition(self, mean: Tensor, std: Tensor) -> Tensor:
        """
        Compute the Probability of Improvement.

        Args:
            mean: The predictive mean.
            std: The predictive standard deviation.

        Returns:
            The Probability of Improvement.
        """
        u = _scaled_improvement(mean, std, self.best_f, self.maximize)
        return Phi(u)


class LogProbabilityOfImprovement(AnalyticAcquisitionFucntion):
    def __init__(self, model, best_f: float | Tensor, maximize: bool = True):
        """
        Single-outcome Log Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, maximize=maximize)
        self.best_f = torch.as_tensor(best_f)

    def _compute_acquisition(self, mean: Tensor, std: Tensor) -> Tensor:
        """
        Compute the Log Probability of Improvement.

        Args:
            mean: The predictive mean.
            std: The predictive standard deviation.

        Returns:
            The Log Probability of Improvement.
        """
        u = _scaled_improvement(mean, std, self.best_f, self.maximize)
        return log_Phi(u)


class ExpectedImprovement(AnalyticAcquisitionFucntion):
    def __init__(self, model, best_f: float | Tensor, maximize: bool = True):
        """
        Single-outcome Expected Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, maximize=maximize)
        self.best_f = torch.as_tensor(best_f)

    def _compute_acquisition(self, mean: Tensor, std: Tensor) -> Tensor:
        """
        Compute the Expected Improvement.

        Args:
            mean: The predictive mean.
            std: The predictive standard deviation.

        Returns:
            The Expected Improvement.
        """
        u = _scaled_improvement(mean, std, self.best_f, self.maximize)
        return std * _ei_helper(u)


class LogExpectedImprovement(AnalyticAcquisitionFucntion):
    def __init__(self, model, best_f: float | Tensor, maximize: bool = True):
        """
        Single-outcome Log Expected Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, maximize=maximize)
        self.best_f = torch.as_tensor(best_f)

    def _compute_acquisition(self, mean: Tensor, std: Tensor) -> Tensor:
        """
        Compute the Log Expected Improvement.

        Args:
            mean: The predictive mean.
            std: The predictive standard deviation.

        Returns:
            The Log Expected Improvement.
        """
        u = _scaled_improvement(mean, std, self.best_f, self.maximize)
        return _log_ei_helper(u) + std.log()


class UpperConfidenceBound(AnalyticAcquisitionFucntion):
    def __init__(self, model, beta: float | Tensor = 1.0, maximize: bool = True):
        """
        Single-outcome Upper Confidence Bound (UCB).

        Analytic upper confidence bound that comprises of the posterior mean plus an
        additional term: the posterior standard deviation weighted by a trade-off
        parameter, `beta`.

        Args:
            model: A fitted single-outcome model.
            beta: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the trade-off parameter between mean and covariance.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, maximize=maximize)
        self.beta = torch.as_tensor(beta)

    def _compute_acquisition(self, mean: Tensor, std: Tensor) -> Tensor:
        """
        Compute the Upper Confidence Bound.

        Args:
            mean: The predictive mean.
            std: The predictive standard deviation.

        Returns:
            The Upper Confidence Bound.
        """
        return (mean if self.maximize else -mean) + self.beta.sqrt() * std


# ------------------------ Helper Functions ------------------------ #
def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)


def _log_ei_helper(u: Tensor) -> Tensor:

    # the following two numbers are needed for _log_ei_helper
    _neg_inv_sqrt2 = -(2**-0.5)
    _log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2

    """Accurately computes log(phi(u) + u * Phi(u)) in a differentiable manner for u in
    [-10^100, 10^100] in double precision, and [-10^20, 10^20] in single precision.
    Beyond these intervals, a basic squaring of u can lead to floating point overflow.
    In contrast, the implementation in _ei_helper only yields usable gradients down to
    u ~ -10. As a consequence, _log_ei_helper improves the range of inputs for which a
    backward pass yields usable gradients by many orders of magnitude.
    """
    if not (u.dtype == torch.float32 or u.dtype == torch.float64):
        raise TypeError(
            f"LogExpectedImprovement only supports torch.float32 and torch.float64 "
            f"dtypes, but received {u.dtype = }."
        )
    # The function has two branching decisions. The first is u < bound, and in this
    # case, just taking the logarithm of the naive _ei_helper implementation works.
    bound = -1
    u_upper = u.masked_fill(u < bound, bound)  # mask u to avoid NaNs in gradients
    log_ei_upper = _ei_helper(u_upper).log()

    # When u <= bound, we need to be more careful and rearrange the EI formula as
    # log(phi(u)) + log(1 - exp(w)), where w = log(abs(u) * Phi(u) / phi(u)).
    # To this end, a second branch is necessary, depending on whether or not u is
    # smaller than approximately the negative inverse square root of the machine
    # precision. Below this point, numerical issues in computing log(1 - exp(w)) occur
    # as w approaches zero from below, even though the relative contribution to log_ei
    # vanishes in machine precision at that point.
    neg_inv_sqrt_eps = -1e6 if u.dtype == torch.float64 else -1e3

    # mask u for to avoid NaNs in gradients in first and second branch
    u_lower = u.masked_fill(u > bound, bound)
    u_eps = u_lower.masked_fill(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps)
    # compute the logarithm of abs(u) * Phi(u) / phi(u) for moderately large negative u
    w = _log_abs_u_Phi_div_phi(u_eps)

    # 1) Now, we use a special implementation of log(1 - exp(w)) for moderately
    # large negative numbers, and
    # 2) capture the leading order of log(1 - exp(w)) for very large negative numbers.
    # The second special case is technically only required for single precision numbers
    # but does "the right thing" regardless.
    log_ei_lower = log_phi(u) + (
        torch.where(
            u > neg_inv_sqrt_eps,
            log1mexp(w),
            # The contribution of the next term relative to log_phi vanishes when
            # w_lower << eps but captures the leading order of the log1mexp term.
            -2 * u_lower.abs().log(),
        )
    )
    return torch.where(u > bound, log_ei_upper, log_ei_lower)


def _log_abs_u_Phi_div_phi(u: Tensor) -> Tensor:

    # the following two numbers are needed for _log_ei_helper
    _neg_inv_sqrt2 = -(2**-0.5)
    _log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2

    """Computes log(abs(u) * Phi(u) / phi(u)), where phi and Phi are the normal pdf
    and cdf, respectively. The function is valid for u < 0.

    NOTE: In single precision arithmetic, the function becomes numerically unstable for
    u < -1e3. For this reason, a second branch in _log_ei_helper is necessary to handle
    this regime, where this function approaches -abs(u)^-2 asymptotically.

    The implementation is based on the following implementation of the logarithm of
    the scaled complementary error function (i.e. erfcx). Since we only require the
    positive branch for _log_ei_helper, _log_abs_u_Phi_div_phi does not have a branch,
    but is only valid for u < 0 (so that _neg_inv_sqrt2 * u > 0).

        def logerfcx(x: Tensor) -> Tensor:
            return torch.where(
                x < 0,
                torch.erfc(x.masked_fill(x > 0, 0)).log() + x**2,
                torch.special.erfcx(x.masked_fill(x < 0, 0)).log(),
        )

    Further, it is important for numerical accuracy to move u.abs() into the
    logarithm, rather than adding u.abs().log() to logerfcx. This is the reason
    for the rather complex name of this function: _log_abs_u_Phi_div_phi.
    """
    # get_constants_like allocates tensors with the appropriate dtype and device and
    # caches the result, which improves efficiency.
    a, b = get_constants_like(values=(_neg_inv_sqrt2, _log_sqrt_pi_div_2), ref=u)
    return torch.log(torch.special.erfcx(a * u) * u.abs()) + b
