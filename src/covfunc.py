import torch
import torch.special as tspecial


default_bounds = {
    'l': [1e-4, 1],
    'sigmaf': [1e-4, 2],
    'sigman': [1e-6, 2],
    'v': [1e-3, 10],
    'gamma': [1e-3, 1.99],
    'alpha': [1e-3, 1e4],
    'period': [1e-3, 10]
}


def l2norm_(X, Xstar):
    """
    Computes the pairwise Euclidean distance between rows of X and Xstar.

    Parameters
    ----------
    X: torch.Tensor, shape=(n, nfeatures)
        Instances.
    Xstar: torch.Tensor, shape=(m, nfeatures)
        Instances.

    Returns
    -------
    torch.Tensor
        Pairwise Euclidean distances.
    """
    X = X.unsqueeze(1)  # Shape: (n, 1, nfeatures)
    Xstar = Xstar.unsqueeze(0)  # Shape: (1, m, nfeatures)
    return torch.sqrt(torch.sum((X - Xstar) ** 2, dim=2))


def kronDelta(X, Xstar):
    """
    Computes Kronecker delta for rows in X and Xstar.

    Parameters
    ----------
    X: torch.Tensor, shape=(n, nfeatures)
        Instances.
    Xstar: torch.Tensor, shape=(m, nfeatures)
        Instances.

    Returns
    -------
    torch.Tensor
        Kronecker delta between row pairs of `X` and `Xstar`.
    """
    return (X.unsqueeze(1) == Xstar.unsqueeze(0)).all(dim=2).float()


class squaredExponential:
    def __init__(self, l=1.0, sigmaf=1.0, sigman=1e-6, bounds=None, parameters=['l', 'sigmaf', 'sigman']):
        """
        Squared exponential kernel class using PyTorch.

        Parameters
        ----------
        l: float
            Characteristic length-scale.
        sigmaf: float
            Signal variance.
        sigman: float
            Noise variance.
        bounds: list
            Hyperparameter bounds for optimization.
        parameters: list
            List of hyperparameters to optimize.
        """
        self.l = l
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        self.bounds = bounds if bounds is not None else [default_bounds[param] for param in self.parameters]

    def K(self, X, Xstar):
        """
        Computes covariance matrix between X and Xstar.

        Parameters
        ----------
        X: torch.Tensor, shape=(n, nfeatures)
        Xstar: torch.Tensor, shape=(m, nfeatures)

        Returns
        -------
        torch.Tensor
            Covariance matrix.
        """
        r = l2norm_(X, Xstar)
        return self.sigmaf * torch.exp(-0.5 * (r / self.l) ** 2) + self.sigman * kronDelta(X, Xstar)

    def gradK(self, X, Xstar, param='l'):
        """
        Computes gradient of the covariance matrix with respect to hyperparameter `param`.

        Parameters
        ----------
        X: torch.Tensor, shape=(n, nfeatures)
        Xstar: torch.Tensor, shape=(m, nfeatures)
        param: str
            Hyperparameter to compute gradient with respect to.

        Returns
        -------
        torch.Tensor
            Gradient matrix.
        """
        r = l2norm_(X, Xstar)
        K = self.K(X, Xstar) - self.sigman * kronDelta(X, Xstar)  # Remove noise term
        if param == 'l':
            return K * (r ** 2) / (self.l ** 3)
        elif param == 'sigmaf':
            return K / self.sigmaf
        elif param == 'sigman':
            return kronDelta(X, Xstar)
        else:
            raise ValueError('Param not found')
        

class matern32:
    def __init__(self, l=1.0, sigmaf=1.0, sigman=1e-6, bounds=None, parameters=['l', 'sigmaf', 'sigman']):
        """
        Matern kernel with v=3/2, implemented using PyTorch.

        Parameters
        ----------
        l : float, default=1.0
            Characteristic length-scale of the kernel. Controls how quickly the covariance decays with distance.
        sigmaf : float, default=1.0
            Signal variance. Controls the overall scale of the covariance function.
        sigman : float, default=1e-6
            Noise variance. Represents the additive noise in the output space.
        bounds : list of tuples, optional
            List of tuples specifying the bounds for each hyperparameter to be used in optimization. 
            Defaults to `default_bounds` if not provided.
        parameters : list of str, default=['l', 'sigmaf', 'sigman']
            List of hyperparameters to be optimized. 

        Attributes
        ----------
        l : float
            Characteristic length-scale.
        sigmaf : float
            Signal variance.
        sigman : float
            Noise variance.
        parameters : list of str
            Hyperparameters to optimize.
        bounds : list of tuples
            Bounds for optimization.
        """
        self.l = l
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        self.bounds = bounds if bounds is not None else [default_bounds[param] for param in self.parameters]

    def K(self, X, Xstar):
        """
        Computes the covariance matrix using the Matern 3/2 kernel.

        Parameters
        ----------
        X : torch.Tensor, shape=(n_samples, n_features)
            Input data.
        Xstar : torch.Tensor, shape=(m_samples, n_features)
            Input data for prediction.

        Returns
        -------
        torch.Tensor
            Covariance matrix calculated using the Matern 3/2 kernel.
        """
        r = l2norm_(X, Xstar)
        sqrt3_r_l = torch.sqrt(torch.tensor(3.0)) * r / self.l
        K = self.sigmaf * (1 + sqrt3_r_l) * torch.exp(-sqrt3_r_l)
        K += self.sigman * kronDelta(X, Xstar)
        return K


class matern52:
    def __init__(self, l=1.0, sigmaf=1.0, sigman=1e-6, bounds=None, parameters=['l', 'sigmaf', 'sigman']):
        """
        Matern kernel with v=5/2, implemented using PyTorch.

        Parameters
        ----------
        l : float, default=1.0
            Characteristic length-scale of the kernel. Controls how quickly the covariance decays with distance.
        sigmaf : float, default=1.0
            Signal variance. Controls the overall scale of the covariance function.
        sigman : float, default=1e-6
            Noise variance. Represents the additive noise in the output space.
        bounds : list of tuples, optional
            List of tuples specifying the bounds for each hyperparameter to be used in optimization. 
            Defaults to `default_bounds` if not provided.
        parameters : list of str, default=['l', 'sigmaf', 'sigman']
            List of hyperparameters to be optimized. 

        Attributes
        ----------
        l : float
            Characteristic length-scale.
        sigmaf : float
            Signal variance.
        sigman : float
            Noise variance.
        parameters : list of str
            Hyperparameters to optimize.
        bounds : list of tuples
            Bounds for optimization.
        """
        self.l = l
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        self.bounds = bounds if bounds is not None else [default_bounds[param] for param in self.parameters]

    def K(self, X, Xstar):
        """
        Computes the covariance matrix using the Matern 5/2 kernel.

        Parameters
        ----------
        X : torch.Tensor, shape=(n_samples, n_features)
            Input data.
        Xstar : torch.Tensor, shape=(m_samples, n_features)
            Input data for prediction.

        Returns
        -------
        torch.Tensor
            Covariance matrix calculated using the Matern 5/2 kernel.
        """
        r = l2norm_(X, Xstar)
        sqrt5_r_l = torch.sqrt(torch.tensor(5.0)) * r / self.l
        r2_l2 = (r ** 2) / self.l ** 2
        K = self.sigmaf * (1 + sqrt5_r_l + (5 * r2_l2) / 3) * torch.exp(-sqrt5_r_l)
        K += self.sigman * kronDelta(X, Xstar)
        return K