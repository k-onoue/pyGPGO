from typing import Dict

import torch
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from torch.nn.functional import softplus

from .config import ModelConfig


def kronDelta(X, Xstar):
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if Xstar.dim() == 1:
        Xstar = Xstar.unsqueeze(1)
    
    return (X.unsqueeze(1) == Xstar.unsqueeze(0)).all(dim=2).float()


# ======================
# Kernel Creation Function
# ======================


def create_kernel(config: ModelConfig):
    """
    Creates the kernel for the GP or TP model based on the specified kernel type.
    """
    # Select base kernel based on kernel_type
    if config.kernel_type == "rbf":
        base_kernel = RBFKernel(
            ard_num_dims=config.ard_num_dims,
            lengthscale_prior=(
                config.priors.lengthscale_prior
                if not (
                    config.trained_params and "lengthscale" in config.trained_params
                )
                else None
            ),
            lengthscale_constraint=config.constraints.lengthscale_constraint,
        )
    elif config.kernel_type == "matern32":
        base_kernel = MaternKernel(
            nu=1.5,
            ard_num_dims=config.ard_num_dims,
            lengthscale_prior=(
                config.priors.lengthscale_prior
                if not (
                    config.trained_params and "lengthscale" in config.trained_params
                )
                else None
            ),
            lengthscale_constraint=config.constraints.lengthscale_constraint,
        )
    elif config.kernel_type == "matern52":
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=config.ard_num_dims,
            lengthscale_prior=(
                config.priors.lengthscale_prior
                if not (
                    config.trained_params and "lengthscale" in config.trained_params
                )
                else None
            ),
            lengthscale_constraint=config.constraints.lengthscale_constraint,
        )
    else:
        raise ValueError(
            "Invalid kernel type specified. Choose from 'rbf', 'matern32', or 'matern52'."
        )

    # Set trained lengthscale if available
    if config.trained_params and "lengthscale" in config.trained_params:
        lengthscale_value = torch.tensor(config.trained_params["lengthscale"]).to(
            device=config.device.device, dtype=config.device.dtype
        )
        # Adjust shape based on ard_num_dims
        if config.ard_num_dims is not None:
            # ARD case: lengthscale is of shape (1, ard_num_dims, 1)
            lengthscale_value = lengthscale_value.view(1, config.ard_num_dims, 1)
        else:
            # Non-ARD case: lengthscale is scalar
            lengthscale_value = lengthscale_value
        base_kernel.initialize(lengthscale=lengthscale_value)

    # Wrap the base kernel with ScaleKernel
    kernel = ScaleKernel(
        base_kernel=base_kernel,
        outputscale_prior=(
            config.priors.outputscale_prior
            if not (config.trained_params and "outputscale" in config.trained_params)
            else None
        ),
        outputscale_constraint=config.constraints.outputscale_constraint,
    )

    # Set trained outputscale if available
    if config.trained_params and "outputscale" in config.trained_params:
        outputscale_value = torch.tensor(config.trained_params["outputscale"]).to(
            device=config.device.device, dtype=config.device.dtype
        )
        kernel.initialize(outputscale=outputscale_value)

    return kernel


# ======================
# CovFunc Class with Noise Handling using kronDelta
# ======================


class CovarianceFunction:
    def __init__(self, config: ModelConfig):
        """
        Covariance function wrapper with noise handling for Gaussian Process kernels.
        
        Parameters
        ----------
        config : ModelConfig
            Configuration object that includes kernel type, priors, constraints, and noise settings.
        """
        self.config = config
        self.kernel = create_kernel(config)
        
        # Set noise level as a trainable parameter
        if config.trained_params and "noise" in config.trained_params:
            self.raw_noise = torch.nn.Parameter(
                torch.tensor(config.trained_params["noise"], device=config.device.device, dtype=config.device.dtype)
            )
        else:
            # Initialize with the mean of the noise prior if not specified in trained_params
            self.raw_noise = torch.nn.Parameter(
                torch.tensor(config.priors.noise_prior.mean, device=config.device.device, dtype=config.device.dtype)
            )

    @property
    def noise(self):
        """Apply softplus to ensure noise is positive."""
        return softplus(self.raw_noise)
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of trainable parameters for optimization.

        Returns
        -------
        list
            List of trainable parameters (raw lengthscale, outputscale, noise).
        """
        return {
            "lengthscale": self.kernel.base_kernel.raw_lengthscale,
            "outputscale": self.kernel.raw_outputscale,
            "noise": self.raw_noise
        }

    def K(self, X, Xstar):
        """
        Computes the covariance matrix including noise using kronDelta.

        Parameters
        ----------
        X : torch.Tensor, shape=(n_samples, n_features)
            Input data.
        Xstar : torch.Tensor, shape=(m_samples, n_features)
            Input data for prediction.

        Returns
        -------
        torch.Tensor
            Covariance matrix with noise component.
        """

        # Kernel matrix computation
        K = self.kernel(X, Xstar).evaluate()
        
        # Add observation noise
        K += self.noise * kronDelta(X, Xstar)
        
        return K


# if __name__ == "__main__":
#     # ======================
#     # Sample Usage
#     # ======================

#     # Config with Specific Kernel Type and Initial Values for Lengthscale, Outputscale, and Noise
#     config = ModelConfig(
#         ard_num_dims=None,  # Set to None for non-ARD (scalar lengthscale)
#         kernel_type="matern32",  # "rbf", "matern32", or "matern52"
#         trained_params={"lengthscale": 1.0, "outputscale": 1.0, "noise": 0.1},
#     )

#     cov_func = CovFunc(config)

#     # Example data
#     X = torch.tensor([[1.0], [2.0], [3.0]])
#     K = cov_func.K(X, X)  # X and Xstar are the same, so noise is added

#     # ======================
#     # Print Results
#     # ======================

#     print("Covariance matrix with noise:\n", K)
#     print(f"kernel: \n{cov_func.kernel}\n")
#     print(f"lengthscale: {cov_func.kernel.base_kernel.lengthscale.item()}")
#     print(f"outputscale: {cov_func.kernel.outputscale.item()}")
