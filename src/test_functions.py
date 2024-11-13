from typing import Literal

import numpy as np


def generate_noise(size, noise_type="normal", df=1):
    """Generates noise based on the specified type and size."""
    if noise_type == "normal":
        return np.random.normal(0, 1, size=size)
    elif noise_type == "t":
        return np.random.standard_t(df, size=size)
    elif noise_type == "uniform":
        return np.random.uniform(-1, 1, size=size)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")


def add_noise(func):
    """Decorator to add noise to the output of a function based on noise settings in self."""
    def wrapper(self, X):
        result = func(self, X)
        # Add noise only if noise_coeff is non-zero
        if self.noise_coeff != 0:
            noise = generate_noise(result.shape, self.noise_type, self.df)
            result += self.noise_coeff * noise
        return result
    return wrapper


class TestFunction:
    def __init__(
        self,
        direction: Literal["minimize", "maximize"] = "minimize",
        noise_coeff: float = 0.0,
        noise_type: Literal["normal", "t", "uniform"] = "normal",
        df: int = 1,
    ):
        self.search_space: np.ndarray = np.array([])  # Set the search space externally
        self.direction = direction
        self.noise_coeff = noise_coeff
        self.noise_type = noise_type
        self.df = df

    @add_noise
    def __call__(self, X):
        raise NotImplementedError("Subclasses should implement this method.")
    

class SinusoidalSynthetic(TestFunction):
    r"""
    Computes the function f(x) = sin(x) for a given numpy input x, with independent noise added.

    Args:
        x (np.ndarray): Input array of shape (N, 1) where N is the number of data points.
                        If the input is (N,), it will be automatically reshaped to (N, 1).

    Returns:
        np.ndarray: Output array of shape (N, 1) representing the computed values of f(x).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_space = np.array([[5], [10]])
        self.is_maximize = False
        self.max_x = 9.03835
        self.max_f = 64.4207
        self.min_x = 10
        self.min_f = -80.9928606687

    @add_noise
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Reshape input if necessary
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim == 2 and x.shape[1] == 1:
            pass
        else:
            raise ValueError("Input must be of shape (N,) or (N, 1)")

        # Compute the function
        term1 = -((x - 1) ** 2)
        term2 = np.sin(3 * x + 5 / x + 1)
        return term1 * term2


class BraninHoo(TestFunction):
    r"""
    Computes the Branin-Hoo function, typically used for benchmarking optimization algorithms.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_space = np.array([[0, -5], [15, 15]])
        self.is_maximize = False
        self.min_f = 0.397887

    @add_noise
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(
                "Input array must be two-dimensional with exactly two features per data point."
            )

        # Extract x1 and x2
        x1 = x[:, 0]
        x2 = x[:, 1]

        pi = np.pi

        # Compute the Branin-Hoo function components
        term1 = (x2 - (5.1 / (4 * pi**2)) * x1**2 + (5 / pi) * x1 - 6) ** 2
        term2 = 10 * (1 - 1 / (8 * pi)) * np.cos(x1)

        # Final value computation and reshaping to (N, 1)
        val = (term1 + term2 + 10).reshape(-1, 1)

        return val


class Hartmann6(TestFunction):
    r"""
    Computes the 6-dimensional Hartmann function, typically used for benchmarking optimization algorithms.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_space = np.array([[0] * 6, [1] * 6])
        self.is_maximize = False
        self.min_f = -3.32237

    @add_noise
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError(
                "Input array must be two-dimensional with exactly six features per data point."
            )

        # Define constants for the Hartmann function
        alpha = np.array([1.00, 1.20, 3.00, 3.20])
        A = np.array(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )
        P = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )

        # Compute the Hartmann function
        outer_sum = 0
        for i in range(4):
            inner_sum = np.sum(A[i] * (x - P[i]) ** 2, axis=1)
            outer_sum += alpha[i] * np.exp(-inner_sum)

        # Negate the result to match the typical form of the Hartmann-6 function
        val = -outer_sum.reshape(-1, 1)

        return val



# if __name__ == "__main__":
#     # 動作確認コード
#     # Example usage
#     sinusoidal_func = SinusoidalSynthetic(noise_coeff=3)

#     # Generate data points for visualization
#     x_values = np.linspace(5, 10, 100).reshape(-1, 1)
#     y_values = sinusoidal_func(x_values)

#     # Plot using Plotly
#     import plotly.graph_objects as go

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=x_values.flatten(), y=y_values.flatten(), mode='markers', name='With Noise'))
#     fig.update_layout(
#         title="SinusoidalSynthetic Function with Independent Noise per Input",
#         xaxis_title="x",
#         yaxis_title="f(x)"
#     )
#     fig.show()

#     # BraninHoo関数の動作確認
#     branin_func = BraninHoo(noise_coeff=0.1)  # ノイズを0.1の強度で追加
#     # サンプル入力 (3つのデータポイントで各点が2次元)
#     branin_input = np.array([
#         [1.0, 2.0],
#         [5.0, 5.0],
#         [10.0, 10.0]
#     ])
#     branin_output = branin_func(branin_input)
#     print("BraninHoo Function Output with Noise:")
#     print(branin_output)

#     # Hartmann6関数の動作確認
#     hartmann_func = Hartmann6(noise_coeff=0.1)  # ノイズを0.1の強度で追加
#     # サンプル入力 (3つのデータポイントで各点が6次元)
#     hartmann_input = np.array([
#         [0.2, 0.3, 0.5, 0.7, 0.1, 0.8],
#         [0.4, 0.6, 0.3, 0.9, 0.5, 0.2],
#         [0.1, 0.8, 0.9, 0.4, 0.6, 0.3]
#     ])
#     hartmann_output = hartmann_func(hartmann_input)
#     print("\nHartmann6 Function Output with Noise:")
#     print(hartmann_output)