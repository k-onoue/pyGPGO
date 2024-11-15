import torch
from torch.optim import Adam
from _src import ModelConfig, GaussianProcess, SinusoidalSynthetic
from _src import (
    ProbabilityOfImprovement,
    LogProbabilityOfImprovement,
    ExpectedImprovement,
    LogExpectedImprovement,
    UpperConfidenceBound
)

def find_argmax_acquisition(acquisition, search_space, num_restarts=10, raw_samples=100):
    bounds = torch.tensor(search_space, dtype=torch.float32)
    best_value = None
    best_point = None

    for _ in range(num_restarts):
        # Random initialization within the bounds
        x0 = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, dtype=torch.float32)
        x0.requires_grad = True

        optimizer = Adam([x0], lr=0.01)

        for _ in range(raw_samples):
            optimizer.zero_grad()
            loss = -acquisition(x0)
            loss.backward(retain_graph=True)
            optimizer.step()

            # Clamp to the bounds
            with torch.no_grad():
                x0.clamp_(bounds[0], bounds[1])

        value = acquisition(x0).item()
        if best_value is None or value > best_value:
            best_value = value
            best_point = x0.detach().clone()

    return best_point, best_value


def main():

    n_iter = 20
    is_maximize = False

    objective_function = SinusoidalSynthetic(maximize=is_maximize)
    search_space = objective_function.search_space # lb = search_space[0], ub = search_space[1]

    print(f"lb: {search_space[0]}, ub: {search_space[1]}")
    print(f"is_maximize: {is_maximize}")

    initial_X = []
    initial_X.append(search_space[0])
    initial_X.append(search_space[1])
    initial_X.append((search_space[1] + search_space[0]) / 2)
    initial_X = torch.tensor(initial_X)

    initial_y = objective_function(initial_X)

    print(f"initial_X: \n{initial_X}")
    print(f"initial_y: \n{initial_y}")

    X = initial_X
    y = initial_y

    for iteration in range(n_iter):
        config = ModelConfig(ard_num_dims=None, kernel_type="rbf")
        model = GaussianProcess(config)
        model.fit(X, y)

        print(f'y_max {y.max()}, y_min {y.min()}')
        print(f'best_f {y.max() if is_maximize else y.min()}, maximize {is_maximize}')

        best_f = y.max() if is_maximize else y.min()
        # acquisition = ProbabilityOfImprovement(model, best_f=best_f, maximize=is_maximize)
        # acquisition = UpperConfidenceBound(model, maximize=is_maximize)
        acquisition = ExpectedImprovement(model, best_f=best_f, maximize=is_maximize)

        print(f"Acquisition is_maximize?: {acquisition.maximize}")

        best_point, best_value = find_argmax_acquisition(acquisition, search_space)
        print(f"Iteration {iteration + 1}: Best point: {best_point}, Best value: {best_value}")

        X = torch.cat((X, best_point))
        y = torch.cat((y, objective_function(best_point)))

        print()

    print(y)
    optimal_idx = torch.argmax(y) if is_maximize else torch.argmin(y)
    print(f"Optimal point: {X[optimal_idx]}, Optimal value: {y[optimal_idx]}")


if __name__ == "__main__":
    main()