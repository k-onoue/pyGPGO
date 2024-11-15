import torch
from torch.optim import Adam
import optuna
from _src import ModelConfig, GaussianProcess, SinusoidalSynthetic
from _src import (
    ProbabilityOfImprovement,
    LogProbabilityOfImprovement,
    ExpectedImprovement,
    LogExpectedImprovement,
    UpperConfidenceBound
)

def optimize_acquisition_with_optuna(acquisition, search_space, n_trials=100):
    bounds = torch.tensor(search_space, dtype=torch.float32)

    def objective(trial):
        x = bounds[0] + (bounds[1] - bounds[0]) * torch.tensor(trial.suggest_uniform('x', 0, 1), dtype=torch.float32)
        return acquisition(x).item()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_x = bounds[0] + (bounds[1] - bounds[0]) * torch.tensor(study.best_params['x'], dtype=torch.float32)
    best_value = study.best_value

    return best_x, best_value

def main():
    objective_function = SinusoidalSynthetic(maximize=True)
    search_space = objective_function.search_space # lb = search_space[0], ub = search_space[1]
    is_maximize = objective_function.maximize

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

    for iteration in range(50):
        config = ModelConfig(ard_num_dims=None, kernel_type="rbf")
        model = GaussianProcess(config)
        model.fit(X, y)

        print(f'y_max {y.max()}, y_min {y.min()}')
        print(f'best_f {y.max() if is_maximize else y.min()}, maximize {is_maximize}')

        best_f = y.max() if is_maximize else y.min()
        acquisition = ProbabilityOfImprovement(model, best_f=best_f, maximize=is_maximize)

        print(f"Acquisition is_maximize?: {acquisition.maximize}")

        best_point, best_value = optimize_acquisition_with_optuna(acquisition, search_space)
        print(f"Iteration {iteration + 1}: Best point: {best_point}, Best value: {best_value}")

        # X = torch.cat((X, best_point.unsqueeze(0)))
        # y = torch.cat((y, objective_function(best_point.unsqueeze(0))))

        X = torch.cat((X, best_point))
        y = torch.cat((y, objective_function(best_point.unsqueeze(0))))

        print()

    print(y)
    optimal_idx = torch.argmax(y) if is_maximize else torch.argmin(y)
    print(f"Optimal point: {X[optimal_idx]}, Optimal value: {y[optimal_idx]}")


if __name__ == "__main__":
    main()
