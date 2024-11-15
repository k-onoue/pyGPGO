import optuna
import torch
from _src import SinusoidalSynthetic

# Define the objective function for Optuna
def objective(trial):
    # Instantiate the SinusoidalSynthetic objective function
    is_maximize = False  # Change to True if you want to maximize
    objective_function = SinusoidalSynthetic(maximize=is_maximize)

    # Define the search space for the optimization
    lb, ub = objective_function.search_space[0].item(), objective_function.search_space[1].item()
    x = trial.suggest_float("x", lb, ub)

    # Convert x to a tensor for evaluation
    x_tensor = torch.tensor([[x]], dtype=torch.float32)

    # Evaluate the objective function
    y = objective_function(x_tensor).item()

    # Return the objective value (negative for maximization)
    return y if not is_maximize else -y

# Run the optimization with Optuna
if __name__ == "__main__":
    # Use TPE Sampler
    sampler = optuna.samplers.GPSampler()
    # sampler = optuna.samplers.TPESampler()

    # Create a study
    study = optuna.create_study(sampler=sampler, direction="minimize")  # Use "maximize" if is_maximize=True

    # Optimize the objective
    study.optimize(objective, n_trials=20)

    # Print the best result
    print(f"Best trial: {study.best_trial}")
    print(f"Best value: {study.best_value}")
    print(f"Best parameters: {study.best_params}")
