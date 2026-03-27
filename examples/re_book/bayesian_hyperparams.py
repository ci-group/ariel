import optuna
import subprocess
import re
import sys
from rich.console import Console

console = Console()

# Name of your target file
TARGET_SCRIPT = "/home/aronf/Desktop/EvoDevo/pure-ariel/ariel/examples/re_book/1_brain_evolution_multithreaded.py"


def objective(trial):
    # 1. Define the search space
    # Sigma is usually sensitive to orders of magnitude, so log=True is best
    cma_sigma = trial.suggest_float("sigma", 0.01, 0.3, log=True)
    cma_popsize = trial.suggest_int("population", 4, 32, step=4)
    
    # 2. Build the command
    # Keeping budget/dur low so trials don't take hours
    cmd = [
        sys.executable, TARGET_SCRIPT,
        "--population", str(cma_popsize),
        "--sigma", str(cma_sigma),
        "--budget", "5",     # Lower budget for faster sweeps
        "--dur", "5",        # 5-second simulation per evaluation
        "--workers", "4",     # Adjust to your CPU cores
        "--tune"
    ]

    print(f"Running Trial {trial.number}: population={cma_popsize}, sigma={cma_sigma:.4f}")

    # 3. Run the subprocess
    # We capture standard output. Because it's piped, 'rich' will automatically 
    # disable colors/ANSI escape codes, making regex parsing very clean.
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600 # 10-minute timeout per trial safeguard
        )
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out. Pruning.")
        raise optuna.exceptions.TrialPruned()

    # 4. Parse the objective from your existing print statements
    # We look for all instances of "Best Fit (Gen): XXX"
    matches = re.findall(r"Best Fit \(Gen\): ([0-9.-]+)", result.stdout)
    
    if not matches:
        print(f"Failed to find fitness output in Trial {trial.number}.")
        print(f"STDERR:\n{result.stderr}")
        raise optuna.exceptions.TrialPruned("Failed to parse fitness.")

    # The final generation's best fit is the last match in the list
    final_fitness = float(matches[-1])
    
    if final_fitness == float('inf'):
        raise optuna.exceptions.TrialPruned("Evolution returned infinite fitness.")

    return final_fitness

def main():
    print("Starting Optuna Hyperparameter Tuning...")
    
    # You are minimizing distance to target, so direction is "minimize"
    study = optuna.create_study(
        study_name="spider_cma_tuning",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run 20 trials
    try:
        study.optimize(objective, n_trials=20)
    except KeyboardInterrupt:
        print("\nOptimization interrupted. Showing best results so far...")

    # Print results
    print("\n--- Tuning Complete ---")
    best_trial = study.best_trial

    print(f"Best Trial ID: {best_trial.number}")
    print(f"Best Fitness:  {best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()