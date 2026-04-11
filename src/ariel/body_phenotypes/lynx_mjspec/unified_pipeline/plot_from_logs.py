import matplotlib.pyplot as plt
import numpy as np
import re


def _parse_log(file_path: str) -> dict[str, list]:
    with open(file_path, 'r') as f:
        content = f.read()
    gens   = [int(m)   for m in re.findall(r"gen=(\d+)",        content)]
    bests  = [float(m) for m in re.findall(r"best=([-\d\.]+)",  content)]
    means  = [float(m) for m in re.findall(r"mean=([-\d\.]+)",  content)]
    worsts = [float(m) for m in re.findall(r"worst=([-\d\.]+)", content)]
    n = min(len(gens), len(bests), len(means), len(worsts))
    return {"gens": gens[:n], "bests": bests[:n], "means": means[:n], "worsts": worsts[:n]}


def plot_training_logs(log_files):
    """
    Parses multiple log files and plots aggregated statistics across runs.
    Each metric (best, mean) is shown as the cross-run mean ± 1 std band.

    Args:
        log_files (str or list): A single file path or a list of file paths.
    """
    if isinstance(log_files, str):
        log_files = [log_files]

    # Parse all runs and align to the shortest common length.
    runs = []
    for fp in log_files:
        try:
            parsed = _parse_log(fp)
            if parsed["gens"]:
                runs.append(parsed)
            else:
                print(f"Warning: No valid data in '{fp}'.")
        except FileNotFoundError:
            print(f"Error: '{fp}' not found.")

    if not runs:
        print("No data to plot.")
        return

    min_len = min(len(r["gens"]) for r in runs)
    gens = runs[0]["gens"][:min_len]

    bests_mat  = np.array([r["bests"][:min_len]  for r in runs])  # (n_runs, n_gens)
    means_mat  = np.array([r["means"][:min_len]  for r in runs])
    worsts_mat = np.array([r["worsts"][:min_len] for r in runs])

    agg_best_mean  = bests_mat.mean(axis=0)
    agg_best_std   = bests_mat.std(axis=0)
    agg_mean_mean  = means_mat.mean(axis=0)
    agg_mean_std   = means_mat.std(axis=0)
    agg_worst_mean = worsts_mat.mean(axis=0)

    n_runs = len(runs)
    plt.figure(figsize=(10, 5))

    # Mean-of-best across runs with ±1σ band.
    plt.plot(gens, agg_best_mean, label=f'Best (mean of {n_runs} runs)', color='darkorange', linewidth=2, linestyle='--')
    # plt.fill_between(gens,
    #                  agg_best_mean - agg_best_std,
    #                  agg_best_mean + agg_best_std,
    #                  color='darkorange', alpha=0.2, label='Best ±1σ across runs')

    # Mean-of-mean across runs with ±1σ band.
    plt.plot(gens, agg_mean_mean, label=f'Mean (mean of {n_runs} runs)', color='steelblue', linewidth=2)
    plt.fill_between(gens,
                     agg_mean_mean - agg_mean_std,
                     agg_mean_mean + agg_mean_std,
                     color='steelblue', alpha=0.2, label='Mean ±1σ across runs')

    # # Worst envelope (mean across runs) for context.
    # plt.plot(gens, agg_worst_mean, label='Worst (mean across runs)',
    #          color='gray', linewidth=1, linestyle=':')

    # Chart Formatting
    plt.title(f'Training Metrics — {n_runs} runs aggregated', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    
    # Place legend outside the plot so it doesn't obscure data
    plt.legend(loc='upper right')    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('lynx_arm_experiment_10x.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Example Usage
# ==========================================

# 1. Plotting a single file:
# plot_training_logs('train.log')

# 2. Plotting multiple files to compare runs:
# plot_training_logs(['train_run1.log', 'train_run2.log'])

if __name__ == "__main__":
    from pathlib import Path

    batch_dir = Path("__data__/lynx_mjspec/unified_batch_10x")

    # Collect train.log from every real run directory (skip symlinks like latest_run).
    log_files = sorted(
        str(p / "train.log")
        for p in batch_dir.iterdir()
        if p.is_dir() and not p.is_symlink() and (p / "train.log").exists()
    )

    if not log_files:
        print(f"No train.log files found under {batch_dir}")
    else:
        print(f"Plotting {len(log_files)} runs:")
        for f in log_files:
            print(f"  {f}")
        plot_training_logs(log_files)
