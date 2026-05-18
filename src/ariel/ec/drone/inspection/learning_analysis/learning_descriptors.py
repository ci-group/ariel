import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

def read_file(monitor_file):
    data = pd.read_csv(monitor_file, skiprows=1)  # Skip the first row (comments)
    episode_rewards = data["r"]  # Rewards per episode
    time_steps = data["t"]  # Timesteps at each episode

    # remove the last episode if it is not finished
    if type(episode_rewards[len(episode_rewards)-1]) == str:
        episode_rewards = episode_rewards[:-1]
        time_steps = time_steps[:-1]

    episode_rewards = np.array(episode_rewards, dtype=float)
    time_steps = np.array(time_steps, dtype=int)

    return episode_rewards, time_steps

def sliding_window_performance(episode_rewards, window_size=50, type_of='mean'):
    performances = []
    for i in range(window_size, len(episode_rewards)+window_size):
        window = episode_rewards[i-window_size:i]
        # set dtype to float to avoid overflow errors
        try:
            window = np.array(window, dtype=float)
        except:
            print("Error in window")
            print(window)
            print(episode_rewards)
            window = np.array(window, dtype=float)
        if type_of == 'mean':
            value = np.mean(window)
        elif type_of == 'max':
            value = np.max(window)
        elif type_of == 'min':
            value = np.min(window)
        elif type_of == 'median':
            try:
                value = np.nanmedian(window)
            except:
                print("Error in window, median")
                print(window)
                print(episode_rewards)
                exit()
        elif type_of == 'std':
            value = np.std(window)
        elif type_of == 'var':
            value = np.var(window)

        performances.append(value)

    return performances

def asymptotic_performance(episode_rewards):
    return np.max(episode_rewards)

def stability_of_learning(episode_rewards): # Mean of the differences between consecutive episode rewards
    diffs = np.abs(np.diff(episode_rewards))
    return np.mean(diffs)

def calculate_burnin_phase_start_time(smoothed_rewards, time_steps):
    return 0

def calculate_burnin_phase_end_time(smoothed_rewards, time_steps):
    last_reward = smoothed_rewards[-1]
    diffs = np.diff(smoothed_rewards)
    idx = np.argmax(diffs)
    return time_steps[idx+1]

def calculate_burnin_phase_start_performance(smoothed_rewards, window_size=50):
    return smoothed_rewards[0]

def calculate_burnin_phase_end_performance(smoothed_rewards, window_size=50):
    last_reward = smoothed_rewards[-1]
    diffs = np.diff(smoothed_rewards)
    idx = np.argmax(diffs)

    if idx >= len(smoothed_rewards):
        return smoothed_rewards[-1]
    
    return smoothed_rewards[idx]
    
def calculate_convergence_phase_start_time(smoothed_rewards_median, time_steps, tol=0.05):
    last_reward = smoothed_rewards_median[-1]
    
    # Find the first index where the rolling std remains below threshold
    if last_reward < 0:
        lower_bound_maintained = smoothed_rewards_median > last_reward*(1+tol)
        upper_bound_maintained = smoothed_rewards_median < last_reward*(1-tol)
    else:
        lower_bound_maintained = smoothed_rewards_median < last_reward*(1+tol)
        upper_bound_maintained = smoothed_rewards_median > last_reward*(1-tol)

    bounds_maintained = np.logical_and(lower_bound_maintained, upper_bound_maintained)
    reversed_bounds_maintained = np.flip(bounds_maintained)

    idx_reversed = np.where(np.logical_not(reversed_bounds_maintained))

    if len(idx_reversed[0]) == 0:
        return time_steps[0]
    
    idx = len(time_steps) - idx_reversed[0][0]

    return time_steps[idx]

def calculate_convergence_phase_end_time(smoothed_rewards_median, time_steps, tol=0.05):
    return time_steps[-1]

def calculate_convergence_phase_start_performance(smoothed_rewards_median, convergence_phase_start_time, time_steps):
    idx = np.where(time_steps == convergence_phase_start_time)[0][0]
    return smoothed_rewards_median[idx]

def calculate_convergence_phase_end_performance(smoothed_rewards_median, time_steps, tol=0.05):
    return smoothed_rewards_median[-1]

def calculate_burnin_phase_speed(burnin_performance, burnin_time):
    if burnin_time == 0:
        return burnin_performance
    return burnin_performance / burnin_time

def calculate_convergence_phase_speed(convergence_time, convergence_performance):
    if convergence_time == 0:
        return convergence_performance
    return convergence_performance / convergence_time

def calculate_intermediate_phase_time_performance(convergence_time, convergence_performance, burnin_time, burnin_performance):
    if burnin_time > convergence_time:
        return np.nan, np.nan
    
    return convergence_time - burnin_time, convergence_performance - burnin_performance

def calculate_intermediate_speed(int_time, int_performance):
    if int_time == 0:
        return np.nan
    return int_performance / int_time

def process_individual(args):
    """Process a single individual's data - designed for parallel execution"""
    directory, experiment, g, local_idx, ind_num, window_size = args
    
    monitor_file_path = os.path.join(directory, experiment, f'gen{g}', f'ind{ind_num}', 'monitor.csv')
    
    if os.path.exists(monitor_file_path):
        try:
            episode_rewards, time_steps = read_file(monitor_file_path)
            smoothed_rewards_median = sliding_window_performance(episode_rewards, window_size=window_size, type_of='median')

            burnin_start_time = calculate_burnin_phase_start_time(smoothed_rewards_median, time_steps)
            burnin_end_time = calculate_burnin_phase_end_time(smoothed_rewards_median, time_steps)
            burnin_start_performance = calculate_burnin_phase_start_performance(smoothed_rewards_median, window_size=window_size)
            burnin_end_performance = calculate_burnin_phase_end_performance(smoothed_rewards_median, window_size=window_size)
            convergence_start_time = calculate_convergence_phase_start_time(smoothed_rewards_median, time_steps, tol=0.15)
            convergence_end_time = calculate_convergence_phase_end_time(smoothed_rewards_median, time_steps, tol=0.15)
            convergence_start_performance = calculate_convergence_phase_start_performance(smoothed_rewards_median, convergence_start_time, time_steps)
            convergence_end_performance = calculate_convergence_phase_end_performance(smoothed_rewards_median, time_steps, tol=0.15)
            
            burnin_speed = calculate_burnin_phase_speed(burnin_end_performance-burnin_start_performance, burnin_end_time)
            convergence_speed = calculate_convergence_phase_speed(convergence_end_time-convergence_start_time, convergence_end_performance-convergence_start_performance)
            intermediate_time, intermediate_performance = calculate_intermediate_phase_time_performance(convergence_start_time, convergence_start_performance, burnin_end_time, burnin_end_performance)
            intermediate_speed = calculate_intermediate_speed(intermediate_time, intermediate_performance)

            mxr = np.max(episode_rewards)
            volatility = stability_of_learning(episode_rewards)
            
            return (g, local_idx, [burnin_start_time, burnin_end_time, burnin_start_performance, burnin_end_performance, burnin_speed,
                          convergence_start_time, convergence_end_time, convergence_start_performance, convergence_end_performance, convergence_speed,
                          intermediate_time, intermediate_performance, intermediate_speed,
                          mxr, volatility])
        except Exception as e:
            print(f'Error processing {monitor_file_path}: {e}')
            return (g, local_idx, [np.nan, np.nan, np.nan, np.nan, np.nan, 
                          np.nan, np.nan, np.nan, np.nan, np.nan,
                          np.nan, np.nan, np.nan,
                          np.nan, np.nan])
    else:
        print(f'File {monitor_file_path} does not exist')
        return (g, local_idx, [np.nan, np.nan, np.nan, np.nan, np.nan, 
                      np.nan, np.nan, np.nan, np.nan, np.nan,
                      np.nan, np.nan, np.nan,
                      np.nan, np.nan])

def get_learning_data(directory, num_generations=40, window_size=1000, type_of='median', 
                     multiple_experiments=True, n_processes=None):
    """
    Extract learning data from experiment directories with optional parallelization.
    Automatically detects the number of individuals per generation from folder structure.
    
    Parameters:
    -----------
    directory : str
        Path to the data directory
    num_generations : int, default=40
        Number of generations to process
    window_size : int, default=1000
        Window size for sliding window performance calculation
    type_of : str, default='median'
        Type of sliding window calculation
    multiple_experiments : bool, default=True
        Whether to process multiple experiment folders or treat directory as single experiment
    n_processes : int, optional
        Number of processes for parallelization. If None, uses all available CPU cores
    
    Returns:
    --------
    total_data : list
        Learning data with shape (num_experiments, num_generations, max_individuals_found, 15)
    """
    
    if n_processes is None:
        n_processes = cpu_count()
    
    start_time = time.time()
    
    if multiple_experiments:
        # List all folders in the directory as experiment folders
        experiment_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    else:
        # Treat the directory itself as the single experiment
        experiment_folders = ['.']
        directory = os.path.dirname(directory)
        if directory == '':
            directory = '.'
    
    total_data = []
    
    print(os.listdir(directory))
    for experiment in experiment_folders:
        print(f'Processing experiment: {experiment}')
        exp_start_time = time.time()
        
        # Discover the actual structure by scanning directories
        experiment_path = os.path.join(directory, experiment)
        generation_structure = {}
        max_individuals = 0
        
        # Scan each generation to find individual folders
        for g in range(num_generations):
            gen_path = os.path.join(experiment_path, f'gen{g}')
            if os.path.exists(gen_path):
                # Find all individual folders (ind*)
                individual_folders = [f for f in os.listdir(gen_path) 
                                    if os.path.isdir(os.path.join(gen_path, f)) and f.startswith('ind')]
                
                # Extract individual numbers and sort them
                individual_numbers = []
                for ind_folder in individual_folders:
                    try:
                        ind_num = int(ind_folder.replace('ind', ''))
                        individual_numbers.append(ind_num)
                    except ValueError:
                        print(f"Warning: Could not parse individual folder name: {ind_folder}")
                        continue
                
                individual_numbers.sort()
                generation_structure[g] = individual_numbers
                max_individuals = max(max_individuals, len(individual_numbers))
                
                print(f"  Generation {g}: Found {len(individual_numbers)} individuals (ind{min(individual_numbers)} to ind{max(individual_numbers)})")
            else:
                print(f"  Generation {g}: Directory not found - {gen_path}")
                generation_structure[g] = []
        
        print(f"Maximum individuals in any generation: {max_individuals}")
        
        # Initialize data structure for this experiment
        exp_data = []
        
        # Prepare arguments for parallel processing
        parallel_args = []
        
        for g in range(num_generations):
            individual_numbers = generation_structure.get(g, [])
            for local_idx, ind_num in enumerate(individual_numbers):
                parallel_args.append((directory, experiment, g, local_idx, ind_num, window_size))
        
        # Process individuals in parallel
        print(f'Processing {len(parallel_args)} individuals using {n_processes} processes...')
        
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_individual, parallel_args)
        
        # Organize results back into the expected structure
        for g in range(num_generations):
            individual_numbers = generation_structure.get(g, [])
            gen_data = [None] * max_individuals  # Initialize with max size
            
            # Fill in the actual data
            for local_idx, ind_num in enumerate(individual_numbers):
                # Find the result for this generation and local index
                for gen_idx, loc_idx, data in results:
                    if gen_idx == g and loc_idx == local_idx:
                        gen_data[local_idx] = data
                        break
                else:
                    # If no data found, fill with NaN
                    gen_data[local_idx] = [np.nan] * 15
            
            exp_data.append(gen_data)
        
        total_data.append(exp_data)
        print(f'Experiment {experiment} completed in {time.time() - exp_start_time:.2f} seconds')
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return total_data

def plot_learning_data(learning_data):
    num_experiments, num_generations, pop_size, nparams = learning_data.shape
    fig1, axs1 = plt.subplots(3, 2, figsize=(15, 10))
    axs1 = axs1.flatten()
    fig2, axs2 = plt.subplots(3, 2, figsize=(15, 10))
    axs2 = axs2.flatten()
    labels = ['Time of burnin phase', 'Peak after burnin phase', 'Asymptotic performance', 'Stability of learning', 'Point of convergence', 'Learnability']
    
    avr_data_across_gens = np.nanmean(learning_data, axis=2)

    avr_data = np.nanmean(avr_data_across_gens, axis=0)
    std_data = np.nanstd(avr_data_across_gens, axis=0)

    for i in range(nparams):
        ax = axs1[i]
        ax.set_title(labels[i])
        for j in range(num_experiments):
            ax.plot(avr_data_across_gens[j, :, i].T, alpha=0.3, label=f'Experiment {j}')

        ax.grid(True)
    
    fig1.suptitle('Learning descriptors across different experiments')

    for i in range(nparams):
        ax = axs2[i]
        ax.set_title(labels[i])
        ax.plot(avr_data[:, i], label='Mean')
        ax.fill_between(np.arange(num_generations), avr_data[:, i] - std_data[:, i], avr_data[:, i] + std_data[:, i], alpha=0.3, label='Std')
        ax.grid(True)

    fig2.suptitle('Learning descriptors across different generations')

    return fig1, axs1, fig2, axs2

def save_learning_data_as_csv(learning_data, output_path):
    """
    Save learning data as a CSV file with proper column headers.
    
    Parameters:
    -----------
    learning_data : list
        Learning data from get_learning_data function
    output_path : str
        Path where to save the CSV file
    """
    # Column headers for the learning data
    columns = [
        'experiment_id', 'generation', 'individual', 
        'burnin_start_time', 'burnin_end_time', 'burnin_start_performance', 
        'burnin_end_performance', 'burnin_speed',
        'convergence_start_time', 'convergence_end_time', 'convergence_start_performance', 
        'convergence_end_performance', 'convergence_speed',
        'intermediate_time', 'intermediate_performance', 'intermediate_speed',
        'max_reward', 'volatility'
    ]
    
    # Flatten the data and create DataFrame
    rows = []
    for exp_id, experiment in enumerate(learning_data):
        for gen_id, generation in enumerate(experiment):
            for ind_id, individual_data in enumerate(generation):
                if individual_data is not None and not all(np.isnan(individual_data) if isinstance(individual_data, (list, np.ndarray)) else [False]):
                    row = [exp_id, gen_id, ind_id] + individual_data
                    rows.append(row)
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Learning data saved to: {output_path}")
    print(f"Total records saved: {len(df)}")
    
    # Print summary of individuals per generation
    if len(df) > 0:
        gen_counts = df.groupby('generation')['individual'].count()
        print("Individuals per generation:")
        for gen, count in gen_counts.items():
            print(f"  Generation {gen}: {count} individuals")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process learning data and optionally plot results.")
    
    # Add arguments
    parser.add_argument('--mode', type=str, choices=['single', 'collect'], required=True,
                       help='Mode: "single" for single file analysis, "collect" for full data collection')
    parser.add_argument('--data_path', type=str, nargs='*', action='append', required=True,
                       help='Path(s) to data directory/directories (for collect mode) or monitor.csv file (for single mode). Can be specified multiple times or as space-separated list.')
    parser.add_argument('--num_generations', type=int, default=40,
                       help='Number of generations to process (collect mode only)')
    parser.add_argument('--window_size', type=int, default=1000,
                       help='Window size for sliding window performance calculation')
    parser.add_argument('--multiple_experiments', action='store_true', default=True,
                       help='Whether to process multiple experiment folders (collect mode only)')
    parser.add_argument('--n_processes', type=int, default=None,
                       help='Number of processes for parallelization (collect mode only)')
    
    args = parser.parse_args()
    
    # Flatten the data_path list (in case of nested lists from action='append')
    data_paths = []
    for path_group in args.data_path:
        if isinstance(path_group, list):
            data_paths.extend(path_group)
        else:
            data_paths.append(path_group)
    
    # Remove any empty strings
    data_paths = [path for path in data_paths if path]
    
    if not data_paths:
        print("Error: No data paths provided")
        exit(1)
    
    start_time = time.time()
    
    if args.mode == 'single':
        # Single file analysis mode
        print(f"Performing single analysis on: {args.data_path[0][0]}")
        
        if not os.path.exists(args.data_path[0][0]):
            print(f"Error: File {args.data_path[0][0]} does not exist")
            exit(1)
            
        episode_rewards, time_steps = read_file(args.data_path[0][0])
        last_reward = episode_rewards[-1]
        smoothed_rewards_median = sliding_window_performance(episode_rewards, window_size=args.window_size, type_of='median')
        minxr = np.min(smoothed_rewards_median)

        burnin_start_time = calculate_burnin_phase_start_time(smoothed_rewards_median, time_steps)
        burnin_end_time = calculate_burnin_phase_end_time(smoothed_rewards_median, time_steps)
        burnin_start_performance = calculate_burnin_phase_start_performance(smoothed_rewards_median, window_size=args.window_size)
        burnin_end_performance = calculate_burnin_phase_end_performance(smoothed_rewards_median, window_size=args.window_size)
        convergence_start_time = calculate_convergence_phase_start_time(smoothed_rewards_median, time_steps, tol=0.1)
        convergence_end_time = calculate_convergence_phase_end_time(smoothed_rewards_median, time_steps, tol=0.1)
        convergence_start_performance = calculate_convergence_phase_start_performance(smoothed_rewards_median, convergence_start_time, time_steps)
        convergence_end_performance = calculate_convergence_phase_end_performance(smoothed_rewards_median, time_steps, tol=0.1)
        burnin_speed = calculate_burnin_phase_speed(burnin_end_performance-burnin_start_performance, burnin_end_time)
        convergence_speed = calculate_convergence_phase_speed(convergence_end_time-convergence_start_time, convergence_end_performance-convergence_start_performance)
        intermediate_time, intermediate_performance = calculate_intermediate_phase_time_performance(convergence_start_time, convergence_start_performance, burnin_end_time, burnin_end_performance)
        intermediate_speed = calculate_intermediate_speed(intermediate_time, intermediate_performance)
        
        ap = asymptotic_performance(episode_rewards)
        sl = stability_of_learning(episode_rewards)

        print(f"Burnin phase start time: {burnin_start_time}")
        print(f"Burnin phase end time: {burnin_end_time}")
        print(f"Burnin phase start performance: {burnin_start_performance}")
        print(f"Burnin phase end performance: {burnin_end_performance}")
        print(f"Convergence phase start time: {convergence_start_time}")
        print(f"Convergence phase end time: {convergence_end_time}")
        print(f"Convergence phase start performance: {convergence_start_performance}")
        print(f"Convergence phase end performance: {convergence_end_performance}")
        print(f"Burnin phase speed: {burnin_speed}")
        print(f"Convergence phase speed: {convergence_speed}")
        print(f"Intermediate phase time: {intermediate_time}")
        print(f"Intermediate phase performance: {intermediate_performance}")
        print(f"Intermediate phase speed: {intermediate_speed}")
        print("Asymptotic performance: ", ap)
        print("Stability of learning: ", sl)

        # Create plot /home/jed/workspaces/airevolve/data_backup/asym_circle/asym_circle4evo_logs_20250320_095305/gen15/ind364/figure.png
        plt.figure(figsize=(12, 8))
        plt.plot(time_steps, episode_rewards, label='Episode rewards', color='gray', alpha=0.5)
        plt.plot(time_steps, smoothed_rewards_median, label='Smoothed rewards median', linewidth=5)
        plt.plot(burnin_end_time, burnin_end_performance, color='red', label='Point of Fastest Learning Progress', marker='o', markersize=20)
        plt.plot([burnin_end_time, burnin_end_time], [minxr, burnin_end_performance], color='red', linestyle='--', linewidth=5)
        plt.plot([0, burnin_end_time], [burnin_end_performance, burnin_end_performance], color='red', linestyle='--', linewidth=5)
        plt.plot([0, time_steps[-1]], [last_reward*(1+0.1), last_reward*(1+0.1)], color='purple', linestyle='--', linewidth=5)
        plt.plot([0, time_steps[-1]], [last_reward, last_reward], color='purple', linestyle='-', label='Last reward', linewidth=5)
        plt.plot([0, time_steps[-1]], [last_reward*(1-0.1), last_reward*(1-0.1)], color='purple', linestyle='--', label='Convergence bounds', linewidth=5)
        plt.plot(convergence_start_time, convergence_start_performance, color='purple', label='Point of convergence', marker='o', markersize=20)
        plt.legend(fontsize=21, loc='lower right')
        plt.xlabel('Time Steps', fontsize=28)
        plt.ylabel('Reward', fontsize=28)
        # Set ticksize
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        # plt.title('Learning Analysis')
        plt.grid(True)
        
        # Save plot in the same directory as the data file
        plot_path = os.path.join(os.path.dirname(args.data_path[0][0]), 'learning_analysis.png')
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        
    elif args.mode == 'collect':
        # Full data collection mode

        
        # Check if multiple paths are provided
        if len(data_paths) > 1:
            print(f"Processing multiple directories: {data_paths}")
            for path in data_paths:            
                if not os.path.exists(path):
                    print(f"Error: Directory {path} does not exist")
                    exit(1)
                    
                learning_data = get_learning_data(
                    directory=path,
                    num_generations=args.num_generations,
                    window_size=args.window_size,
                    multiple_experiments=args.multiple_experiments,
                    n_processes=args.n_processes
                )
                
                # Save the data as CSV
                output_path = os.path.join(path, 'learning_data.csv')
                save_learning_data_as_csv(learning_data, output_path)
        else:
            print(f"Performing full data collection on: {args.data_path}")
            if not os.path.exists(data_paths[0]):
                print(f"Error: Directory {data_paths[0]} does not exist")
                exit(1)
                
            learning_data = get_learning_data(
                directory=data_paths[0],
                num_generations=args.num_generations,
                window_size=args.window_size,
                multiple_experiments=args.multiple_experiments,
                n_processes=args.n_processes
            )
            
            # Save the data as CSV
            output_path = os.path.join(data_paths[0], 'learning_data.csv')
            save_learning_data_as_csv(learning_data, output_path)
        # Optionally create plots
        try:
            learning_data_array = np.array(learning_data)
            fig1, axs1, fig2, axs2 = plot_learning_data(learning_data_array)
            
            # Save plots
            plot1_path = os.path.join(args.data_path, 'learning_descriptors_experiments.png')
            plot2_path = os.path.join(args.data_path, 'learning_descriptors_generations.png')
            fig1.savefig(plot1_path)
            fig2.savefig(plot2_path)
            print(f"Plots saved to: {plot1_path} and {plot2_path}")
            
        except Exception as e:
            print(f"Warning: Could not create plots - {e}")
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    # Example usage:
    # 
    # Single analysis:
    # python script.py --mode single --data_path /path/to/monitor.csv
    #
    # Multiple directories data collection (auto-detects population sizes):
    # python airevolve/inspection_tools/learning_descriptors.py --mode collect --data_path /path/to/data1 /path/to/data2 --multiple_experiments
    # python airevolve/inspection_tools/learning_descriptors.py --mode collect --data_path /media/jed/My\ Passport/asym_figure8/ /media/jed/My\ Passport/asym_slalom/ --multiple_experiments
    # 
    # Using wildcards (bash will expand):
    # python script.py --mode collect --data_path /path/to/exp_* --num_generations 50
    #
    # From file list:
    # python script.py --mode collect --data_path $(cat directory_list.txt)

# def did_catastrophic_forgetting_occur():
#     pass