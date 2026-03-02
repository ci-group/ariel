# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np

def plot_fit_per_gen():
    data = pd.read_sql("SELECT * FROM individual", sqlite3.connect("__data__/database.db"))

    # Get minimum and maximum generation number
    min_gen = int(data['time_of_birth'].min())
    max_gen = int(data['time_of_death'].max())

    # Get all individuals that are alive in each generation
    population_per_gen = {
        gen: data.loc[(data['time_of_birth'] <= gen) & (data['time_of_death'] > gen), 'id'].tolist()
        for gen in range(min_gen, max_gen + 1)
    }

    # Structure dataframe for easier viewing
    pop_df = pd.DataFrame({
        'generation': list(population_per_gen.keys()),
        'individuals': list(population_per_gen.values()),
        'pop size': [len(v) for v in population_per_gen.values()]
        })



    # Get the fitness of every individual
    fitness_by_id = data.set_index('id')['fitness_']

    # Initialise lists
    means = []
    stds = []
    maxs = []

    # Get data from dataframe
    for gen in pop_df['generation']:
        ids = population_per_gen.get(int(gen), [])
        if not ids:
            means.append(np.nan)
            stds.append(np.nan)
            maxs.append(np.nan)
            continue

        fits = fitness_by_id.reindex(ids).dropna().astype(float).values

        # If there is no data, initialise it as nan to avoid erors
        if fits.size == 0:
            means.append(np.nan)
            stds.append(np.nan)
            maxs.append(np.nan)
        
        # If there is data, add it to the list
        else:
            means.append(float(np.mean(fits)))
            stds.append(float(np.std(fits, ddof=0)))
            maxs.append(float(np.min(fits)))

    # Add data to separate columns
    pop_df['fitness_mean'] = means
    pop_df['fitness_std'] = stds
    pop_df['fitness_best'] = maxs

    # Create a copy of dataframe
    df = pop_df.copy()

    # Create mask to avoid missing data 
    # if the evolution process was done correctly this should not change anything.
    # This mask could be used to get custom data out of the dataframe
    mask = df['fitness_mean'].notna()

    # Get data according to the mask
    x = df.loc[mask, 'generation']
    mean = df.loc[mask, 'fitness_mean']
    std = df.loc[mask, 'fitness_std']
    maxv = df.loc[mask, 'fitness_best']

    # Generate line plot of fitness per generation
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean, label='Fitness mean', color='C0', linewidth=2)
    plt.plot(x, maxv, label='Fitness best', color='C1', linestyle='--', linewidth=2)
    plt.fill_between(x, mean - std, mean + std, color='C0', alpha=0.25, label='Mean ± Std')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness statistics per generation')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yticks(range(0, int(max(df["fitness_mean"]) + 5), 2))
    plt.tight_layout()
    plt.show()