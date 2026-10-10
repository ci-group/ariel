import pandas as pd


def tournament_selection(population: pd.DataFrame, tournament_size: int = 3, k: int = 1) -> pd.DataFrame:
    """Tournament selection on a population DataFrame with a 'fitness' column."""
    assert len(population) > 0, "Population must not be empty"

    selected_individuals = []
    for _ in range(k):
        tournament = population.sample(n=tournament_size, replace=True).reset_index(drop=True)
        winner = tournament.loc[tournament["fitness"].idxmax()]
        selected_individuals.append(winner.to_dict())

    return pd.DataFrame(selected_individuals).reset_index(drop=True)
