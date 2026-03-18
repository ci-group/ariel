"""Example: one-max EA using the refactored ariel.ec module."""

from typing import cast

from rich.console import Console
from rich.traceback import install

from ariel.ec import (
    EA,
    Crossover,
    EAStep,
    Individual,
    IntegerMutator,
    IntegersGenerator,
    Population,
    config,
)

install()
console = Console()


# ── Individual factory ────────────────────────────────────────────────────────

def make_individual() -> Individual:
    ind = Individual()
    ind.genotype = IntegersGenerator.integers(low=0, high=1, size=10)
    return ind


# ── EA steps ─────────────────────────────────────────────────────────────────

def evaluate(population: Population) -> Population:
    for ind in population.unevaluated:
        ind.fitness = float(sum(1 for gene in ind.genotype if gene == 1))
    return population


def parent_selection(population: Population) -> Population:

    shuffled = population.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
            if ind_a.fitness_ >= ind_b.fitness_:
                ind_a.tags = {"selected": True}
                ind_b.tags = {"selected": False}
            else:
                ind_a.tags = {"selected": False}
                ind_b.tags = {"selected": True}

    return shuffled


def crossover(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    for idx in range(0, len(parents) - 1, 2):
        p_a = parents[idx]
        p_b = parents[idx + 1]
        g_a, g_b = Crossover.one_point(
            cast("list[int]", p_a.genotype),
            cast("list[int]", p_b.genotype),
        )

        child_a = Individual()
        child_a.genotype = g_a
        child_a.tags = {"mutate": True}

        child_b = Individual()
        child_b.genotype = g_b
        child_b.tags = {"mutate": True}

        population.extend([child_a, child_b])
    return population


def mutate(population: Population) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        ind.genotype = IntegerMutator.integer_creep(
            individual=cast("list[int]", ind.genotype),
            span=1,
            mutation_probability=0.2,
        )
        ind.requires_eval = True
    return population


def survivor_selection(population: Population) -> Population:
    shuffled = population.alive.shuffle()
    alive_count = len(shuffled)
    for idx in range(0, len(shuffled) - 1, 2):
        if alive_count <= config.target_population_size:
            break
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if (ind_a.fitness_ or 0.0) >= (ind_b.fitness_ or 0.0):
            ind_b.alive = False
        else:
            ind_a.alive = False
        alive_count -= 1
    return population


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    initial = Population([make_individual() for _ in range(20)])
    initial = evaluate(initial)

    ops: list[EAStep] = [
        EAStep("parent_selection", parent_selection),
        EAStep("crossover", crossover),
        EAStep("mutate", mutate),
        EAStep("evaluate", evaluate),
        EAStep("survivor_select", survivor_selection),
    ]

    ea = EA(initial, ops, num_of_generations=50)
    ea.run()

    console.log("─── Results ───")
    console.log(f"best = {ea.best('best', only_alive=False)}")
    console.log(f"median = {ea.best('median', only_alive=False)}")
    console.log(f"worst = {ea.best('worst', only_alive=False)}")

    # Population API examples
    db_pop = ea._fetch(only_alive=False)
    top_10 = db_pop.best(sort="max", attribute="fitness", n=10)
    console.log(f"top-10 from DB = {top_10}")

    sampled_best = db_pop.sample(30).best(sort="max", attribute="fitness_", n=5)
    console.log(f"sample(30).best(n=5) = {sampled_best}")


if __name__ == "__main__":
    main()
