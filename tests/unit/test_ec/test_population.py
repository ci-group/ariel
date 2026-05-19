"""Test: Population class behaviour."""

import pytest

from ariel.ec.individual import Individual
from ariel.ec.population import Population


def _make_ind(fitness: float | None = None, alive: bool = True) -> Individual:
    ind = Individual(alive=alive)
    if fitness is not None:
        ind.fitness = fitness
    return ind


def test_population_len() -> None:
    """len() returns the number of individuals."""
    pop = Population([_make_ind(1.0), _make_ind(2.0)])
    assert len(pop) == 2


def test_population_empty_constructor() -> None:
    """Population.empty() creates a zero-length population."""
    pop = Population.empty()
    assert len(pop) == 0
    assert not pop


def test_population_bool_non_empty() -> None:
    """A non-empty population is truthy."""
    pop = Population([_make_ind()])
    assert pop


def test_population_iter() -> None:
    """Iterating yields all individuals in insertion order."""
    inds = [_make_ind(float(i)) for i in range(3)]
    pop = Population(inds)
    assert list(pop) == inds


def test_population_getitem_int() -> None:
    """Integer indexing returns an Individual."""
    inds = [_make_ind(float(i)) for i in range(3)]
    pop = Population(inds)
    assert pop[0] is inds[0]
    assert pop[2] is inds[2]


def test_population_getitem_slice() -> None:
    """Slice indexing returns a Population."""
    inds = [_make_ind(float(i)) for i in range(4)]
    pop = Population(inds)
    sliced = pop[:2]
    assert isinstance(sliced, Population)
    assert len(sliced) == 2


def test_population_add() -> None:
    """Adding two populations concatenates individuals."""
    p1 = Population([_make_ind(1.0)])
    p2 = Population([_make_ind(2.0)])
    combined = p1 + p2
    assert len(combined) == 2


def test_population_repr() -> None:
    """repr shows the size."""
    pop = Population([_make_ind()])
    assert "Population(n=1)" == repr(pop)


def test_population_append() -> None:
    """append adds one individual in place."""
    pop = Population.empty()
    pop.append(_make_ind(5.0))
    assert len(pop) == 1


def test_population_extend() -> None:
    """extend adds multiple individuals in place."""
    pop = Population([_make_ind(1.0)])
    pop.extend([_make_ind(2.0), _make_ind(3.0)])
    assert len(pop) == 3


def test_population_to_list() -> None:
    """to_list returns a plain Python list."""
    inds = [_make_ind()]
    pop = Population(inds)
    assert isinstance(pop.to_list(), list)
    assert pop.to_list() == inds


def test_population_sample_respects_max() -> None:
    """sample(n) returns at most n individuals."""
    pop = Population([_make_ind(float(i)) for i in range(10)])
    sampled = pop.sample(5)
    assert len(sampled) == 5


def test_population_sample_never_exceeds_size() -> None:
    """sample(n) where n > size returns all individuals."""
    pop = Population([_make_ind() for _ in range(3)])
    sampled = pop.sample(100)
    assert len(sampled) == 3


def test_population_best_returns_top_n() -> None:
    """best(n=2) returns the two highest-fitness individuals."""
    pop = Population([_make_ind(float(i)) for i in range(5)])
    top2 = pop.best(n=2)
    fitnesses = [ind.fitness_ for ind in top2]
    assert fitnesses == sorted(fitnesses, reverse=True)
    assert len(top2) == 2


def test_population_best_min_sort() -> None:
    """best(sort='min') returns the lowest-fitness individuals first."""
    pop = Population([_make_ind(float(i)) for i in range(5)])
    bottom2 = pop.best(sort="min", n=2)
    assert bottom2[0].fitness_ <= bottom2[1].fitness_


def test_population_sort_descending() -> None:
    """sort() returns individuals ordered by fitness descending."""
    pop = Population([_make_ind(float(i)) for i in [3, 1, 4, 1, 5]])
    sorted_pop = pop.sort()
    fitnesses = [ind.fitness_ for ind in sorted_pop]
    assert fitnesses == sorted(fitnesses, reverse=True)


def test_population_shuffle_same_size() -> None:
    """shuffle returns a population of the same size."""
    pop = Population([_make_ind(float(i)) for i in range(6)])
    shuffled = pop.shuffle()
    assert len(shuffled) == len(pop)


def test_population_where_filters() -> None:
    """where keeps only individuals satisfying the predicate."""
    pop = Population([_make_ind(1.0, alive=True), _make_ind(2.0, alive=False)])
    alive = pop.where(lambda ind: ind.alive)
    assert len(alive) == 1
    assert alive[0].alive is True


def test_population_alive_property() -> None:
    """alive property returns only living individuals."""
    pop = Population([_make_ind(alive=True), _make_ind(alive=False)])
    assert len(pop.alive) == 1


def test_population_dead_property() -> None:
    """dead property returns only dead individuals."""
    pop = Population([_make_ind(alive=True), _make_ind(alive=False)])
    assert len(pop.dead) == 1


def test_population_unevaluated_property() -> None:
    """unevaluated returns individuals where requires_eval is True."""
    ind_eval = _make_ind(5.0)
    ind_uneval = Individual()
    pop = Population([ind_eval, ind_uneval])
    assert len(pop.unevaluated) == 1


def test_population_evaluated_property() -> None:
    """evaluated returns individuals where requires_eval is False."""
    ind_eval = _make_ind(5.0)
    ind_uneval = Individual()
    pop = Population([ind_eval, ind_uneval])
    assert len(pop.evaluated) == 1


def test_population_size_property() -> None:
    """size property equals len()."""
    pop = Population([_make_ind() for _ in range(4)])
    assert pop.size == len(pop)


def test_population_chain_api() -> None:
    """Chainable API: alive.sample(10).best(n=1) works end-to-end."""
    inds = [_make_ind(float(i), alive=(i % 2 == 0)) for i in range(10)]
    pop = Population(inds)
    result = pop.alive.sample(10).best(n=1)
    assert len(result) == 1
    assert result[0].alive is True
