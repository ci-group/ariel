"""Test: EASettings, EAOperation, and EA classes."""

from pathlib import Path

import pytest

from ariel.ec.ea import EA, EAOperation, EASettings
from ariel.ec.individual import Individual
from ariel.ec.population import Population


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pop(n: int = 4) -> Population:
    inds = []
    for i in range(n):
        ind = Individual()
        ind.genotype = [float(i)]
        ind.fitness = float(i)
        inds.append(ind)
    return Population(inds)


@EAOperation
def noop(population: Population) -> Population:
    return population


@EAOperation
def double_pop(population: Population) -> Population:
    return Population(population.to_list() + population.to_list())


def _ea(tmp_path: Path, pop: Population | None = None, ops=None, **kwargs) -> EA:
    return EA(
        population=pop or _make_pop(),
        operations=ops or [noop],
        db_file_path=tmp_path / "test.db",
        quiet=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# EASettings
# ---------------------------------------------------------------------------


def test_easettings_defaults() -> None:
    """EASettings initializes with expected defaults."""
    s = EASettings()
    assert s.num_steps == 100
    assert s.target_population_size == 100
    assert s.db_file_name == "database.db"
    assert s.db_handling == "delete"
    assert s.is_maximisation is True
    assert s.first_generation_id == 0
    assert s.quiet is False


def test_easettings_db_file_path() -> None:
    """db_file_path computed field combines output_folder and db_file_name."""
    s = EASettings(output_folder=Path("/tmp/ea"), db_file_name="run.db")
    assert s.db_file_path == Path("/tmp/ea/run.db")


def test_easettings_override() -> None:
    """EASettings fields can be overridden at construction."""
    s = EASettings(num_steps=50, quiet=True)
    assert s.num_steps == 50
    assert s.quiet is True


# ---------------------------------------------------------------------------
# EAOperation — decorator validation
# ---------------------------------------------------------------------------


def test_eaoperation_valid_decoration() -> None:
    """@EAOperation on a valid function returns an EAOperation."""
    assert isinstance(noop, EAOperation)


def test_eaoperation_no_params_raises() -> None:
    """@EAOperation on a zero-parameter function raises TypeError."""
    with pytest.raises(TypeError, match="first argument"):
        @EAOperation
        def bad() -> Population:  # no params
            return Population([])


def test_eaoperation_wrong_first_param_type_raises() -> None:
    """First parameter not annotated as Population raises TypeError."""
    with pytest.raises(TypeError, match="Population"):
        @EAOperation
        def bad(x: int) -> Population:
            return Population([])


def test_eaoperation_wrong_return_type_raises() -> None:
    """Return annotation not Population raises TypeError."""
    with pytest.raises(TypeError, match="Population"):
        @EAOperation
        def bad(population: Population) -> list:
            return []


# ---------------------------------------------------------------------------
# EAOperation — __call__ with Population (execute)
# ---------------------------------------------------------------------------


def test_eaoperation_call_with_population_executes() -> None:
    """Calling an EAOperation with a Population executes and returns Population."""
    pop = _make_pop()
    result = noop(pop)
    assert isinstance(result, Population)
    assert len(result) == len(pop)


def test_eaoperation_call_result_is_population() -> None:
    """double_pop operation doubles the population size."""
    pop = _make_pop(3)
    result = double_pop(pop)
    assert len(result) == 6


# ---------------------------------------------------------------------------
# EAOperation — __call__ with config args (bind)
# ---------------------------------------------------------------------------


def test_eaoperation_call_with_kwargs_returns_bound_op() -> None:
    """Calling with kwargs returns a new bound EAOperation (not executing)."""
    @EAOperation
    def with_param(population: Population, factor: float = 1.0) -> Population:
        return population

    bound = with_param(factor=2.0)
    assert isinstance(bound, EAOperation)
    assert bound.kwargs == {"factor": 2.0}


def test_eaoperation_bound_executes_with_population() -> None:
    """A bound EAOperation executes correctly when called with a Population."""
    results = []

    @EAOperation
    def capture(population: Population, tag: str = "x") -> Population:
        results.append(tag)
        return population

    op = capture(tag="hello")
    pop = _make_pop()
    op(pop)
    assert results == ["hello"]


# ---------------------------------------------------------------------------
# EAOperation — __repr__
# ---------------------------------------------------------------------------


def test_eaoperation_repr() -> None:
    """__repr__ includes the function name."""
    assert "noop" in repr(noop)


def test_eaoperation_repr_with_kwargs() -> None:
    """__repr__ includes bound kwargs."""
    @EAOperation
    def op(population: Population, k: int = 1) -> Population:
        return population

    bound = op(k=42)
    assert "42" in repr(bound)


# ---------------------------------------------------------------------------
# EA — initialization
# ---------------------------------------------------------------------------


def test_ea_initializes(tmp_path) -> None:
    """EA initializes without error given population and ops."""
    ea = _ea(tmp_path)
    assert ea.current_generation == 0


def test_ea_no_operations_raises(tmp_path) -> None:
    """EA raises ValueError when operations is None."""
    with pytest.raises(ValueError, match="operations"):
        EA(population=_make_pop(), operations=None,
           db_file_path=tmp_path / "db.db", quiet=True)


def test_ea_no_population_no_restart_raises(tmp_path) -> None:
    """EA raises ValueError when neither population nor restart is given."""
    with pytest.raises(ValueError, match="population"):
        EA(population=None, operations=[noop],
           db_file_path=tmp_path / "db.db", quiet=True)


def test_ea_creates_database_file(tmp_path) -> None:
    """EA creates the SQLite database file on disk."""
    db = tmp_path / "ea.db"
    _ea(tmp_path, ops=[noop])
    assert db.exists() or (tmp_path / "test.db").exists()


def test_ea_db_handling_halt_raises(tmp_path) -> None:
    """db_handling='halt' raises FileExistsError when db already exists."""
    db = tmp_path / "existing.db"
    db.touch()
    with pytest.raises(FileExistsError):
        EA(population=_make_pop(), operations=[noop],
           db_file_path=db, quiet=True, db_handling="halt")


def test_ea_db_handling_delete_overwrites(tmp_path) -> None:
    """db_handling='delete' recreates the db without raising."""
    db = tmp_path / "run.db"
    ea1 = EA(population=_make_pop(), operations=[noop],
             db_file_path=db, quiet=True, db_handling="delete")
    # Explicitly dispose the engine so Windows releases the file lock
    ea1.engine.dispose()
    del ea1
    # Second run on same path — should succeed
    EA(population=_make_pop(), operations=[noop],
       db_file_path=db, quiet=True, db_handling="delete")


def test_ea_population_stored(tmp_path) -> None:
    """EA stores the initial population."""
    pop = _make_pop(5)
    ea = _ea(tmp_path, pop=pop)
    assert len(ea.population) == 5


def test_ea_custom_first_generation(tmp_path) -> None:
    """first_generation_id is stored on the EA."""
    ea = _ea(tmp_path, first_generation_id=10)
    assert ea.current_generation == 10


# ---------------------------------------------------------------------------
# EA — _commit and _fetch
# ---------------------------------------------------------------------------


def test_ea_commit_sets_time_of_birth(tmp_path) -> None:
    """_commit assigns time_of_birth=-1 individuals the current generation."""
    ea = _ea(tmp_path)
    # Fetch fresh from DB so SQLAlchemy session context is active
    pop = ea._fetch()
    for ind in pop:
        assert ind.time_of_birth == 0


def test_ea_fetch_returns_population(tmp_path) -> None:
    """_fetch returns a Population instance."""
    ea = _ea(tmp_path)
    pop = ea._fetch()
    assert isinstance(pop, Population)


def test_ea_fetch_only_alive(tmp_path) -> None:
    """_fetch(only_alive=True) returns only alive individuals."""
    ea = _ea(tmp_path)
    pop = ea._fetch(only_alive=True)
    assert all(ind.alive for ind in pop)


def test_ea_fetch_sort_desc(tmp_path) -> None:
    """_fetch(sort='desc') returns individuals sorted by fitness descending."""
    ea = _ea(tmp_path)
    pop = ea._fetch(sort="desc")
    fitnesses = [ind.fitness_ for ind in pop if ind.fitness_ is not None]
    assert fitnesses == sorted(fitnesses, reverse=True)


def test_ea_fetch_sort_asc(tmp_path) -> None:
    """_fetch(sort='asc') returns individuals sorted by fitness ascending."""
    ea = _ea(tmp_path)
    pop = ea._fetch(sort="asc")
    fitnesses = [ind.fitness_ for ind in pop if ind.fitness_ is not None]
    assert fitnesses == sorted(fitnesses)


def test_ea_fetch_requires_eval_false(tmp_path) -> None:
    """_fetch(requires_eval=False) returns only evaluated individuals."""
    ea = _ea(tmp_path)
    pop = ea._fetch(requires_eval=False)
    assert all(not ind.requires_eval for ind in pop)


# ---------------------------------------------------------------------------
# EA — fetch_population
# ---------------------------------------------------------------------------


def test_ea_fetch_population_updates_self(tmp_path) -> None:
    """fetch_population() updates self.population from the database."""
    ea = _ea(tmp_path)
    original_len = len(ea.population)
    ea.fetch_population()
    assert len(ea.population) == original_len


# ---------------------------------------------------------------------------
# EA — size property
# ---------------------------------------------------------------------------


def test_ea_size_property(tmp_path) -> None:
    """size returns the number of alive individuals."""
    ea = _ea(tmp_path, pop=_make_pop(6))
    assert ea.size == 6


# ---------------------------------------------------------------------------
# EA — get_solution
# ---------------------------------------------------------------------------


def test_ea_get_solution_best(tmp_path) -> None:
    """get_solution('best') returns an Individual."""
    ea = _ea(tmp_path)
    sol = ea.get_solution(mode="best")
    assert isinstance(sol, Individual)


def test_ea_get_solution_median(tmp_path) -> None:
    """get_solution('median') returns an Individual."""
    ea = _ea(tmp_path)
    sol = ea.get_solution(mode="median")
    assert isinstance(sol, Individual)


def test_ea_get_solution_worst(tmp_path) -> None:
    """get_solution('worst') returns an Individual."""
    ea = _ea(tmp_path)
    sol = ea.get_solution(mode="worst")
    assert isinstance(sol, Individual)


# ---------------------------------------------------------------------------
# EA — step
# ---------------------------------------------------------------------------


def test_ea_step_increments_generation(tmp_path) -> None:
    """step() increments current_generation by 1."""
    ea = _ea(tmp_path)
    ea.step()
    assert ea.current_generation == 1


def test_ea_step_applies_operations(tmp_path) -> None:
    """step() applies all operations and the result is committed to the DB."""
    called = []

    @EAOperation
    def recording_op(population: Population) -> Population:
        called.append(True)
        return population

    ea = _ea(tmp_path, pop=_make_pop(3), ops=[recording_op])
    ea.step()
    assert len(called) == 1


def test_ea_step_twice(tmp_path) -> None:
    """Two steps advance generation to 2."""
    ea = _ea(tmp_path)
    ea.step()
    ea.step()
    assert ea.current_generation == 2


# ---------------------------------------------------------------------------
# EA — run
# ---------------------------------------------------------------------------


def test_ea_run_executes_num_steps(tmp_path) -> None:
    """run() executes exactly num_steps generations."""
    ea = _ea(tmp_path, num_steps=5)
    ea.run()
    assert ea.current_generation == 5


def test_ea_run_zero_steps(tmp_path) -> None:
    """run() with num_steps=0 leaves generation unchanged."""
    ea = _ea(tmp_path, num_steps=0)
    ea.run()
    assert ea.current_generation == 0
