import pytest
import numpy as np
from scipy.stats import chisquare
from src.evolution.island import boltzmann_sample, migrate, prune, seed, spawn, uniform_sample
from src.evolution.population import Population
from tests.evolution.utils import make_empty_population, make_seeds

def test_seed():
    population = Population() #Must be an empty population
    seed_programs = make_seeds()
    n_islands = 3
    islands = seed(population, seed_programs, n_islands)
    assert len(population) == 2
    assert population[0].name == "Seed1"
    assert population[1].name == "Seed2"
    assert len(islands) == n_islands
    for island in islands:
        assert island == {0, 1}

def test_spawn():
    population = Population()
    seed_programs = make_seeds()
    islands = seed(population, seed_programs, n_islands = 2)
    generation = 0
    spawn(
        population,
        islands,
        generation,
        "test mode",
        1.0,
        batch_size = 1,
        num_parents = 2,
        rng = np.random.default_rng(42)
    )

    #Check population and islands
    assert len(population) == 4
    assert islands[0] == {0,1,2}
    assert islands[1] == {0,1,3}
    isle00 = {0,1,2}
    isle01 = {0,1,3}

    #Check program information
    for i, island in enumerate(islands):
        idx = next(iter(island - {0,1})) # the new program index in this island
        program = population[idx]
        assert program.birth.generation == generation
        assert program.birth.island == i
        assert program.birth.batch_index == 0
        assert program.birth.parent_indices == [0,1]

    generation = 1
    spawn(
        population,
        islands,
        generation,
        "test mode",
        1.0,
        batch_size = 3,
        num_parents = 2,
        rng = np.random.default_rng(42)
    )

    assert len(population) == 10
    assert islands[0] == {0,1,2,4,5,6}
    assert islands[1] == {0,1,3,7,8,9}

    isles = [isle00,isle01]
    expected_parents = [[0,2], [0,1], [0,2], #island0
                        [0,1], [1,3], [1,3]] #island1 #from run with np.random.default_rng(42)
    for i, island in enumerate(islands):
        new_indices = island - isles[i]
        for j,idx in enumerate(sorted(new_indices)):
            program = population[idx]
            assert program.birth.generation == generation
            assert program.birth.island == i
            assert program.birth.batch_index == j
            assert program.birth.parent_indices == expected_parents[i*3 + j]

def test_prune_all_losses_defined():
    losses = (0.5, 100.0, 10, None, 0.6, 7)
    population = make_empty_population(num_programs = len(losses))
    for program, loss in zip(population, losses):
        program.program_losses.discover.final = loss

    islands = [{0,1,2}, {3,4,5}]
    init_programs = list(population._programs)
    prune(islands, population, evolution = {"critical_population_size": 2})
    assert islands[0] == {0,2}
    assert islands[1] == {4,5}
    assert population._programs == init_programs

def test_prune_missing_loss():
    losses = (0.5, 100.0, 10, 1, 0.6) #missing a loss for final program, treated as None so pruned out
    population = make_empty_population(num_programs = 6)
    for i, loss in enumerate(losses):
        population[i].program_losses.discover.final = loss

    islands = [{0,1,2}, {3,4,5}]
    prune(islands, population, evolution = {"critical_population_size": 2})
    assert islands[0] == {0,2}
    assert islands[1] == {3,4}

def test_prune_less_than_critical_num_losses():
    losses = (0.5, 100.0, 10, 0.6) #missing two losses, both treated as None, so one kept as inf
    population = make_empty_population(num_programs = 6)
    for i, loss in enumerate(losses):
        population[i].program_losses.discover.final = loss

    islands = [{0,1,2}, {3,4,5}] #4,5 dont have loss, when we prune we
    prune(islands, population, evolution = {"critical_population_size": 2})
    assert islands[0] == {0,2}
    assert islands[1] == {3,4}

def test_uniform_sample_all_chosen():
    population = make_empty_population(num_programs = 10)
    programs = population._programs
    rng = np.random.default_rng(42)
    samples = uniform_sample(programs, 10, rng)
    assert samples == {p.idx for p in programs}

def test_uniform_sample_too_many():
    population = make_empty_population(num_programs = 5)
    programs = population._programs
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError):
        uniform_sample(programs, 6, rng)

def test_uniform_sample_distribution():
    n, k, N = 6, 2, 10_000
    population = make_empty_population(num_programs=n)
    programs = population._programs
    rng = np.random.default_rng(42)

    counts = np.zeros(n, dtype=int)
    for _ in range(N):
        for idx in uniform_sample(programs, k=k, rng=rng):
            counts[idx] += 1

    _, p_value = chisquare(counts, f_exp=[N * k / n] * n)
    print(p_value)
    assert p_value > 0.01 #fails 1 in 100 times if distribution is uniform, but same seed makes deterministic
 

def test_boltzmann_sample_all_invalid_losses():
    population = make_empty_population(num_programs = 5)
    losses = (None, float("inf"), np.nan, -float("inf"), -np.nan)
    for program, loss in zip(population, losses):
        program.program_losses.discover.final = loss

    programs = population._programs
    rng = np.random.default_rng(42)
    samples = boltzmann_sample(programs, k=5, temperature=1.0, rng=rng)
    assert samples == {0, 1, 2, 3, 4}

def test_boltzmann_sample_highT_isuniform():
    n, k, N = 5, 2, 10_000
    losses = (0.5, 100.0, 10, 1, 0.6)
    population = make_empty_population(num_programs=n)
    for program, loss in zip(population, losses):
        program.program_losses.discover.final = loss

    programs = population._programs
    rng = np.random.default_rng(42)

    counts = np.zeros(n, dtype=int)
    for _ in range(N):
        for idx in boltzmann_sample(programs, k=k, temperature=1e10, rng=rng):
            counts[idx] += 1

    _, p_value = chisquare(counts, f_exp=[N * k / n] * n)
    print(f"p_value: {p_value}")
    assert p_value > 0.01 #1 in 100 times this would fail if distribution is actually uniform, since we use the same seed this is deterministic

def test_boltzmann_sample_lowT_gives_best_losses():
    n, k, N = 5, 2, 10_000
    losses = (0.01, 100.0, 10, 1, 0.6)
    population = make_empty_population(num_programs=n)
    for program, loss in zip(population, losses):
        program.program_losses.discover.final = loss

    programs = population._programs
    rng = np.random.default_rng(42)

    counts = np.zeros(n, dtype=int)
    for _ in range(N):
        for idx in boltzmann_sample(programs, k=k, temperature=1e-3, rng=rng):
            counts[idx] += 1

    # at low temperature, best two programs (idx 0 and 4) should dominate
    best_two_fraction = (counts[0] + counts[4]) / (N * k)
    assert best_two_fraction > 0.99

def test_boltzmann_sample_None_least_sampled():
    n, k, N = 5, 2, 10_000
    losses = (0.01, 100.0, None, 1, 0.6)
    population = make_empty_population(num_programs=n)
    for program, loss in zip(population, losses):
        program.program_losses.discover.final = loss

    programs = population._programs
    rng = np.random.default_rng(42)

    counts = np.zeros(n, dtype=int)
    for _ in range(N):
        for idx in boltzmann_sample(programs, k=k, temperature=1, rng=rng):
            counts[idx] += 1

    assert counts[2] < min(counts[0], counts[1], counts[3], counts[4]) #program with None loss should be least sampled

def test_migration_low_loss_migrated():
    """
        Make islands each with one low loss program and use low temperature.
        Check that migrants are the low loss programs and correctly migrate according to topology.
    """
    population = make_empty_population(num_programs=9)
    losses = (0.001, None, 10, 1, 15, 0.002, 0.9, 0.008, 0.5)
    for program, loss in zip(population, losses):
        program.program_losses.discover.final = loss

    islands = [{0,1,2}, {3,4,5}, {6,7,8}]
    evolution = {"n_migrants": 1, "topology": [1, 2, 0]} #program with idx 0 migrates to island 1, program with idx 5 migrates to island 2, program with idx 7 migrates to island 0, note migration is copying 
    temperature = 1+1e-3 #see temperature warping
    rng = np.random.default_rng(42)
    
    migrate(islands, population, evolution, temperature, rng)
    assert islands[0] == {0,1,2,7}
    assert islands[1] == {0,3,4,5}
    assert islands[2] == {5,6,7,8}

def test_migration_catches_mismatched_topology():
    population = make_empty_population(num_programs=6)
    islands = [{0,1}, {2,3}, {4,5}]
    evolution = {"n_migrants": 1, "topology": [1, 0]} #mismatched topology length
    temperature = 1.0
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        migrate(islands, population, evolution, temperature, rng)