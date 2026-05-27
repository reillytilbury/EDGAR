import pytest
import numpy as np
from scipy.stats import chisquare
from src.evolution.island import boltzmann_sample, migrate, prune, seed, spawn, uniform_sample, _are_duplicates, deduplicate_inner, deduplicate_outer, deduplicate
from src.evolution.population import Population
from tests.evolution.utils import make_empty_population, make_seeds, make_fingerprint_population, make_fingerprint_program

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
    prune(islands, population, evolution = {"critical_population_size": 3, "n_migrants": 1})
    assert islands[0] == {0,2}
    assert islands[1] == {4,5}
    assert population._programs == init_programs

def test_prune_missing_loss():
    losses = (0.5, 100.0, 10, 1, 0.6) #missing a loss for final program, treated as None so pruned out
    population = make_empty_population(num_programs = 6)
    for i, loss in enumerate(losses):
        population[i].program_losses.discover.final = loss

    islands = [{0,1,2}, {3,4,5}]
    prune(islands, population, evolution = {"critical_population_size": 3, "n_migrants": 1})
    assert islands[0] == {0,2}
    assert islands[1] == {3,4}

def test_prune_less_than_critical_num_losses():
    losses = (0.5, 100.0, 10, 0.6) #missing two losses, both treated as None, so one kept as inf
    population = make_empty_population(num_programs = 6)
    for i, loss in enumerate(losses):
        population[i].program_losses.discover.final = loss

    islands = [{0,1,2}, {3,4,5}] #4,5 dont have loss, when we prune we
    prune(islands, population, evolution = {"critical_population_size": 3, "n_migrants": 1})
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


# ─────────────────────────────────────────────────────────────────────────
# Deduplication helpers
# ─────────────────────────────────────────────────────────────────────────

# Orthogonal unit vectors for fingerprint construction
_E0 = [1.0, 0.0, 0.0, 0.0]
_E1 = [0.0, 1.0, 0.0, 0.0]
_E2 = [0.0, 0.0, 1.0, 0.0]
_E3 = [0.0, 0.0, 0.0, 1.0]
_E0_NEAR = [1.0, 0.01, 0.0, 0.0]  # cosine ~0.9999 with _E0


# ─────────────────────────────────────────────────────────────────────────
# _are_duplicates
# ─────────────────────────────────────────────────────────────────────────

def test_are_duplicates_same_fingerprint():
    p0 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    assert _are_duplicates(p0, p1, loss_tol=0.1, cosine_tol=0.95)


def test_are_duplicates_orthogonal_fingerprints():
    p0 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1 = make_fingerprint_program(_E1, loss=1.0, n_params=2)
    assert not _are_duplicates(p0, p1, loss_tol=0.1, cosine_tol=0.95)


def test_are_duplicates_different_n_params():
    p0 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1 = make_fingerprint_program(_E0, loss=1.0, n_params=3)
    assert not _are_duplicates(p0, p1, loss_tol=0.1, cosine_tol=0.95)


def test_are_duplicates_none_n_params():
    p0 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1.n_params = None
    assert not _are_duplicates(p0, p1, loss_tol=0.1, cosine_tol=0.95)


def test_are_duplicates_none_fingerprint():
    p0 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1.eval_fingerprint = None
    assert not _are_duplicates(p0, p1, loss_tol=0.1, cosine_tol=0.95)


def test_are_duplicates_loss_diff_exceeds_tol():
    p0 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1 = make_fingerprint_program(_E0_NEAR, loss=2.0, n_params=2)  # diff = 1.0 > 0.1
    assert not _are_duplicates(p0, p1, loss_tol=0.1, cosine_tol=0.95)


def test_are_duplicates_none_loss_skips_loss_check():
    """When either loss is None, the loss check is skipped and only fingerprint is compared."""
    p0 = make_fingerprint_program(_E0, loss=None, n_params=2)
    p1 = make_fingerprint_program(_E0, loss=5.0, n_params=2)
    assert _are_duplicates(p0, p1, loss_tol=0.01, cosine_tol=0.95)


def test_are_duplicates_near_identical_fingerprint():
    p0 = make_fingerprint_program(_E0, loss=1.0, n_params=2)
    p1 = make_fingerprint_program(_E0_NEAR, loss=1.0, n_params=2)
    assert _are_duplicates(p0, p1, loss_tol=0.1, cosine_tol=0.95)


# ─────────────────────────────────────────────────────────────────────────
# deduplicate_inner
# ─────────────────────────────────────────────────────────────────────────

def test_deduplicate_inner_removes_higher_loss_duplicate():
    pop = make_fingerprint_population([
        (_E0, 1.0),   # idx 0 — lower loss, kept
        (_E0, 1.005), # idx 1 — higher loss, within tol, removed
        (_E1, 1.0),   # idx 2 — different fingerprint, kept
    ])
    islands = [{0, 1, 2}]
    deduplicate_inner(islands, pop)
    assert islands[0] == {0, 2}


def test_deduplicate_inner_keeps_both_when_not_duplicates():
    pop = make_fingerprint_population([
        (_E0, 0.5),  # idx 0
        (_E1, 2.0),  # idx 1 — orthogonal
    ])
    islands = [{0, 1}]
    deduplicate_inner(islands, pop)
    assert islands[0] == {0, 1}


def test_deduplicate_inner_no_fingerprint_not_deduplicated():
    pop = make_fingerprint_population([
        (_E0, 0.5),
        (_E0, 0.5),
    ])
    pop[1].eval_fingerprint = None
    islands = [{0, 1}]
    deduplicate_inner(islands, pop)
    assert islands[0] == {0, 1}


def test_deduplicate_inner_independent_per_island():
    """Duplicates in island 0 are removed; island 1 is unaffected."""
    pop = make_fingerprint_population([
        (_E0, 1.0),    # idx 0 — island 0 low loss
        (_E0, 1.005),  # idx 1 — island 0 dup, removed
        (_E1, 1.0),    # idx 2 — island 1
        (_E2, 3.0),    # idx 3 — island 1
    ])
    islands = [{0, 1}, {2, 3}]
    deduplicate_inner(islands, pop)
    assert islands[0] == {0}
    assert islands[1] == {2, 3}


def test_deduplicate_inner_chain_keeps_best():
    """A dup of B, B dup of C: once B is flagged for removal, A vs C is still checked."""
    pop = make_fingerprint_population([
        (_E0, 1.0),        # idx 0 — best
        (_E0_NEAR, 1.005), # idx 1 — dup of 0, removed
        (_E0, 1.008),      # idx 2 — dup of 0 (and 1), removed
    ])
    islands = [{0, 1, 2}]
    deduplicate_inner(islands, pop)
    assert islands[0] == {0}


# ─────────────────────────────────────────────────────────────────────────
# deduplicate_outer
# ─────────────────────────────────────────────────────────────────────────

def make_islands_with_duplicates():
    """Two islands of 4 programs, 3 fingerprint-duplicate pairs between them."""
    # Pairs (0,4), (1,5), (2,6) are duplicates (losses within 0.005); (3,7) are not
    # Island A has lower losses than island B
    pop = make_fingerprint_population([
        (_E0, 0.100),     # 0 — island A  dup of 4
        (_E1, 0.500),     # 1 — island A  dup of 5
        (_E2, 1.000),     # 2 — island A  dup of 6
        (_E3, 2.000),     # 3 — island A  unique
        (_E0, 0.105),     # 4 — island B  dup of 0
        (_E1, 0.505),     # 5 — island B  dup of 1
        (_E2, 1.005),     # 6 — island B  dup of 2
        (_E0_NEAR, 3.0),  # 7 — island B  unique
    ])
    return pop, [{0, 1, 2, 3}, {4, 5, 6, 7}]


def test_deduplicate_outer_removes_higher_loss_from_j():
    """Duplicate pair where island j has higher loss: j's program is removed, i untouched."""
    pop = make_fingerprint_population([
        (_E0, 1.000),  # 0 — island i, lower loss
        (_E1, 2.000),  # 1 — island i, unique
        (_E0, 1.005),  # 2 — island j, dup of 0, higher loss → removed
        (_E2, 3.000),  # 3 — island j, unique
    ])
    islands = [{0, 1}, {2, 3}]
    deduplicate_outer(islands, pop, min_island_size=2)
    assert islands[0] == {0, 1}
    assert islands[1] == {3}


def test_deduplicate_outer_removes_higher_loss_from_i():
    """Duplicate pair where island i has higher loss: i's program is removed, j untouched."""
    pop = make_fingerprint_population([
        (_E0, 1.005),  # 0 — island i, dup of 2, higher loss → removed
        (_E1, 2.000),  # 1 — island i, unique
        (_E0, 1.000),  # 2 — island j, lower loss
        (_E2, 3.000),  # 3 — island j, unique
    ])
    islands = [{0, 1}, {2, 3}]
    deduplicate_outer(islands, pop, min_island_size=2)
    assert islands[0] == {1}
    assert islands[1] == {2, 3}


def test_deduplicate_outer_tie_keeps_lower_island_index():
    """Equal losses: program in island j (higher index) is removed."""
    pop = make_fingerprint_population([
        (_E0, 1.000),  # 0 — island i
        (_E1, 2.000),  # 1 — island i, unique
        (_E0, 1.000),  # 2 — island j, exact same loss as 0 → j loses tie
        (_E2, 3.000),  # 3 — island j, unique
    ])
    islands = [{0, 1}, {2, 3}]
    deduplicate_outer(islands, pop, min_island_size=2)
    assert islands[0] == {0, 1}
    assert islands[1] == {3}


def test_deduplicate_outer_skips_pair_below_min_island_size():
    """Island smaller than min_island_size: no removals."""
    pop = make_fingerprint_population([
        (_E0, 1.000),  # 0 — island i
        (_E0, 1.005),  # 1 — island j, dup of 0
    ])
    islands = [{0}, {1}]
    deduplicate_outer(islands, pop, min_island_size=2)
    assert islands[0] == {0}
    assert islands[1] == {1}


def test_deduplicate_outer_removes_multiple_duplicates():
    """Three duplicate pairs across islands: all three higher-loss copies are removed."""
    pop, islands = make_islands_with_duplicates()
    deduplicate_outer(islands, pop, min_island_size=4)
    assert islands[0] == {0, 1, 2, 3}
    assert islands[1] == {7}  # 4, 5, 6 removed as higher-loss dups


def test_deduplicate_outer_non_duplicates_untouched():
    """Programs with orthogonal fingerprints are never removed."""
    pop = make_fingerprint_population([
        (_E0, 1.0),  # 0 — island i
        (_E1, 1.0),  # 1 — island i
        (_E2, 1.0),  # 2 — island j
        (_E3, 1.0),  # 3 — island j
    ])
    islands = [{0, 1}, {2, 3}]
    deduplicate_outer(islands, pop, min_island_size=2)
    assert islands[0] == {0, 1}
    assert islands[1] == {2, 3}


def test_deduplicate_outer_three_islands():
    """With three islands, only the overlapping pair has duplicates removed; third island untouched."""
    pop = make_fingerprint_population([
        (_E0, 0.100),  # 0 — island 0
        (_E1, 2.000),  # 1 — island 0, unique
        (_E0, 0.105),  # 2 — island 1, dup of 0, higher loss → removed
        (_E2, 3.000),  # 3 — island 1, unique
        (_E3, 4.000),  # 4 — island 2, distinct
        (_E1, 5.000),  # 5 — island 2, distinct
    ])
    islands = [{0, 1}, {2, 3}, {4, 5}]
    deduplicate_outer(islands, pop, min_island_size=2)
    assert islands[0] == {0, 1}
    assert islands[1] == {3}
    assert islands[2] == {4, 5}


# ─────────────────────────────────────────────────────────────────────────
# deduplicate (integration)
# ─────────────────────────────────────────────────────────────────────────

def test_deduplicate_applies_inner_then_outer():
    """Inner removes within-island dup; outer then removes the cross-island dup."""
    pop = make_fingerprint_population([
        (_E0, 1.000),  # 0 — island 0 best
        (_E0, 1.005),  # 1 — island 0 dup of 0, removed by inner
        (_E1, 2.000),  # 2 — island 0, distinct
        (_E0, 1.003),  # 3 — island 1, dup of 0, higher loss → removed by outer
        (_E1, 2.005),  # 4 — island 1, dup of 2, higher loss → removed by outer
        (_E2, 3.000),  # 5 — island 1, distinct
    ])
    islands = [{0, 1, 2}, {3, 4, 5}]
    evolution = {"critical_population_size": 2}
    deduplicate(islands, pop, evolution)
    # inner: island 0 → {0, 2} (1 removed); island 1 unchanged → {3, 4, 5}
    # outer: min_island_size=1.5; island 0 size=2, island 1 size=3 → both pass
    #   0 dup of 3 (losses 1.0 vs 1.003) → 3 removed from island 1
    #   2 dup of 4 (losses 2.0 vs 2.005) → 4 removed from island 1
    assert islands[0] == {0, 2}
    assert islands[1] == {5}


# ─────────────────────────────────────────────────────────────────────────
# Census save / load
# ─────────────────────────────────────────────────────────────────────────

def test_island_census_round_trip(tmp_path):
    census = [
        [{0, 1}, {0, 1, 2}, {0, 1, 2, 4}],
        [{0, 1}, {0, 1, 3}, {0, 1, 3, 5}],
    ]
    path = str(tmp_path / "census.json")
    from src.evolution.island import save_island_census, load_island_census
    save_island_census(census, path)
    loaded = load_island_census(path)
    assert loaded == census