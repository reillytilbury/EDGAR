"""
island.py

Operations on islands (plain set[int] of global Population indices) and
island census persistence.

Single-island operations (prune, sample, deduplicate) take a set of Program
objects resolved from the island and return a new set of global indices.
Cross-island operation (deduplicate_islands) takes two program sets and resets
the worse island to the two seed programs {0, 1} if the islands are duplicates.

island_census tracks each island's membership at the end of every generation:

    census[island_id][generation] -> set[int]

Example usage:
--------------
    programs_0 = {popn[i] for i in island_0}
    programs_1 = {popn[i] for i in island_1}

    # prune island_0 to 10 best programs
    island_0 = prune(programs_0, keep_n=10)              # set[int]

    # sample 2 parents via Boltzmann distribution
    parents = boltzmann_sample(programs_0, k=2, temperature=1.0)   # set[int]

    # migration — union with a sample from another island
    island_1 = island_1 | uniform_sample(programs_0, k=2)

    # within-island deduplication
    island_0 = deduplicate(programs_0)                   # set[int]

    # TODO: describe old cross-island deduplication 

    # append current island state to census each generation
    for island_id, island in enumerate(islands):
        census[island_id].append(set(island))
    save_island_census(census, "census.json")
"""

from __future__ import annotations
import json
import numpy as np
from itertools import combinations
from .program import Program, BirthCertificate
from .population import Population


# ─────────────────────────────────────────────────────────────────────────
# Initialization: seed and spawn
# ─────────────────────────────────────────────────────────────────────────

def seed(population: Population, seed_programs: list[Program], n_islands: int) -> list[set[int]]:
    """
    Add seed programs to population and initialize islands.

    Mutates: population (adds seed programs)
    Returns: islands — list of n_islands sets, each containing all seed indices
    """
    assert len(population) == 0, "Initial population must be empty"
    for program in seed_programs:
        population.add(program)

    seed_indices = {p.idx for p in seed_programs}
    return [set(seed_indices) for _ in range(n_islands)]

def spawn(
    population: Population,
    islands: list[set[int]],
    generation: int,
    mode: str,
    temperature: float,
    batch_size: int,
    num_parents: int,
    rng: np.random.Generator,
) -> None:
    """
    Sample parents from each island and create empty Program shells.
    Adds shells to population and their birth island.

    Each shell gets a BirthCertificate record (generation, island, batch_index, parents,
    mode, temperature) but no code yet.

    Mutates: population (adds shells), islands (adds new indices)
    """
    for island_idx, island in enumerate(islands):
        programs = [population[i] for i in island]

        for batch_idx in range(batch_size):
            parent_indices = list(uniform_sample(programs, k=num_parents, rng=rng))
            child = Program(
                birth=BirthCertificate(
                    generation=generation,
                    island=island_idx,
                    batch_index=batch_idx,
                    mode=mode,
                    temperature=temperature,
                    parent_indices=parent_indices,
                ),
            )
            population.add(child)
            island.add(child.idx)


# ─────────────────────────────────────────────────────────────────────────
# Sampling and pruning
# ─────────────────────────────────────────────────────────────────────────

def prune(islands: list[set[int]], population: Population, evolution: dict) -> None:
    """Prune each island to critical_population_size - n_migrants best programs, mutating islands in-place.
    This ensures that after migration, each island has at most critical_population_size programs

    Args:
        islands: list of island sets (each set contains program indices)
        population: Population object to resolve indices to Programs
        evolution: evolution config dict containing critical_population_size
    """
    keep_n = evolution["critical_population_size"] - evolution["n_migrants"]
    for i, island in enumerate(islands):
        programs = [population[idx] for idx in island]
        # explicitly handle cases where the loss is None or inf to avoid sorting issues
        ranked = sorted(programs, key=lambda p: float('inf') if p.program_losses.discover.final is None else p.program_losses.discover.final)
        islands[i] = {p.idx for p in ranked[:keep_n]}


def uniform_sample(programs: list[Program], k: int, rng: np.random.Generator) -> set[int]:
    """
    Sample k global indices uniformly at random.

        programs_0 = [pop[i] for i in island_0]
        parents    = uniform_sample(programs_0, k=2)  # e.g. {0, 2}
    """
    if k > len(programs):
        raise ValueError(f"k={k} exceeds number of programs sampled from {len(programs)}")
    chosen = rng.choice(len(programs), size=k, replace=False)
    return {programs[j].idx for j in chosen}


def boltzmann_sample(programs: list[Program], k: int, temperature: float, rng: np.random.Generator) -> set[int]:
    r"""
    Sample k global indices using a Boltzmann distribution over relative,
    std-normalised losses. High temperature → uniform, low → best dominate.
    Programs are sampled from the prob. distribution:
    Math:
        P_i = exp(g_i) / sum_j(exp(g_j)),
        g_i = -z_i / max(temperature, 1e-3),
        z_i = (loss_i - loss_min) / (std_loss + 1e-6)

        programs_0 = [pop[i] for i in island_0]
        parents    = boltzmann_sample(programs_0, k=2, temperature=1.0)  # e.g. {0, 2}
    """
    if k > len(programs):
        raise ValueError(f"k={k} exceeds number of programs sampled from {len(programs)}")
    losses = np.array([p.program_losses.discover.final for p in programs], dtype=float) #dtype = float converts None to nan
    losses = np.where(np.isnan(losses) | np.isinf(losses), float("inf"), losses) #convert NaN, +-inf to +inf
    finite_losses = losses[np.isfinite(losses)]
    worst_finite = finite_losses.max() if len(finite_losses) > 0 else 0.0
    losses = np.where(np.isinf(losses), worst_finite + 1.0, losses) #convert +inf to worst_finite + 1
    logits = -(losses - losses.min()) / (np.std(losses) + 1e-6) / max(temperature, 1e-3) #compute g_i logits
    logits -= logits.max() #ensures largest value being exponentiated is 0, so exp(g_i) is in [0, 1]
    probs = np.exp(logits)
    probs /= probs.sum()
    chosen = rng.choice(len(programs), size=k, replace=False, p=probs)
    return {programs[j].idx for j in chosen}


# ─────────────────────────────────────────────────────────────────────────
# Migration
# ─────────────────────────────────────────────────────────────────────────

def migrate(islands: list[set[int]], population: Population, evolution: dict, temperature: float, rng: np.random.Generator) -> None:
    """Sample migrants from each island via Boltzmann distribution and add to topology destination.

    For each island i, sample n_migrants programs using Boltzmann distribution (biased toward
    better programs) and add them to the destination island specified by topology[i].
    Mutates islands in-place.

    Args:
        islands: list of island sets (each set contains program indices)
        population: Population object to resolve indices to Programs
        evolution: evolution config dict containing n_migrants and topology
        temperature: temperature for Boltzmann sampling
    """
    # Temperature needs warping before being passed to boltzmann_sample.
    # Raw temperature from schedule() lives in [1, 2]. The correct transform is
    #     T_warped = (T - 1.0) ** 4
    # which maps [1, 2] -> [0, 1] with a sharp decay, so migration becomes
    # strongly selective late in the run. Confirmed in the old hypothesis_engine.py.

    if len(islands) != len(evolution["topology"]):
        raise ValueError("Length of topology must match number of islands")

    n_migrants = evolution["n_migrants"]
    topology = evolution["topology"]
    T_warped = (temperature - 1.0) ** 4

    #First collect which programs will be migrated
    updates = []
    for island in islands:
        programs = [population[idx] for idx in island]
        sampled = boltzmann_sample(programs, k=n_migrants, temperature=T_warped, rng=rng)
        updates.append(sampled)

    # Copy the migrants across to destination islands, updating all islands simultaneously
    for destination, migrants in zip(topology, updates):
        islands[destination].update(migrants)

# ─────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────

def _loss(p: Program) -> float:
    v = p.program_losses.discover.final
    return v if v is not None else float("inf")

def _are_duplicates(p_i: Program, p_j: Program, loss_tol: float, cosine_tol: float) -> bool:
    if p_i.n_params is None or p_j.n_params is None or p_i.n_params != p_j.n_params:
        return False
    if p_i.eval_fingerprint is None or p_j.eval_fingerprint is None:
        return False
    if p_i.program_losses.discover.final is not None and p_j.program_losses.discover.final is not None:
        if abs(p_i.program_losses.discover.final - p_j.program_losses.discover.final) > loss_tol:
            return False
    y_i = p_i.eval_fingerprint.flatten()
    y_j = p_j.eval_fingerprint.flatten()
    cosine = np.dot(y_i, y_j) / (np.linalg.norm(y_i) * np.linalg.norm(y_j) + 1e-6)
    return bool(cosine >= cosine_tol)


def deduplicate_inner(islands: list[set[int]], population: Population, loss_tol: float = 0.01, cosine_tol: float = 0.95) -> None:
    """Remove near-duplicate programs within each island, mutating islands in-place.

    Two programs are considered duplicates if they pass all three checks:
        1. same number of parameters
        2. losses within loss_tol of each other
        3. cosine similarity of eval fingerprints >= cosine_tol

    For each duplicate pair, keeps the lower loss copy.

    Args:
        islands: list of island sets (each set contains program indices)
        population: Population object to resolve indices to Programs
        loss_tol: loss tolerance for duplicate detection
        cosine_tol: cosine similarity tolerance for duplicate detection
    """
    for i, island in enumerate(islands):
        programs = [population[idx] for idx in island]
        to_remove = set()

        for p_j, p_k in combinations(programs, 2):
            if p_j.idx in to_remove or p_k.idx in to_remove:
                continue
            if not _are_duplicates(p_j, p_k, loss_tol, cosine_tol):
                continue
            loser = p_j if p_j.program_losses.discover.final >= p_k.program_losses.discover.final else p_k
            to_remove.add(loser.idx)

        islands[i] = {p.idx for p in programs if p.idx not in to_remove}


def deduplicate_outer(islands: list[set[int]], population: Population, min_island_size: int = 6, loss_tol: float = 0.01, cosine_tol: float = 0.99) -> None:
    """Remove cross-island duplicate programs, keeping the lower-loss copy. Note: assumes deduplicate_inner has already been applied to ensure at most one duplicate of a program on the other island.

    For each pair of islands, skips the pair if either has fewer than min_island_size programs.
    Otherwise iterates over all program pairs and removes the higher-loss duplicate
    immediately upon detection. Ties are broken by keeping the program in the
    lower-indexed island. Mutates islands in-place.

    Args:
        islands: list of island sets (each set contains program indices)
        population: Population object to resolve indices to Programs
        min_island_size: minimum island size required to attempt cross-island deduplication
        loss_tol: loss tolerance for duplicate detection
        cosine_tol: cosine similarity tolerance for duplicate detection
    """
    islands = [island for island in islands if len(island) >= min_island_size]

    for i, j in combinations(range(len(islands)), 2):
        snapshot_i = list(islands[i])
        snapshot_j = list(islands[j])

        for idx_j in snapshot_j:
            if idx_j not in islands[j]:
                continue
            p_j = population[idx_j]

            for idx_i in snapshot_i:
                if idx_i not in islands[i]:
                    continue
                p_i = population[idx_i]

                if not _are_duplicates(p_i, p_j, loss_tol, cosine_tol):
                    continue

                if _loss(p_i) < _loss(p_j):
                    loser_island, loser_idx = j, idx_j
                elif _loss(p_i) > _loss(p_j):
                    loser_island, loser_idx = i, idx_i
                else:
                    loser_island, loser_idx = j, idx_j  # tie: keep lower island index

                islands[loser_island].remove(loser_idx)
                break  # deduplicate_inner guarantees at most one duplicate of a program on the other island (each island has unique programs)


def deduplicate(islands: list[set[int]], population: Population, evolution: dict, loss_tol: float = 0.01, cosine_tol: float = 0.95) -> None:
    """Apply within-island and between-island deduplication, mutating islands in-place.

    First removes duplicates within each island, then checks adjacent islands for
    cross-island duplicates and resets worse islands if needed.

    Args:
        islands: list of island sets (each set contains program indices)
        population: Population object to resolve indices to Programs
        evolution: evolution config dict containing loss_tol, cosine_tol, min_island_size
    """
    n_critical = evolution.get("critical_population_size", 12)
    min_island_size = 0.75*n_critical # only remove cross-island duplicates if both islands have at least 75% of critical population size

    deduplicate_inner(islands, population, loss_tol, cosine_tol)
    deduplicate_outer(islands, population, min_island_size, loss_tol, cosine_tol)


# ─────────────────────────────────────────────────────────────────────────
# Census: save and load island membership history
# ─────────────────────────────────────────────────────────────────────────

def save_island_census(census: list[list[set[int]]], path: str) -> None:
    """
    Save island_census to JSON.
    census[island_id][generation] is the set of program indices at end of that generation.
    """
    with open(path, "w") as f:
        json.dump([[list(s) for s in island] for island in census], f)


def load_island_census(path: str) -> list[list[set[int]]]:
    with open(path) as f:
        data = json.load(f)
    return [[set(s) for s in island] for island in data]
