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

    # cross-island: if islands are too similar, reset the worse one to seed programs
    island_0, island_1 = deduplicate_islands(programs_0, programs_1, n_overlap=6)

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
    k_max: int,
) -> None:
    """
    Sample parents from each island and create empty Program shells.
    Adds shells to population and their birth island.

    Each shell gets a BirthCertificate record (generation, island, batch_index, parents,
    mode, temperature) but no code yet.

    Mutates: population (adds shells), islands (adds new indices)
    """
    for island_idx, island in enumerate(islands):
        programs = {population[i] for i in island}
        assert len(programs) >= k_max, f"Not enough programs in island {island_idx} to sample {k_max} parents"
        parent_indices = list(uniform_sample(programs, k=k_max))

        for batch_idx in range(batch_size):
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
    """Prune each island to critical_population_size best programs, mutating islands in-place.

    Args:
        islands: list of island sets (each set contains program indices)
        population: Population object to resolve indices to Programs
        evolution: evolution config dict containing critical_population_size
    """
    keep_n = evolution["critical_population_size"]
    for i, island in enumerate(islands):
        programs = {population[idx] for idx in island}
        ranked = sorted(programs, key=lambda p: p.program_losses.discover.final)
        islands[i] = {p.idx for p in ranked[:keep_n]}


def uniform_sample(programs: set[Program], k: int = 1) -> set[int]:
    """
    Sample k global indices uniformly at random.

        programs_0 = {pop[i] for i in island_0}
        parents    = uniform_sample(programs_0, k=2)  # e.g. {0, 2}
    """
    programs = list(programs)
    if k > len(programs):
        raise ValueError(f"k={k} exceeds population size {len(programs)}")
    chosen = np.random.choice(len(programs), size=k, replace=False)
    return {programs[j].idx for j in chosen}


def boltzmann_sample(programs: set[Program], k: int = 1, temperature: float = 1.0) -> set[int]:
    """
    Sample k global indices using a Boltzmann distribution over relative,
    std-normalised losses. High temperature → uniform, low → best dominate.

        programs_0 = {pop[i] for i in island_0}
        parents    = boltzmann_sample(programs_0, k=2, temperature=1.0)  # e.g. {0, 2}
    """
    programs = list(programs)
    if k > len(programs):
        raise ValueError(f"k={k} exceeds population size {len(programs)}")
    losses = np.array([p.program_losses.discover.final for p in programs], dtype=float)
    logits = -(losses - losses.min()) / (np.std(losses) + 1e-6) / max(temperature, 1e-3)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    chosen = np.random.choice(len(programs), size=k, replace=False, p=probs)
    return {programs[j].idx for j in chosen}


# ─────────────────────────────────────────────────────────────────────────
# Migration
# ─────────────────────────────────────────────────────────────────────────

def migrate(islands: list[set[int]], population: Population, evolution: dict, temperature: float) -> None:
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
    # TODO: temperature needs warping before being passed to boltzmann_sample.
    # Raw temperature from schedule() lives in [1, 2]. The correct transform is
    #     T_warped = (T - 1.0) ** 4
    # which maps [1, 2] -> [0, 1] with a sharp decay, so migration becomes
    # strongly selective late in the run. Confirmed in the old hypothesis_engine.py.
    n_migrants = evolution["n_migrants"]
    topology = evolution["topology"]

    for i, island in enumerate(islands):
        programs = {population[idx] for idx in island}
        sampled = boltzmann_sample(programs, k=n_migrants, temperature=temperature)
        islands[topology[i]].update(sampled)


# ─────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────

def _are_duplicates(p_i: Program, p_j: Program, loss_tol: float, cosine_tol: float) -> bool:
    if p_i.n_params is None or p_j.n_params is None or p_i.n_params != p_j.n_params:
        return False
    if p_i.eval_fingerprint is None or p_j.eval_fingerprint is None:
        return False
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


def deduplicate_outer(islands: list[set[int]], population: Population, n_overlap: int = 6, loss_tol: float = 0.01, cosine_tol: float = 0.99) -> None:
    """Check all pairs of islands for behavioral duplicates, resetting worse islands if needed.

    For each pair of islands, if n_overlap or more programs have a behavioral duplicate
    in the other island, the worse island is reset to {0, 1}.
    The worse island is the one with the higher lowest loss; tiebreak on second-lowest, etc.
    Mutates islands in-place.

    Args:
        islands: list of island sets (each set contains program indices)
        population: Population object to resolve indices to Programs
        n_overlap: threshold for considering islands duplicates
        loss_tol: loss tolerance for duplicate detection
        cosine_tol: cosine similarity tolerance for duplicate detection
    """
    for i, j in combinations(range(len(islands)), 2):
        programs_a = {population[idx] for idx in islands[i]}
        programs_b = {population[idx] for idx in islands[j]}

        overlap = sum(
            1 for p_a in programs_a
            if any(_are_duplicates(p_a, p_b, loss_tol, cosine_tol) for p_b in programs_b)
        )
        if overlap < n_overlap:
            continue

        losses_a = sorted(p.program_losses.discover.final for p in programs_a)
        losses_b = sorted(p.program_losses.discover.final for p in programs_b)

        if losses_a <= losses_b:
            islands[j] = {0, 1}
        else:
            islands[i] = {0, 1}


def deduplicate(islands: list[set[int]], population: Population, evolution: dict, loss_tol: float = 0.01, cosine_tol: float = 0.95) -> None:
    """Apply within-island and between-island deduplication, mutating islands in-place.

    First removes duplicates within each island, then checks adjacent islands for
    cross-island duplicates and resets worse islands if needed.

    Args:
        islands: list of island sets (each set contains program indices)
        population: Population object to resolve indices to Programs
        evolution: evolution config dict containing loss_tol, cosine_tol, n_overlap
    """
    n_critical = evolution.get("critical_population_size", 12)
    n_overlap = n_critical // 2  # default threshold

    deduplicate_inner(islands, population, loss_tol, cosine_tol)
    deduplicate_outer(islands, population, n_overlap, loss_tol, cosine_tol)


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
