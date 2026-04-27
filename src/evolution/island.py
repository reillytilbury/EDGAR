"""
island.py

Operations on islands (plain set[int] of global Population indices) and
island census persistence.

Single-island operations (prune, sample, deduplicate) take a set of Program
objects resolved from the island and return a new set of global indices.
Cross-island operation (deduplicate_islands) takes two program sets and resets
the worse island to the two seed programs {0, 1} if the islands are duplicates.

island_census tracks each island's membership at the end of every iteration:

    census[island_id][iteration] -> set[int]

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

    # append current island state to census each iteration
    for island_id, island in enumerate(islands):
        census[island_id].append(set(island))
    save_island_census(census, "census.json")
"""

from __future__ import annotations
import json
import numpy as np
from .program import Program, BirthCertificate
from .population import Population


# ---------------------------------------------------------------------------
# Seed and spawn — creating new programs and placing them on islands
# ---------------------------------------------------------------------------

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
    iteration = _infer_iteration(population)

    for island_idx, island in enumerate(islands):
        programs = {population[i] for i in island}
        parent_indices = list(uniform_sample(programs, k=min(k_max, len(programs))))

        for batch_idx in range(batch_size):
            child = Program(
                birth=BirthCertificate(
                    generation=iteration,
                    island=island_idx,
                    batch_index=batch_idx,
                    mode=mode,
                    temperature=temperature,
                    parent_indices=parent_indices,
                ),
            )
            population.add(child)
            island.add(child.idx)


def _infer_iteration(population: Population) -> int:
    """Infer current iteration from the max generation in population births."""
    if len(population) == 0:
        return 0
    return max(population[i].birth.generation for i in range(len(population))) + 1


# helper funcs
def relative_logit_probs(losses: np.ndarray, temperature: float) -> np.ndarray:
    relative   = losses - losses.min()
    normalised = relative / (np.std(relative) + 1e-6)
    logits     = -normalised / max(temperature, 1e-3)
    logits    -= logits.max()
    probs      = np.exp(logits)
    probs     /= probs.sum()
    return probs


# island operations: prune, sample, deduplicate. Each takes a set of Programs and returns the new island (set of global indices)
def prune(programs: set[Program], keep_n: int) -> set[int]:
    """
    Return the best keep_n global indices by losses.discover.final.

        programs_0 = {pop[i] for i in island_0}  # e.g. losses 0.5, 0.2, 0.9
        island_0   = prune(programs_0, keep_n=2) # {idx of 0.2, idx of 0.5}
    """
    ranked = sorted(programs, key=lambda p: p.losses.discover.final)
    return {p.idx for p in ranked[:keep_n]}


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
    losses     = np.array([p.losses.discover.final for p in programs], dtype=float)
    probs      = relative_logit_probs(losses, temperature)
    chosen = np.random.choice(len(programs), size=k, replace=False, p=probs)
    return {programs[j].idx for j in chosen}


def _are_duplicates(p_i: Program, p_j: Program, loss_tol: float, cosine_tol: float) -> bool:
    if p_i.n_params is None or p_j.n_params is None or p_i.n_params != p_j.n_params:
        return False
    if p_i.eval_fingerprint is None or p_j.eval_fingerprint is None:
        return False
    if abs(p_i.losses.discover.final - p_j.losses.discover.final) > loss_tol:
        return False
    y_i = p_i.eval_fingerprint.flatten()
    y_j = p_j.eval_fingerprint.flatten()
    cosine = np.dot(y_i, y_j) / (np.linalg.norm(y_i) * np.linalg.norm(y_j) + 1e-6)
    return bool(cosine >= cosine_tol)


def deduplicate(programs: set[Program], loss_tol: float = 0.01, cosine_tol: float = 0.95) -> set[int]:
    """
    Remove near-duplicate programs, keeping the lower loss copy from each pair.
    Two programs are considered duplicates if they pass all three checks:
        1. same number of parameters
        2. losses within loss_tol of each other
        3. cosine similarity of eval fingerprints >= cosine_tol

        programs_0 = {pop[i] for i in island_0}
        island_0   = deduplicate(programs_0)  # e.g. {0, 2} — duplicates removed
    """
    to_remove     = set()
    programs_list = list(programs)

    for i, p_i in enumerate(programs_list):
        if p_i.idx in to_remove:
            continue
        for p_j in programs_list[i+1:]:
            if p_j.idx in to_remove:
                continue
            if not _are_duplicates(p_i, p_j, loss_tol, cosine_tol):
                continue
            loser = p_i if p_i.losses.discover.final >= p_j.losses.discover.final else p_j
            to_remove.add(loser.idx)

    return {p.idx for p in programs if p.idx not in to_remove}

# ---------------------------------------------------------------------------
# Cross-island deduplication
# ---------------------------------------------------------------------------

def deduplicate_islands(
    programs_a: set[Program],
    programs_b: set[Program],
    n_overlap: int = 6,
    loss_tol: float = 0.01,
    cosine_tol: float = 0.99,
) -> tuple[set[int], set[int]]:
    """
    If n_overlap or more programs in island_a have a behavioral duplicate in island_b,
    the islands are considered duplicates and the worse one is reset to {0, 1}.
    The worse island is the one with the higher lowest loss; tiebreak on second-lowest, etc.

        island_a, island_b = deduplicate_islands(programs_a, programs_b, n_overlap=6)
        # if islands are duplicates, the worse one is reset: e.g. island_b == {0, 1}
    """
    overlap = sum(
        1 for p_a in programs_a
        if any(_are_duplicates(p_a, p_b, loss_tol, cosine_tol) for p_b in programs_b)
    )
    if overlap < n_overlap:
        return {p.idx for p in programs_a}, {p.idx for p in programs_b}

    losses_a = sorted(p.losses.discover.final for p in programs_a)
    losses_b = sorted(p.losses.discover.final for p in programs_b)

    if losses_a <= losses_b:
        return {p.idx for p in programs_a}, {0, 1}
    return {0, 1}, {p.idx for p in programs_b}

# ---------------------------------------------------------------------------
# Island census save/load. 
# ---------------------------------------------------------------------------

def save_island_census(census: list[list[set[int]]], path: str) -> None:
    """
    Save island_census to JSON.
    census[island_id][iteration] is the set of program indices at end of that iteration.
    """
    with open(path, "w") as f:
        json.dump([[list(s) for s in island] for island in census], f)


def load_island_census(path: str) -> list[list[set[int]]]:
    with open(path) as f:
        data = json.load(f)
    return [[set(s) for s in island] for island in data]
