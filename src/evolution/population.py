"""
population.py — Global population and island management for EDGAR.

Population : append-only master list of every Program ever generated.
Island     : lightweight ordered set of Population indices.

Key invariant
-------------
Programs live *only* in the Population.  Islands hold indices, not copies.
A program that migrates to a second island is stored once in Population and
referenced by index from both islands.  Dedup/prune operations drop index
references; they never delete programs from Population.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Iterator
import numpy as np
from .program import Program
from .genetic_helpers import EvolutionConfig, RemovalEvent


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class Population:
    """
    Append-only global registry of every Program generated in a run.

    The Population is the single source of truth.  The JSONL log on disk
    is simply a serialised mirror written after each program is scored.
    """

    def __init__(self) -> None:
        self._programs: list[Program] = []

    # ------------------------------------------------------------------
    # Core list interface
    # ------------------------------------------------------------------

    def add(self, program: Program) -> int:
        """Append *program* and return its population index."""
        idx = len(self._programs)
        self._programs.append(program)
        return idx

    def __getitem__(self, idx: int) -> Program:
        return self._programs[idx]

    def __len__(self) -> int:
        return len(self._programs)

    def __iter__(self) -> Iterator[Program]:
        return iter(self._programs)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def scored(self) -> list[tuple[int, Program]]:
        """(index, program) pairs for all programs with a finite train_loss."""
        return [(i, p) for i, p in enumerate(self._programs) if p.is_scored()]

    def best(self, n: int = 1) -> list[tuple[int, Program]]:
        """Top-n (index, program) pairs by train_loss, ascending."""
        ranked = sorted(self.scored(), key=lambda ip: ip[1].train_loss)
        return ranked[:n]

    def best_test(self, n: int = 1) -> list[tuple[int, Program]]:
        """Top-n (index, program) pairs by test_loss, ascending."""
        with_test = [
            (i, p) for i, p in enumerate(self._programs)
            if p.test_loss is not None and np.isfinite(float(p.test_loss))
        ]
        return sorted(with_test, key=lambda ip: ip[1].test_loss)[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Write all programs as JSONL (one record per line).
        Callables are not saved; restore with load(..., compile=True).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for program in self._programs:
                f.write(json.dumps(program.to_record(), default=str) + "\n")
        logging.info("Population saved: %d programs → %s", len(self._programs), path)

    @classmethod
    def load(cls, path: str | Path, compile: bool = True) -> Population:
        """
        Load a Population from a JSONL file.

        Args:
            path:    Path to the JSONL file written by save().
            compile: If True, call program.compile() on each loaded program
                     to restore callable functions from code strings.
        """
        pop = cls()
        path = Path(path)
        if not path.exists():
            logging.warning("Population.load: file not found: %s", path)
            return pop
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pop._programs.append(Program.from_record(record, compile=compile))
                except Exception as exc:
                    logging.warning(
                        "Population.load: skipping malformed line %d (%s)", lineno, exc
                    )
        logging.info("Population.load: %d programs from %s", len(pop._programs), path)
        return pop


# ---------------------------------------------------------------------------
# Island
# ---------------------------------------------------------------------------

class Island:
    """
    Lightweight ordered view into the global Population.

    Stores population indices only — zero program data duplication.
    All genetic operations (dedup, prune, sample, migrate) accept a
    Population reference to resolve indices to Program objects.
    """

    def __init__(self, island_id: int, config: EvolutionConfig | None = None) -> None:
        self.island_id = island_id
        self.config = config or EvolutionConfig()
        self._indices: list[int] = []

    # ------------------------------------------------------------------
    # Core index operations
    # ------------------------------------------------------------------

    def add(self, idx: int) -> None:
        """Register population index *idx* on this island."""
        self._indices.append(idx)

    def remove(self, idx: int) -> bool:
        """Remove population index *idx*.  Returns True if it was present."""
        try:
            self._indices.remove(idx)
            return True
        except ValueError:
            return False

    def indices(self) -> list[int]:
        """Current population indices, in insertion order."""
        return list(self._indices)

    def __contains__(self, idx: int) -> bool:
        return idx in self._indices

    def __len__(self) -> int:
        return len(self._indices)

    def __repr__(self) -> str:
        return (
            f"Island(id={self.island_id}, "
            f"size={len(self)}, "
            f"capacity={self.config.capacity_per_island})"
        )

    # ------------------------------------------------------------------
    # Program access (always via population)
    # ------------------------------------------------------------------

    def programs(self, population: Population) -> list[Program]:
        return [population[i] for i in self._indices]

    def indexed_programs(self, population: Population) -> list[tuple[int, Program]]:
        """(population_index, program) pairs in island order."""
        return [(i, population[i]) for i in self._indices]

    def best(self, population: Population) -> tuple[int, Program] | None:
        """(index, program) for the lowest-loss scored program, or None."""
        scored = [(i, population[i]) for i in self._indices if population[i].is_scored()]
        return min(scored, key=lambda ip: ip[1].train_loss) if scored else None

    def sort_by_loss(self, population: Population) -> None:
        """Sort island indices by train_loss ascending, inf last."""
        self._indices.sort(key=lambda i: population[i].train_loss)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        population: Population,
        k: int,
        temperature: float = 1.0,
    ) -> list[tuple[int, Program]]:
        """
        Sample k programs with probability weighted by loss (lower = more likely).

        Returns (population_index, program) pairs without replacement.
        """
        if not self._indices:
            return []
        k = min(k, len(self._indices))

        losses = np.array([population[i].train_loss for i in self._indices], dtype=float)
        finite = np.isfinite(losses)
        if finite.any():
            worst = float(np.max(losses[finite]))
            losses = np.where(finite, losses, worst + 1.0)

        relative = losses - losses.min()
        temp = max(temperature, 1e-3)
        logits = -(relative / (np.std(relative) + 1e-6)) / temp
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        chosen = np.random.choice(len(self._indices), size=k, replace=False, p=probs)
        return [(self._indices[j], population[self._indices[j]]) for j in chosen]

    # ------------------------------------------------------------------
    # Genetic operations
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        population: Population,
        loss_tol: float | None = None,
        cosine_tol: float | None = None,
        iteration: int = -1,
    ) -> list[RemovalEvent]:
        """
        Drop within-island near-duplicates, keeping the lower-loss copy.
        Returns RemovalEvents for the JSONL log.
        """
        loss_tol = loss_tol if loss_tol is not None else self.config.loss_tolerance
        cosine_tol = cosine_tol if cosine_tol is not None else self.config.cosine_tolerance

        to_remove: set[int] = set()   # positions in self._indices
        events: list[RemovalEvent] = []
        n = len(self._indices)

        for i in range(n):
            if i in to_remove:
                continue
            p_i = population[self._indices[i]]
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                p_j = population[self._indices[j]]
                if not p_i.is_similar_to(p_j, loss_tol, cosine_tol):
                    continue

                # Keep the lower-loss program
                if p_i.train_loss <= p_j.train_loss:
                    loser_pos, loser, winner = j, p_j, p_i
                else:
                    loser_pos, loser, winner = i, p_i, p_j

                if loser_pos in to_remove:
                    continue
                to_remove.add(loser_pos)
                events.append(RemovalEvent(
                    uid=loser.uid,
                    category="deduplication",
                    event_type="DEDUP_WITHIN_ISLAND_REMOVE",
                    island_id=self.island_id,
                    iteration=iteration,
                    rule="behavioral_similarity",
                    details={
                        "kept_uid": list(winner.uid),
                        "kept_loss": float(winner.train_loss),
                        "removed_loss": float(loser.train_loss),
                    },
                ))
                if loser_pos == i:
                    break  # p_i is gone — move to next outer i

        for pos in sorted(to_remove, reverse=True):
            self._indices.pop(pos)

        if to_remove:
            logging.info(
                "Island %d deduplicated: removed %d programs.",
                self.island_id, len(to_remove),
            )
        return events

    def prune(
        self,
        population: Population,
        keep_n: int | None = None,
        iteration: int = -1,
    ) -> list[RemovalEvent]:
        """
        Prune to *keep_n* programs, dropping those with highest loss.
        Returns RemovalEvents for the JSONL log.
        """
        keep_n = keep_n if keep_n is not None else self.config.capacity_per_island
        if len(self._indices) <= keep_n:
            return []

        # Rank by loss; keep the best keep_n positions
        ranked = sorted(
            range(len(self._indices)),
            key=lambda pos: population[self._indices[pos]].train_loss,
        )
        keep_positions = set(ranked[:keep_n])

        events: list[RemovalEvent] = []
        new_indices: list[int] = []
        for pos, pop_idx in enumerate(self._indices):
            if pos in keep_positions:
                new_indices.append(pop_idx)
            else:
                p = population[pop_idx]
                events.append(RemovalEvent(
                    uid=p.uid,
                    category="pruning",
                    event_type="PRUNE_REMOVE",
                    island_id=self.island_id,
                    iteration=iteration,
                    rule="capacity_keep_lowest_loss",
                    details={"removed_loss": float(p.train_loss), "capacity": keep_n},
                ))
        self._indices = new_indices

        if events:
            logging.info(
                "Island %d pruned: removed %d programs (capacity=%d).",
                self.island_id, len(events), keep_n,
            )
        return events

    def get_migrants(
        self,
        population: Population,
        n: int,
        temperature: float = 1.0,
    ) -> list[int]:
        """Return n population indices selected for migration (by loss-weighted sampling)."""
        return [idx for idx, _ in self.sample(population, k=n, temperature=temperature)]


