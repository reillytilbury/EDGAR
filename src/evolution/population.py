"""
population.py

Population is an append-only list of Programs and the only way to persist them.
Each Program's global index (program.idx) is set automatically on add(), so it
is stable even as programs are grouped into islands elsewhere.

Islands are plain set[int] (global indices) managed externally. Island operations
(prune, sample, deduplicate, deduplicate_islands) live in island.py.

Example usage:
--------------
    popn = Population()

    # add programs — idx is set automatically
    popn.add(Program(birth=BirthCertificate(generation=0, island=0, batch_index=0),
                     code=Code(model="...", param_est="...")))
    popn.add(Program(birth=BirthCertificate(generation=0, island=0, batch_index=1),
                     code=Code(model="...", param_est="...")))
    # popn[0].idx == 0, popn[1].idx == 1, etc.

    # look up by global index
    p = popn[0]

    # resolve an island (set of indices) into Program objects for island operations
    island = {0, 1}
    programs = {popn[i] for i in island}

    # save and load
    popn.save("population.jsonl")
    popn = Population.load("population.jsonl")
"""

from __future__ import annotations
import json
from dataclasses import asdict
import numpy as np
from .program import Program, BirthCertificate, Code, LossPair, Losses


def _params_to_json(params: dict) -> dict:
    """Recursively convert param pytree leaves (arrays) to nested Python lists."""
    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in params.items()}


def _params_from_json(params: dict) -> dict:
    """Reconstruct param pytree from JSON (nested lists → numpy arrays)."""
    return {k: np.array(v) if isinstance(v, list) else v for k, v in params.items()}


class Population:
    def __init__(self) -> None:
        self._programs: list[Program] = []

    def add(self, program: Program) -> None:
        program.idx = len(self._programs)
        self._programs.append(program)

    def __getitem__(self, idx: int) -> Program:
        return self._programs[idx]

    def __len__(self) -> int:
        return len(self._programs)

    def prepare_validation_scoring(self, islands: dict | list) -> None:
        """Set validation loss to None for programs alive at validation time.

        Programs on an island are eligible for validation scoring. This resets their
        validation.final loss from _NotYetPrepared to None so _needs_scoring can identify them.

        Args:
            islands: dict or list of island sets containing program indices
        """
        alive_indices = set()
        island_list = islands.values() if isinstance(islands, dict) else islands
        for island in island_list:
            alive_indices.update(island)

        for i in alive_indices:
            self[i].program_losses.validate.final = None

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for p in self._programs:
                d = asdict(p)
                if d["eval_fingerprint"] is not None:
                    d["eval_fingerprint"] = p.eval_fingerprint.tolist()
                if d["params"] is not None:
                    d["params"] = _params_to_json(p.params)
                if d["sample_losses"] is not None:
                    d["sample_losses"] = p.sample_losses.tolist()
                f.write(json.dumps(d) + "\n")

    @classmethod
    def load(cls, path: str) -> Population:
        pop = cls()
        with open(path) as f:
            for line in f:
                d = json.loads(line.strip())
                d.pop("idx", None)
                fingerprint = d["eval_fingerprint"]
                if fingerprint is not None:
                    fingerprint = np.array(fingerprint)
                raw_params = d.get("params")
                raw_sample_losses = d.get("sample_losses")
                program = Program(
                    birth=BirthCertificate(**d["birth"]),
                    code=Code(**d["code"]),
                    code_jax=Code(**d["code_jax"]),
                    name=d["name"],
                    program_losses=Losses(
                        discover=LossPair(**d["program_losses"]["discover"]),
                        validate=LossPair(**d["program_losses"]["validate"]),
                    ),
                    n_params=d["n_params"],
                    eval_fingerprint=fingerprint,
                    params=_params_from_json(raw_params) if raw_params is not None else None,
                    sample_losses=np.array(raw_sample_losses) if raw_sample_losses is not None else None,
                    image_path=d.get("image_path"),
                )
                pop.add(program)
        return pop
