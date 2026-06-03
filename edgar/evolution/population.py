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
from .program import NotValidated, Program, BirthCertificate, Code, LossPair, Losses


def _params_to_json(params: dict) -> dict:
    """Convert param pytree leaves (arrays) to Python lists."""
    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in params.items()}


def _params_from_json(params: dict) -> dict:
    """Convert param pytree from JSON (lists → numpy arrays)."""
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
        validation.final loss from NotValidatedto None so _needs_scoring can identify them.

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
        """Atomically write the population to a JSONL file.

        Uses write-to-tmp-then-rename so a live dashboard polling this file
        never observes a partially-written state.
        """
        from io import StringIO
        from ..io.status import atomic_write_text

        buf = StringIO()
        for p in self._programs:
            d = asdict(p)
            if d["eval_fingerprint"] is not None:
                d["eval_fingerprint"] = p.eval_fingerprint.tolist()
            if d["params"] is not None:
                d["params"] = _params_to_json(p.params)
            if d.get("params_init") is not None:
                d["params_init"] = _params_to_json(p.params_init)
            if d["sample_losses"] is not None:
                d["sample_losses"] = p.sample_losses.tolist()
            if d.get("sample_losses_init") is not None:
                d["sample_losses_init"] = p.sample_losses_init.tolist()
            if isinstance(d["program_losses"]["validate"]["final"], NotValidated):
                d["program_losses"]["validate"]["final"] = "NOTVALIDATED"
            llm = d["birth"]["llm_name"]
            if llm is not None and not isinstance(llm, str):
                d["birth"]["llm_name"] = getattr(llm, "model_name", repr(llm))
            buf.write(json.dumps(d) + "\n")
        atomic_write_text(path, buf.getvalue())

    def get_sorted(self) -> Population:
        """Return a copy of the population sorted by rank"""
        if all(p.rank is None for p in self._programs):
            raise RuntimeError(
                "Population has not been ranked yet, call scoring.rank(population) first"
            )
        return sorted(
            self._programs, key=lambda p: p.rank if p.rank is not None else float("inf")
        )

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
                raw_params_init = d.get("params_init")
                raw_sample_losses = d.get("sample_losses")
                raw_sample_losses_init = d.get("sample_losses_init")
                program = Program(
                    birth=BirthCertificate(**d["birth"]),
                    code=Code(**d["code"]),
                    name=d["name"],
                    program_losses=Losses(
                        discover=LossPair(**d["program_losses"]["discover"]),
                        validate=LossPair(**d["program_losses"]["validate"]),
                    ),
                    n_params=d["n_params"],
                    eval_fingerprint=fingerprint,
                    params=_params_from_json(raw_params)
                    if raw_params is not None
                    else None,
                    params_init=_params_from_json(raw_params_init)
                    if raw_params_init is not None
                    else None,
                    sample_losses=np.array(raw_sample_losses)
                    if raw_sample_losses is not None
                    else None,
                    sample_losses_init=np.array(raw_sample_losses_init)
                    if raw_sample_losses_init is not None
                    else None,
                    image_path=d.get("image_path"),
                    fit_image_path=d.get("fit_image_path"),
                    rank=d.get("rank"),
                )
                pop.add(program)
        return pop
