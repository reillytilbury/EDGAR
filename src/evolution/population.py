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
    popn.add(Program(uid=(0,0,0), model_code="...", param_est_code="..."))
    popn.add(Program(uid=(0,0,1), model_code="...", param_est_code="..."))
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
import numpy as np
from .program import Program

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

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for p in self._programs:
                d = p.__dict__.copy()
                d["uid"]        = list(d["uid"])
                d["parent_ids"] = [list(x) for x in d["parent_ids"]]
                if d["eval_fingerprint"] is not None:
                    d["eval_fingerprint"] = d["eval_fingerprint"].tolist()
                f.write(json.dumps(d) + "\n")

    @classmethod
    def load(cls, path: str) -> Population:
        pop = cls()
        with open(path) as f:
            for line in f:
                d = json.loads(line.strip())
                d.pop("idx")
                d["uid"]        = tuple(d["uid"])
                d["parent_ids"] = [tuple(p) for p in d["parent_ids"]]
                if d["eval_fingerprint"] is not None:
                    d["eval_fingerprint"] = np.array(d["eval_fingerprint"])
                pop.add(Program(**d))
        return pop
