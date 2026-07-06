"""
Population.py

This module defines the `Population` class, which serves as an append-only,
globally indexed collection of `Program` objects.
It manages the persistence of all program data to a JSONL file.

Each `Program` added to the `Population` automatically receives a stable,
global index (`program.idx`), which is used for consistent referencing
across different components of the system, such as islands and the dashboard.

The `Population` is central to tracking the evolutionary progress, providing
methods for preparing programs for validation scoring, saving the entire
population state, and retrieving a rank-sorted copy.

Island operations (e.g., pruning, sampling, deduplication) are handled
externally in `edgar.evolution.island.py`, using the global indices
managed by the `Population` to refer to specific programs.

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
    programs = [popn[i] for i in island]

    # save and load
    popn.save("population.jsonl")
    popn = Population.load("population.jsonl")
"""

from __future__ import annotations
import json
from dataclasses import asdict
import numpy as np
from .program import NotValidated, Program, BirthCertificate, Code, LossPair, Losses
from ..io.utils import EDGARJSONEncoder


def _params_to_json(params: dict) -> dict:
    """Converts a JAX pytree of parameters to a JSON-serializable dictionary.

    JAX pytree leaves, typically numpy arrays, are converted to standard Python lists
    to ensure compatibility with JSON serialization. Non-array elements are kept as is.

    Args:
        params: A dictionary representing a JAX pytree of parameters, potentially
            containing numpy arrays as leaf nodes.

    Returns:
        A dictionary where numpy arrays have been converted to Python lists,
        suitable for JSON serialization.
    """
    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in params.items()}


def _params_from_json(params: dict) -> dict:
    """Converts a dictionary from JSON format back to a JAX pytree of parameters.

    Python lists that were originally numpy arrays are converted back to numpy arrays.
    Other elements remain unchanged.

    Args:
        params: A dictionary loaded from JSON, where lists might represent
            original numpy arrays.

    Returns:
        A dictionary where Python lists representing numerical data have been
        converted back to numpy arrays, suitable for use as JAX pytree parameters.
    """
    return {k: np.array(v) if isinstance(v, list) else v for k, v in params.items()}


class Population:
    """Manages the collection of all `Program` instances throughout an EDGAR experiment.

    This class is an append-only list, automatically assigning a stable global
    index (`program.idx`) to each `Program` upon addition. It handles the
    persistence of all program data, including numpy arrays for fingerprints and
    parameters, to a JSONL file (`population.jsonl`) using atomic writes.
    """

    def __init__(self) -> None:
        """Initializes an empty Population instance."""
        self._programs: list[Program] = []

    def add(self, program: Program) -> None:
        """Adds a program to the population and assigns it a global index.

        The `program.idx` attribute is automatically set to the current size
        of the population before the program is appended. This ensures a
        unique and stable global identifier for each program.

        Args:
            program: The `Program` instance to add to the population.
        """
        program.idx = len(self._programs)
        self._programs.append(program)

    def __getitem__(self, idx: int) -> Program:
        """Retrieves a program by its global index.

        Args:
            idx: The global index of the program to retrieve.

        Returns:
            The `Program` instance at the specified index.
        """
        return self._programs[idx]

    def __len__(self) -> int:
        """Returns the total number of programs in the population.

        Returns:
            The integer count of programs.
        """
        return len(self._programs)

    def prepare_validation_scoring(self, islands: list) -> None:
        """Prepares programs for final validation scoring.

        This method identifies all programs that are "alive" (i.e., currently
        members of an island) and sets their `program_losses.validate.final`
        attribute from `NotValidated` to `None`. This makes them eligible for
        validation scoring by the `_needs_scoring` filter in `scoring.py`.

        Args:
            islands: A list of sets of program indices, representing the current
                state of all islands in the evolutionary algorithm.
        """
        alive_indices = set()
        for island in islands:
            alive_indices.update(island)

        for i in alive_indices:
            self[i].program_losses.validate.final = None

    def save(self, path: str) -> None:
        """Atomically writes the entire population to a JSONL file.

        Each `Program` object is serialized into a JSON string on a new line.
        Numpy arrays (e.g., `params`, `sample_losses`) are
        converted to Python lists for JSON compatibility. The `NotValidated`
        sentinel is converted to the string "NOTVALIDATED". LLM model names
        which might be objects are converted to strings.

        The method uses a write-to-temporary-file-then-rename strategy
        (`atomic_write_text` from `edgar.io.status`) to ensure that any
        dashboard or external process polling this file never observes a
        partially written or corrupted state.

        Args:
            path: The file path where the population should be saved.
        """
        from io import StringIO

        from ..io.status import atomic_write_text

        buf = StringIO()
        for p in self._programs:
            d = asdict(p)
            d.pop("data", None)  # Do not serialize potentially large data attribute
            d.pop(
                "eval_fingerprint", None
            )  # Do not serialize fingerprint, which is also potentially large
            if isinstance(d["program_losses"]["validate"]["final"], NotValidated):
                d["program_losses"]["validate"]["final"] = "NOTVALIDATED"
            llm = d["birth"]["llm_name"]
            if llm is not None and not isinstance(llm, str):
                d["birth"]["llm_name"] = getattr(llm, "model_name", repr(llm))
            buf.write(json.dumps(d, cls=EDGARJSONEncoder) + "\n")
        atomic_write_text(path, buf.getvalue())

    def get_sorted(self) -> list[Program]:
        """Returns a new list of programs sorted by their final rank.

        Programs are sorted in ascending order based on their `rank` attribute.
        Programs with `None` ranks (meaning they haven't been ranked yet) are
        treated as having an infinite rank and are placed at the end.

        Raises:
            RuntimeError: If `scoring.rank()` has not been called on the population
                and therefore no programs have a `rank` assigned.

        Returns:
            A new list containing `Program` objects, sorted by rank.
        """
        if all(p.rank is None for p in self._programs):
            raise RuntimeError(
                "Population has not been ranked yet, call scoring.rank(population) first"
            )
        return sorted(
            self._programs, key=lambda p: p.rank if p.rank is not None else float("inf")
        )

    @classmethod
    def load(cls, path: str) -> Population:
        """Loads a population from a JSONL file.

        Each line in the file is parsed as a JSON object, representing a
        serialized `Program`. The `idx` attribute is intentionally ignored
        from the loaded JSON, as `add()` will re-assign new, correct indices
        during the loading process. Numpy arrays and `NotValidated` sentinels
        are reconstructed to their original types.

        Args:
            path: The file path to the JSONL file containing the serialized population.

        Returns:
            A new `Population` instance populated with the loaded programs.
        """
        pop = cls()
        with open(path) as f:
            for line in f:
                d = json.loads(line.strip())
                # idx is set automatically by add()
                d.pop("idx", None)
                raw_params = d.get("params")
                raw_params_init = d.get("params_init")
                raw_default_params = d.get("_default_params")
                raw_sample_losses = d.get("sample_losses")
                raw_sample_losses_init = d.get("sample_losses_init")
                # save() serializes the NotValidated sentinel as "NOTVALIDATED";
                # restore the sentinel so rank() / prepare_validation_scoring
                # see a real NotValidated() instance, not a bare string (which
                # would crash sorts and break the validate-eligible filter).
                validate_raw = d["program_losses"]["validate"]
                if validate_raw.get("final") == "NOTVALIDATED":
                    validate_raw = {**validate_raw, "final": NotValidated()}

                program = Program(
                    birth=BirthCertificate(**d["birth"]),
                    status=d.get("status", "alive"),
                    code=Code(**d["code"]),
                    name=d["name"],
                    program_losses=Losses(
                        discover=LossPair(**d["program_losses"]["discover"]),
                        validate=LossPair(**validate_raw),
                    ),
                    n_params=d["n_params"],
                    params=_params_from_json(raw_params)
                    if raw_params is not None
                    else None,
                    params_init=_params_from_json(raw_params_init)
                    if raw_params_init is not None
                    else None,
                    _default_params=_params_from_json(raw_default_params)
                    if isinstance(raw_default_params, dict)
                    else raw_default_params,
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
