"""
A Program is one evolved candidate, representing a potential
solution to a scientific problem. It is the fundamental unit of evolution, managed
by `Population` and processed by various components, including LLMs for generation
and the scoring module for evaluation.

Top-level fields:
- birth: BirthCertificate — lineage and generation-time metadata
- code: Code — numpy source for model, param_est and jax model code
- name: descriptive model name from the LLM (e.g. "Double Gaussian Model")
- program_losses: Losses — per-split init/final scalar losses (include complexity penalty)
- sample_losses: per-sample cross-validated losses (no penalty), shape (n_samples,)
- n_params: total number of model parameters, set by count_params()
- eval_fingerprint: array of model outputs used for deduplication
- idx: global Population index, set automatically when added to a Population

Methods:
- compile_model: parse JAX model source string into a callable function.
- compile_param_est: parse parameter estimator source string into a callable function.
- default_params: property for getting and setting default parameters, which
  automatically calculates `n_params`.

Saving and loading of Program instances is handled by the `Population` class,
not by `Program` directly.
"""

from __future__ import annotations
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Callable
from ..llm.code_loading import load_function_from_source

MODEL_ENTRYPOINT = "model"
PARAM_EST_ENTRYPOINT = "parameter_estimator"


class ModelLoadingError(Exception):
    """Raised when a JAX model cannot be loaded from its source code."""

    pass


class ParamEstLoadingError(Exception):
    """Raised when a parameter estimator cannot be loaded from its source code."""

    pass


@dataclass
class BirthCertificate:
    """Metadata detailing the origin and lineage of a Program.

    Attributes:
        generation: The evolutionary generation in which the program was created.
        island: The ID of the island where the program was spawned.
        batch_index: The index of the program within its generation's batch.
        mode: The evolutionary mode (e.g., 'explore', 'exploit') at the time of creation.
        temperature: The sampling temperature used for LLM generation.
        parent_indices: A list of global indices of parent programs.
        llm_name: The name of the LLM model used to generate this program.
        ideas: A list of ideas included in the prompt to generate the program.
    """

    generation: int
    island: int
    batch_index: int
    mode: str | None = None
    temperature: float | None = None
    parent_indices: list[int] = field(default_factory=list)
    llm_name: str | None = None
    ideas: list[str] | None = None


@dataclass
class Code:
    """Stores the Python source code for a program's components.

    Attributes:
        model: The numpy source code for the scientific model.
        param_est: The numpy source code for the parameter estimation function(s).
        model_jax: The JAX-compatible source code for the scientific model.
        best_param_est: The numpy source code for the best (by discover loss) parameter estimation function.
    """

    model: str | None = None
    param_est: list[str] | str | None = None
    model_jax: str | None = None
    best_param_est: str | None = None

    def __setattr__(self, name, value):
        # Override so that param_est is always stored as a list of strings, both on initialization or when set later.
        if name == "param_est":
            if value is None:
                super().__setattr__("param_est", None)
            elif isinstance(value, str):
                super().__setattr__("param_est", [value])
            elif isinstance(value, list):
                super().__setattr__("param_est", list(value))
            else:
                super().__setattr__("param_est", [str(value)])
        else:
            super().__setattr__(name, value)


class NotValidated:
    """Marker for programs that haven't been prepared for validation scoring.
    See, prepare_validation_scoring"""

    def __repr__(self):
        return "NOTVALIDATED"

    def __float__(self):
        raise TypeError("Invalid operation on unset loss")


@dataclass
class LossPair:
    """Stores the initial and final scalar loss values for a given data split.

    Attributes:
        init: The initial loss value before parameter optimization.
        final: The final loss value after parameter optimization. Can be `NotValidated`
            if the program is awaiting validation scoring.
        all_init: Initial losses for all individual parameter estimators.
        all_final: Final losses for all individual parameter estimators.
    """

    init: float | None = None
    final: float | NotValidated | None = None
    all_init: list[float] | None = None
    all_final: list[float] | None = None


@dataclass
class Losses:
    """Aggregates loss pairs for different data splits (discover and validate).

    Attributes:
        discover: `LossPair` for the 'discover' data split, used during evolution.
        validate: `LossPair` for the 'validate' data split, used for final ranking.
            Initialized with `NotValidated` for `final` loss.
    """

    discover: LossPair = field(default_factory=LossPair)
    validate: LossPair = field(
        default_factory=lambda: LossPair(init=None, final=NotValidated())
    )


@dataclass
class Program:
    """Represents a single evolved candidate program.

    This dataclass encapsulates all relevant information about a program, including
    its origin, source code, performance metrics, parameters, and unique identifiers.
    It serves as the fundamental unit manipulated by the evolutionary algorithm
    and LLMs.

    Attributes:
        birth: A `BirthCertificate` object detailing the program's lineage.
        code: A `Code` object holding the program's source code components.
        name: A descriptive name for the model, often provided by the LLM.
        program_losses: A `Losses` object containing scalar loss values for
            discover and validate splits, including any complexity penalties.
        n_params: The total number of free parameters in the model.
        eval_fingerprint: A numpy array representing a low-dimensional
            fingerprint of the model's output, used for deduplication.
        params: The optimized parameters of the model (as a JAX pytree).
        params_init: The initial parameters of the model (as a JAX pytree).
        sample_losses: A numpy array of per-sample loss values (without penalty)
            for the optimized parameters.
        sample_losses_init: A numpy array of per-sample loss values (without penalty)
            for the initial parameters.
        image_path: Path to a generated feedback image for the LLM.
        fit_image_path: Path to an image visualizing the model's fit to data.
        idx: A globally unique index assigned to the program within the `Population`.
        rank: The final rank of the program based on its validation loss.
        status: The current evolutionary status of the program (e.g., 'alive', 'pruned', 'deduplicated').
        _default_params: Internal storage for the model's default parameters.
    """

    birth: BirthCertificate
    status: str = "alive"
    code: Code = field(default_factory=Code)
    name: str | None = None
    program_losses: Losses = field(default_factory=Losses)
    n_params: int | None = None
    eval_fingerprint: np.ndarray | None = field(default=None, repr=False)
    params: dict | None = field(default=None, repr=False)
    params_init: dict | None = field(default=None, repr=False)
    sample_losses: np.ndarray | None = field(default=None, repr=False)
    sample_losses_init: np.ndarray | None = field(default=None, repr=False)
    image_path: str | None = None
    fit_image_path: str | None = None
    idx: int | None = field(default=None, init=False)
    rank: int | None = None
    data: dict | None = field(default=None, repr=False)
    _default_params: dict | Callable | None = None

    def __post_init__(self):
        """Post-initialization hook for Program objects, called after the default dataclass `__init__` method.

        If `_default_params` are provided during initialization, this method
        uses the `default_params` setter to validate them and
        automatically calculate and cache the number of parameters (`n_params`).
        """
        if self._default_params is not None:
            self.default_params = (
                self._default_params
            )  # use the setter to validate and set n_params

    def compile_model(self) -> Callable:
        """Loads and compiles the JAX model callable from its source code.

        This method uses `load_function_from_source` to dynamically load the
        JAX-translated model code (`self.code.model_jax`) into a callable function.

        Returns:
            A callable Python function representing the JAX model.

        Raises:
            ModelLoadingError: If the JAX model source code is missing or cannot
                be loaded/compiled into a valid function.
        """
        model_fn = load_function_from_source(self.code.model_jax, MODEL_ENTRYPOINT)
        if model_fn is None:
            raise ModelLoadingError(
                f"{self.birth}: could not load '{MODEL_ENTRYPOINT}'"
            )
        return model_fn

    def compile_param_ests(self) -> list[Callable]:
        """Loads and compiles all parameter estimators from their source code.

        If compiling any parameter estimator fails or returns None, a warning is
        issued and we continue attempting to compile the remaining estimators.

        Returns:
            A list of callable Python functions representing the parameter estimators that
            compiled successfully.
        """
        estimators = self.code.param_est
        compiled = []
        for i, est_code in enumerate(estimators):
            est_fn = load_function_from_source(est_code, PARAM_EST_ENTRYPOINT)
            if est_fn is not None:
                compiled.append(est_fn)
            else:
                warnings.warn(
                    f"Program {self.idx}: Failed to compile parameter estimator {i}",
                    UserWarning,
                )
        return compiled

    # ── prompt template properties ──
    # These match the parent_program_vars used in prompt_defaults.yaml so that
    # getattr(program, var_name) works in PromptSchema.build_prompt.

    @property
    def descriptive_name(self) -> str:
        """Returns the descriptive name of the model.

        If the `name` attribute is None, an empty string is returned. This property
        is used for prompt templating.

        Returns:
            The descriptive name of the model.
        """
        return self.name or ""

    @property
    def loss_discover(self) -> float | str:
        """Returns the final discover loss of the program.

        If the final discover loss has not yet been scored, it returns the string
        "not yet scored". This property is used for prompt templating.

        Returns:
            The final discover loss as a float or "not yet scored".
        """
        v = self.program_losses.discover.final
        return v if v is not None else "not yet scored"

    @property
    def model_code(self) -> str:
        """Returns the numpy source code of the model.

        If the model code is None, an empty string is returned. This property
        is used for prompt templating.

        Returns:
            The numpy source code of the model.
        """
        return self.code.model or ""

    @property
    def param_est_code(self) -> str:
        """Returns the numpy source code of the program's best parameter estimator, which has been previously set.

        If the parameter estimator code is None, an empty string is returned.
        This property is used for prompt templating.

        Returns:
            The numpy source code of the parameter estimator.
        """
        return self.code.best_param_est or ""

    @property
    def default_params(self) -> dict:
        """Returns the dictionary of default parameters for the model.

        If `default_params` was not set or setting failed, a warning is issued,
        and `None` might be returned (though it's usually `None` if not set).
        Programs without valid default parameters will typically be assigned
        infinite loss during scoring.

        Returns:
            A dictionary of default parameters, or `None` if not set or failed.
        """
        if self._default_params is None:
            warnings.warn(
                f"Accessing default_params=None of Program #{self.idx}, default_params was not set, or setting failed",
                UserWarning,
            )
        return self._default_params

    @default_params.setter
    def default_params(self, default_params: dict | Callable):
        """Sets the default parameters for the program.
        If params is a callable, attempt to resolve it using self.data.

        This setter automatically calculates the total number of free parameters
        and caches it in `self.n_params.`.
        If an error occurs during this process, a warning is issued, and
        `_default_params` and `n_params` are set to `None`. Programs with
        `n_params=None` will be assigned infinite loss during scoring.

        Args:
            default_params: A dictionary or callable where `default_params(data)` returns a dictionary.
            Keys of the dictionary are parameter names and values are their default values (can be numpy arrays or scalars).
        """
        # Try to resolve into a dict using data to obtain correct shapes of parameters
        if callable(default_params):
            if self.data is not None:
                try:
                    default_params = default_params(self.data)
                except Exception as e:
                    warnings.warn(
                        f"Failed to resolve dynamic default_params for Program #{self.idx}: {e}",
                        UserWarning,
                    )
            else:
                raise RuntimeError(
                    f"Cannot resolve dynamic default_params for Program #{self.idx} because program.data is None"
                )

        # Set default_params from dict and count n_params
        if isinstance(default_params, dict):
            try:
                self._default_params = default_params
                self.n_params = sum(np.asarray(v).size for v in default_params.values())
            except Exception as e:
                warnings.warn(
                    f"Failed to set default_params for Program #{self.idx}: {e}",
                    UserWarning,
                )
                self._default_params = None
                self.n_params = None
        else:
            warnings.warn(
                f"Invalid default_params for Program #{self.idx}: either passed non-dict, non-callable or failed to resolve callable, type passed: {type(default_params)}. Setting default_params and n_params to None, program will be assigned infinite loss during scoring.",
                UserWarning,
            )
            self._default_params = None
            self.n_params = None
