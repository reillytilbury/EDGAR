"""
Tests forsrc/evolution/population.py.

Covers:
    Population:
    - add(): check adding programs to a population works as expected
    - save() / load(): check whether all of the programs in a population are saved and loaded back in with the same attributes
    - _params_to_json / _params_from_json: round-trip serialization of param dicts
    - prepare_validation_scoring(): check that validation.final is set to None for programs in specified islands, and unchanged for others

    _params_to_json / _params_from_json: Check that a dict of params with numpy arrays and scalars can be serialized to JSON and back.
"""

import json
import numpy as np
from edgar.evolution.population import Population, _params_to_json, _params_from_json
from edgar.evolution.program import (
    NotValidated,
    BirthCertificate,
    Program,
    Code,
    Losses,
    LossStats,
)
from tests.evolution.utils import make_program, linear_model_code, linear_param_est_code


def initialize_program(i, model_code, param_est_code):
    """
    Helper function to create a Program with variables defined by input index i, and given model_code and param_est_code (strings).
    """
    return Program(
        birth=BirthCertificate(generation=i, island=i, batch_index=i),
        status="alive",
        code=Code(
            model=model_code,
            param_est=param_est_code,
            model_jax=model_code,
            best_param_est=param_est_code,
        ),
        name=f"Program {i}",
        program_losses=Losses(
            discover=LossStats(init=float(i), final=float(i) / 2),
            validate=LossStats(init=float(i) * 2, final=float(i)),
        ),
        n_params=i,
        params_init={"w": np.array([float(i)])},
        sample_losses=np.array([float(i), float(i + 1)]),
        sample_losses_init=np.array([float(i + 2), float(i + 3)]),
        image_path=f"path/to/image_{i}.png",
        fit_image_path=f"path/to/fit_{i}.png",
        trajectory_image_path=f"path/to/trajectory_{i}.png",
        rank=i,
        best_estimator_idx=0,
        _default_params={"w": np.array([float(i)])},
    )


class TestPopulation:
    def test_population_add_programs(self):
        pop = Population()
        assert len(pop) == 0

        program1 = make_program()
        program1.name = "Program 1"
        pop.add(program1)
        assert len(pop) == 1
        assert pop[0].name == program1.name

        program2 = make_program()
        program2.name = "Program 2"
        pop.add(program2)
        assert len(pop) == 2
        assert pop[1].name == program2.name

    def test_population_save_and_load(self, tmp_path):
        pop = Population()
        for i in range(3):
            program = initialize_program(
                i, linear_model_code(), linear_param_est_code()
            )
            pop.add(program)
        pop.save(tmp_path / "population.jsonl", save_trajectories=True)
        loaded_pop = Population.load(tmp_path / "population.jsonl")
        assert len(loaded_pop) == len(pop)
        # Check saved and loaded programs have the same attributes
        for p_original, p_loaded in zip(pop._programs, loaded_pop._programs):
            assert p_original.birth == p_loaded.birth
            assert p_original.status == p_loaded.status
            assert p_original.code == p_loaded.code
            assert p_original.name == p_loaded.name
            assert (
                p_original.program_losses.discover.init
                == p_loaded.program_losses.discover.init
            )
            assert (
                p_original.program_losses.discover.final
                == p_loaded.program_losses.discover.final
            )
            assert (
                p_original.program_losses.validate.init
                == p_loaded.program_losses.validate.init
            )
            assert (
                p_original.program_losses.validate.final
                == p_loaded.program_losses.validate.final
            )
            assert p_original.n_params == p_loaded.n_params
            assert np.allclose(p_original.params_init["w"], p_loaded.params_init["w"])
            assert np.allclose(p_original.sample_losses, p_loaded.sample_losses)
            assert np.allclose(
                p_original.sample_losses_init, p_loaded.sample_losses_init
            )
            assert p_original.image_path == p_loaded.image_path
            assert p_original.fit_image_path == p_loaded.fit_image_path
            assert p_original.trajectory_image_path == p_loaded.trajectory_image_path
            assert p_original.rank == p_loaded.rank
            assert p_original.best_estimator_idx == p_loaded.best_estimator_idx
            assert p_original.idx == p_loaded.idx
            assert p_original.default_params == p_loaded.default_params

    def test_population_save_and_load_with_numpy_array_default_params(self, tmp_path):
        pop = Population()
        program = initialize_program(0, linear_model_code(), linear_param_est_code())

        # Manually set resolved _default_params with numpy arrays
        default_params = {"a": np.array([1.0, 2.0]), "b": np.array([0.5])}
        program._default_params = default_params
        # n_params should correspond to the size of the arrays (3)
        program.n_params = 3
        pop.add(program)

        pop.save(tmp_path / "population.jsonl")
        loaded_pop = Population.load(tmp_path / "population.jsonl")

        assert len(loaded_pop) == 1
        loaded_program = loaded_pop[0]

        # Check _default_params is correctly loaded back with numpy arrays
        assert isinstance(loaded_program.default_params, dict)
        np.testing.assert_array_equal(
            loaded_program.default_params["a"], default_params["a"]
        )
        np.testing.assert_array_equal(
            loaded_program.default_params["b"], default_params["b"]
        )
        assert loaded_program.n_params == 3

    def test_population_save_excludes_large_data_and_eval(self, tmp_path):
        pop = Population()
        program = initialize_program(0, linear_model_code(), linear_param_est_code())
        program.data = {"response": np.array([1, 2, 3]), "signal": np.array([4, 5, 6])}
        pop.add(program)

        jsonl_path = tmp_path / "population.jsonl"
        pop.save(jsonl_path)

        # Inspect the saved JSON line directly to ensure "data" is not in there
        with open(jsonl_path) as f:
            saved_json = json.loads(f.readline().strip())
            assert "data" not in saved_json
            assert "eval_fingerprint" not in saved_json

    def test_population_save_with_save_trajectories_false(self, tmp_path):
        pop = Population()
        program = initialize_program(0, linear_model_code(), linear_param_est_code())
        # Set some trajectories
        program.program_losses.discover.trajectories = np.array([1.2, 0.9, 0.5])
        program.program_losses.validate.trajectories = np.array([2.4, 1.8, 1.0])
        pop.add(program)

        jsonl_path = tmp_path / "population_no_trajectories.jsonl"
        # Save with save_trajectories=False
        pop.save(jsonl_path, save_trajectories=False)

        # 1. Verify JSON file does not contain trajectories
        with open(jsonl_path) as f:
            saved_json = json.loads(f.readline().strip())
            assert "trajectories" not in saved_json["program_losses"]["discover"]
            assert "trajectories" not in saved_json["program_losses"]["validate"]

        # 2. Verify loading back successfully handles missing trajectories
        loaded_pop = Population.load(jsonl_path)
        assert len(loaded_pop) == 1
        loaded_program = loaded_pop[0]
        assert loaded_program.program_losses.discover.trajectories is None
        assert loaded_program.program_losses.validate.trajectories is None
        # Other values should load correctly
        assert loaded_program.program_losses.discover.init == 0.0

    def test_population_prepare_validation_scoring(self):
        pop = Population()
        for i in range(5):
            program = make_program()
            program.name = f"Program {i}"
            pop.add(program)
            assert isinstance(
                pop[i].program_losses.validate.final, NotValidated
            )  # Initially set to NotValidated

        # Prepare validation scoring for programs at indices 1, 3, 4
        pop.prepare_validation_scoring(islands=[{1, 3, 4}])

        # Check that validation.final is None for programs at indices 1, 3, 4 and unchanged for others
        for i in range(5):
            if i in {1, 3, 4}:
                assert pop[i].program_losses.validate.final is None
            else:
                assert isinstance(pop[i].program_losses.validate.final, NotValidated)


class TestParamSerialization:
    def test_round_trip(self, tmp_path):
        params = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([0.5]),
            "c": 4.2,
        }
        json_path = tmp_path / "params.json"
        serialized = _params_to_json(params)
        json_path.write_text(json.dumps(serialized))
        recovered = _params_from_json(json.loads(json_path.read_text()))
        np.testing.assert_array_equal(recovered["a"], params["a"])
        np.testing.assert_array_equal(recovered["b"], params["b"])
        assert recovered["c"] == params["c"]
