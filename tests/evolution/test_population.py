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
from src.evolution.population import Population, _params_to_json, _params_from_json
from src.evolution.program import NotValidated, BirthCertificate, Program, Code, Losses, LossPair
from tests.evolution.utils import make_program, linear_model_code, linear_param_est_code

def initialize_program(i, model_code, param_est_code):
    """
    Helper function to create a Program with variables defined by input index i, and given model_code and param_est_code (strings).
    """
    return Program(
        birth=BirthCertificate(generation=i, island=i, batch_index=i),
        code=Code(model=model_code, param_est=param_est_code, model_jax = model_code),
        name=f"Program {i}",
        program_losses=Losses(
            discover=LossPair(init=float(i), final=float(i) / 2),
            validate=LossPair(init=float(i) * 2, final=float(i)),
        ),
        n_params=i,
        eval_fingerprint=np.array([i, i + 1, i + 2]),
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

    def test_population_save_and_load(
        self, tmp_path
    ):
        pop = Population()
        for i in range(3):
            program = initialize_program(i, linear_model_code(), linear_param_est_code())
            pop.add(program)
        pop.save(tmp_path / "population.jsonl")
        loaded_pop = Population.load(tmp_path / "population.jsonl")
        assert len(loaded_pop) == len(pop)
        # Check saved and loaded programs have the same attributes
        for p_original, p_loaded in zip(pop._programs, loaded_pop._programs):
            assert p_original.birth == p_loaded.birth
            assert p_original.code == p_loaded.code
            assert p_original.code.model_jax == p_loaded.code.model_jax
            assert p_original.name == p_loaded.name
            assert p_original.program_losses.discover.init == p_loaded.program_losses.discover.init
            assert p_original.program_losses.discover.final == p_loaded.program_losses.discover.final
            assert p_original.program_losses.validate.init == p_loaded.program_losses.validate.init
            assert p_original.program_losses.validate.final == p_loaded.program_losses.validate.final
            assert p_original.n_params == p_loaded.n_params
            np.testing.assert_array_equal(
                p_original.eval_fingerprint, p_loaded.eval_fingerprint
            )
            assert p_original.idx == p_loaded.idx

    def test_population_prepare_validation_scoring(self):
        pop = Population()
        for i in range(5):
            program = make_program()
            program.name = f"Program {i}"
            pop.add(program)
            assert isinstance(pop[i].program_losses.validate.final, NotValidated)  # Initially set to NotValidated

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