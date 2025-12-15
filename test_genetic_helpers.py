import unittest
import jax.numpy as jnp

import genetic_helpers as gh
from entities import Island, Program


def _make_program(
    params,
    birth_island=0,
    generation=0,
    batch_index=0,
    code_string="def neuron_model(theta): return theta",
    loss=0.1,
    eval_matrix=None,
    llm_name="g",
):
    if eval_matrix is None:
        eval_matrix = jnp.ones((params.shape[0], 5))
    return Program(
        function_code_string=code_string,
        function=lambda theta, *args: theta,
        parameter_estimator_code_string="def parameter_estimator(x, y): return x",
        parameter_estimator=lambda x, y: x,
        generation=generation,
        birth_island=birth_island,
        batch_index=batch_index,
        train_loss=loss,
        params=params,
        evaluation_matrix=eval_matrix,
        llm_name=llm_name,
    )


class CompareProgramsTest(unittest.TestCase):
    def test_simple_mode_matches_by_identifier(self):
        params = jnp.ones((2, 3))
        prog_a = _make_program(params)
        prog_b = _make_program(params, code_string="def neuron_model(theta): return -theta")
        self.assertTrue(gh.compare_programs(prog_a, prog_b, mode="simple"))

    def test_shape_mismatch_fails(self):
        prog_a = _make_program(jnp.ones((2, 3)))
        prog_b = _make_program(jnp.ones((3, 3)))
        self.assertFalse(gh.compare_programs(prog_a, prog_b, mode="complicated"))

    def test_complicated_mode_checks_losses(self):
        params = jnp.ones((2, 3))
        prog_a = _make_program(params, loss=0.1)
        prog_b = _make_program(params, loss=0.5)
        self.assertFalse(gh.compare_programs(prog_a, prog_b, mode="complicated"))


class RemoveDuplicatesTest(unittest.TestCase):
    def test_removes_higher_loss_duplicate(self):
        params = jnp.ones((2, 3))
        dup_high_loss = _make_program(params, loss=0.4)
        dup_low_loss = _make_program(params, loss=0.1)
        island = Island(0, [dup_high_loss, dup_low_loss])
        pruned = gh.remove_duplicates(island, mode="complicated")
        self.assertEqual(len(pruned), 1)
        self.assertAlmostEqual(pruned.programs[0].train_loss, 0.1)


class ComputeIntersectionTest(unittest.TestCase):
    def test_detects_overlap_between_islands(self):
        params = jnp.ones((2, 3))
        shared = _make_program(params)
        island_a = Island(0, [shared])
        island_b = Island(1, [shared, _make_program(params, birth_island=1, generation=1)])
        duplicates = gh.compute_intersection(island_a, island_b, mode="simple")
        self.assertEqual(duplicates, [0])


class ProbabilisticMigrationTest(unittest.TestCase):
    def test_migrants_appended_to_destination(self):
        params_a = jnp.ones((2, 2))
        params_b = jnp.ones((2, 2)) * 2
        island_0 = Island(0, [_make_program(params_a)])
        island_1 = Island(1, [_make_program(params_b, birth_island=1)])
        migrated = gh.perform_probabilistic_migration(
            [island_0, island_1], n_migrants=1, destination_islands=[1, 0], temperature=0.1
        )
        self.assertEqual(len(migrated[1]), 2)
        self.assertEqual(len(migrated[0]), 2)


if __name__ == "__main__":
    unittest.main()
