import unittest
import jax.numpy as jnp

import utils


class ExtractCodeBlockTest(unittest.TestCase):
    def test_extracts_python_block(self):
        text = "Some text```python\nprint('hi')\n```\nmore"
        block = utils.extract_code_block(text)
        self.assertEqual(block, "print('hi')")

    def test_returns_full_text_when_no_markers(self):
        text = "print('hi')"
        block = utils.extract_code_block(text)
        self.assertEqual(block, text)


class SplitViaAstTest(unittest.TestCase):
    def test_finds_and_renames_functions(self):
        code = """
import numpy as np
def neuron_model_v5(theta, a):
    return a * theta
def parameter_estimator_v5(theta, y):
    return np.array([0.0])
"""
        model_code, estimator_code = utils.split_via_ast(code, function_name="neuron_model")
        self.assertIn("def neuron_model(", model_code)
        self.assertIn("def parameter_estimator(", estimator_code)


class VmapOverCellsTest(unittest.TestCase):
    def test_runs_model_for_each_cell(self):
        def toy(theta, scale, offset):
            return theta * scale + offset

        vmapped = utils.vmap_over_cells(toy)
        theta = jnp.array([0.0, 1.0, 2.0])
        params = jnp.array([[1.0, 0.0], [2.0, 1.0]])
        outputs = vmapped(theta, params)
        self.assertEqual(outputs.shape, (2, 3))
        self.assertTrue(jnp.allclose(outputs[0], theta))
        self.assertTrue(jnp.allclose(outputs[1], theta * 2 + 1))


if __name__ == "__main__":
    unittest.main()
