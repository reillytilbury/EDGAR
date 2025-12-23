import os
import tempfile
import textwrap
import unittest

import jax.numpy as jnp

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

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


class VmapOverUnitsTest(unittest.TestCase):
    def test_runs_model_for_each_unit(self):
        def toy(theta, scale, offset):
            return theta * scale + offset

        vmapped = utils.vmap_over_units(toy)
        theta = jnp.array([0.0, 1.0, 2.0])
        params = jnp.array([[1.0, 0.0], [2.0, 1.0]])
        outputs = vmapped(theta, params)
        self.assertEqual(outputs.shape, (2, 3))
        self.assertTrue(jnp.allclose(outputs[0], theta))
        self.assertTrue(jnp.allclose(outputs[1], theta * 2 + 1))


@unittest.skipIf(np is None, "NumPy is required for translation validation tests")
class ValidateTranslationTest(unittest.TestCase):
    def test_translation_matches_numpy(self):
        def numpy_model(theta, A=1.5, B=0.3, phase=0.1):
            theta = np.asarray(theta)
            return B + A * np.sin(theta - phase)

        def jax_model(theta, A=1.5, B=0.3, phase=0.1):
            theta = jnp.asarray(theta)
            return B + A * jnp.sin(theta - phase)

        self.assertTrue(utils.validate_jax_translation(numpy_model, jax_model))

    def test_translation_detects_mismatch(self):
        def numpy_model(theta, A=1.0, B=0.5):
            theta = np.asarray(theta)
            return B + A * theta

        def jax_model(theta, A=1.0, B=0.5):
            theta = jnp.asarray(theta)
            return B - A * theta

        self.assertFalse(utils.validate_jax_translation(numpy_model, jax_model))


class ConfigLoaderTest(unittest.TestCase):
    def tearDown(self):
        utils.reset_prompt_overrides()

    def test_load_edgar_config_and_overrides(self):
        yaml_text = textwrap.dedent("""
        engine:
          function_name: neuron_model
          n_generations: 1
          unused_key: should_be_ignored
        seed_programs:
          - seed_1:
            function: |
              import numpy as np
              def neuron_model(theta, A=1.0, B=0.0):
                  theta = np.asarray(theta)
                  return B + A * theta
            parameter_estimator: |
              import numpy as np
              def parameter_estimator(theta, spikes):
                  return np.array([1.0, 0.0])
        prompt:
          program_creation_context: "OVERRIDE {function_name}"
        diagnostic_function: |
          def diagnostic_image(programs, X, Y, save_path, metadata=None):
              return b"image-bytes"
        diagnostic_function_name: diagnostic_image
        """)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write(yaml_text)
            tmp_path = tmp.name
        try:
            config = utils.load_edgar_config(tmp_path)
        finally:
            os.remove(tmp_path)

        self.assertIn("seed_functions_numpy", config)
        self.assertEqual(len(config["seed_functions_numpy"]), 1)
        self.assertIn("func_name", config)
        self.assertNotIn("function_name", config)
        prompt = utils.create_program_prompt([], mode="explore", use_image=False, function_name="neuron_model")
        self.assertIn("OVERRIDE neuron_model", prompt)
        self.assertIn("diagnostic_image_fn", config)

# class TestLLMCalls(unittest.TestCase):
#     def test_llm_call(self):
#         prompt = "What is the capital of France?"
#         for llm in ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.5-flash']:
#             response = utils.call_llm(prompt, model_name=llm)
#             self.assertIsInstance(response, str)
#             # assert paris in response and if not raise AssertionError with llm name
#             self.assertIn("Paris", response, f"LLM {llm} did not return expected answer.")

if __name__ == "__main__":
    unittest.main()
