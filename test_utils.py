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
