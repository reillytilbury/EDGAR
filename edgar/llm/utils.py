def translate_to_jax(code: str) -> str:
    """Translates numpy-based code to JAX-compatible code.

    This function replaces `import numpy as np` with `import jax.numpy as jnp`
    and all instances of `np.` with `jnp.`. This is a utility for converting
    LLM-generated Python code, which often uses `numpy`, into a format
    compatible with JAX for efficient execution and gradient computation within EDGAR.

    Args:
        code (str): The Python source code string containing numpy references.

    Returns:
        str: The JAX-translated source code string.
    """
    if code is None:
        return ""
    code = code.replace("import numpy as np", "import jax.numpy as jnp")
    code = code.replace("np.", "jnp.")
    return code
