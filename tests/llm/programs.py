"""
Predetermined program code strings for fake LLM testing.

Provides Program1, Program2, ProgramSolution, InvalidProgram, and SeedPrograms
with numpy model, JAX model, and parameter estimator code as class attributes.
"""


class Program1:
    model = (
        "import numpy as np\n\n"
        "def model(data, params):\n"
        '\t""" y = ax^2 +bx """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * x**2 + b * x\n\n"
    )
    default_params = {"a": 1.0, "b": 0.0}
    latex_equation = r'y = ax^2 + bx'

    model_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t""" y = ax^2 +bx """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * x**2 + b * x\n\n"
    )

    param_est = (
        'def parameter_estimator(data):\n\treturn {"a": float(1), "b": float(0)}\n'
    )


class Program2:
    model = (
        "import numpy as np\n\n"
        "def model(data, params):\n"
        '\t"""y = ax^3 + bx^2 + cx"""\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        '\tc = params["c"]\n'
        "\treturn a * x**3 + b * x**2 + c * x\n\n"
    )

    default_params = {"a": 1.0, "b": 0.0, "c": 0.0}
    latex_equation = r'y = ax^3 + bx^2 + cx'

    model_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t"""y = ax^3 + bx^2 + cx"""\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        '\tc = params["c"]\n'
        "\treturn a * x**3 + b * x**2 + c * x\n\n"
    )

    param_est = (
        "def parameter_estimator(data):\n"
        '\treturn {"a": float(1), "b": float(0), "c": float(0)}\n'
    )


class ProgramSolution:
    """Exact solution: y = (a*x^2 + b*x + c) * sin(k*x + phi_0)"""

    model = (
        "import numpy as np\n\n"
        "def model(data, params):\n"
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        '\tc = params["c"]\n'
        '\tk = params["k"]\n'
        '\tphi_0 = params["phi_0"]\n'
        "\treturn (a * x**2 + b * x + c) * np.sin(k * x + phi_0)\n\n"
    )

    default_params = {"a": 1.0, "b": 0.0, "c": 0.0, "k": 6.0, "phi_0": 0.0}
    latex_equation = r'y = (ax^2 + bx + c) \sin(kx + \phi_0)'

    model_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        '\tc = params["c"]\n'
        '\tk = params["k"]\n'
        '\tphi_0 = params["phi_0"]\n'
        "\treturn (a * x**2 + b * x + c) * jnp.sin(k * x + phi_0)\n\n"
    )

    param_est = (
        "def parameter_estimator(data):\n"
        '\treturn {"a": float(1), "b": float(0), "c": float(0), "k": float(6), "phi_0": float(0)}\n'
    )


class InvalidProgram:
    """Program that fails to run due to a TypeError."""

    model = (
        "import numpy as np\n\n"
        "def model(data, params):\n"
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        '\tc = np.zeros(("c"))\n'
        "\treturn a * x**4 + b * x + c\n\n"
    )

    default_params = {"a": 1.0, "b": 0.0}
    latex_equation = r'y = ax^4 + bx + c'

    model_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        '\tc = jnp.zeros(("c"))\n'
        "\treturn a * x**4 + b * x + c\n\n"
    )

    param_est = (
        "def parameter_estimator(data):\n"
        '\treturn {"a": float(1), "b": float(0)}\n'
    )


class SeedPrograms:
    """JAX model strings used by SeedFakeLLM."""

    model_v1_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t""" y = a * relu(x - b) """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * jnp.maximum(0, x - b)\n\n"
    )

    default_params_v1 = {"a": 1.0, "b": 0.0}

    latex_equation_v1 = r'y = a \mathrm{relu}(x - b)'

    model_v2_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t""" y = a * x + b """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * x + b\n\n"
    )

    default_params_v2 = {"a": 1.0, "b": 0.0}

    latex_equation_v2 = r'y = ax + b'

    param_est = (
        'def parameter_estimator(data):\n\treturn {"a": float(1), "b": float(0)}\n'
    )
