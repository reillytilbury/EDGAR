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
        "\treturn a * x**2 + b * x"
    )
    default_params = {"a": 1.0, "b": 2.0}
    latex_equation = r'y = ax^2 + bx'

    model_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t""" y = ax^2 +bx """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * x**2 + b * x"
    )

    param_est = (
        'def parameter_estimator(data):\n\treturn {"a": float(1), "b": float(2)}\n'
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
        "\treturn a * x**3 + b * x**2 + c * x"
    )

    default_params = {"a": 1.0, "b": 2.0, "c": 3.0}
    latex_equation = r'y = ax^3 + bx^2 + cx'

    model_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t"""y = ax^3 + bx^2 + cx"""\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        '\tc = params["c"]\n'
        "\treturn a * x**3 + b * x**2 + c * x"
    )

    param_est = (
        "def parameter_estimator(data):\n"
        '\treturn {"a": float(1), "b": float(2), "c": float(3)}\n'
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
        "\treturn (a * x**2 + b * x + c) * np.sin(k * x + phi_0)"
    )

    default_params = {"a": 1.0, "b": 2.0, "c": 3.0, "k": 6.0, "phi_0": 0.0}
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
        "\treturn (a * x**2 + b * x + c) * jnp.sin(k * x + phi_0)"
    )

    param_est = (
        "def parameter_estimator(data):\n"
        '\treturn {"a": float(1), "b": float(2), "c": float(3), "k": float(6), "phi_0": float(0)}\n'
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
        "\treturn a * x**4 + b * x + c"
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
        "\treturn a * x**4 + b * x + c"
    )

    param_est = (
        "def parameter_estimator(data):\n"
        '\treturn {"a": float(1), "b": float(0)}\n'
    )


class Seed1:
    model = (
        "def model(data, params):\n"
        '\t"""y = a * relu(x - b)"""\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * np.maximum(0, x - b)\n\n"
    )

    model_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t""" y = a * relu(x - b) """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * jnp.maximum(0, x - b)\n\n"
    )

    param_est = (
        "def parameter_estimator(data):\n"
        '\t"""Grid search parameter estimator for the ReLU model."""\n'
        '\tx = data["x"]\n'
        '\ty = np.asarray(data["y"])\n'
        "\n"
        '\tbest_loss, best_params = float("inf"), (1.0, 0.0)\n'
        "\tfor a in np.linspace(0.1, 5.0, 20):\n"
        "\t\tfor b in np.linspace(-1.0, 1.0, 20):\n"
        "\t\t\tloss = np.mean((y - a * np.maximum(0, x - b)) ** 2)\n"
        "\t\t\tif loss < best_loss:\n"
        "\t\t\t\tbest_loss, best_params = loss, (a, b)\n"
        '\treturn {"a": float(best_params[0]), "b": float(best_params[1])}\n'
    )

    default_params = {"a": 1.0, "b": 0.0}

    latex_equation = r'y = a \mathrm{relu}(x - b)'

class Seed2:
    model = (
        "def model(data, params):\n"
        '\t"""y = a * x + b"""\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * x + b\n\n"
    )

    model_jax = (
        "import jax.numpy as jnp\n\n"
        "def model(data, params):\n"
        '\t""" y = a * x + b """\n'
        '\tx = data["x"]\n'
        '\ta = params["a"]\n'
        '\tb = params["b"]\n'
        "\treturn a * x + b\n\n"
    )

    param_est = (
        "def parameter_estimator(data):\n"
        '\t"""Least squares parameter estimator for the linear model."""\n'
        '\tx = data["x"]\n'
        '\ty = np.asarray(data["y"])\n'
        "\n"
        "\tA = np.vstack([x, np.ones(len(x))]).T\n"
        "\ta, b = np.linalg.lstsq(A, y, rcond=None)[0]\n"
        '\treturn {"a": float(a), "b": float(b)}\n'
    )

    default_params = {"a": 1.0, "b": 0.0}
    
    latex_equation = r'y = ax + b'
