# Define programs which will be returned by the fake LLM call


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
        "\treturn a * x**3 + b * x**2 + c * x"
    )

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
        '\treturn {"a": float(1), "b": float(0), "c": float(0)}\n'
    )


class ProgramSolution:
    """Exact solution of the noiseless synthetic data problem: y = (a*x^2 + b*x + c) * sin(k*x + phi_0)"""

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
        '\treturn {"a": float(1), "b": float(0), "c": float(0), "k": float(6), "phi_0": float(0)}\n'
    )
