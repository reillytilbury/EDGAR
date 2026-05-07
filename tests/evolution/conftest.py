import pytest
from src.evolution.population import Population
from src.evolution.program import BirthCertificate, Code, Program

@pytest.fixture
def linear_model_code():
    return """
import numpy as np

def model(data, params):
    return params['a'] * data['x'] + params['b']

model.DEFAULT_PARAMS = {'a': 1.0, 'b': 0.0}
"""


@pytest.fixture
def linear_param_est_code():
    return """
import numpy as np

def parameter_estimator(data):
    x_min, x_max = np.min(data['x']), np.max(data['x'])
    a = x_min
    b = x_max
    return {'a': float(a), 'b': float(b)}
"""

@pytest.fixture
def quadratic_model_code():
    return """
import numpy as np

def model(data, params):
    return params['a'] * data['x']**2 + params['b'] * data['x'] + params['c']


model.DEFAULT_PARAMS = {'a': 1.0, 'b': 0.0, 'c': 0.0}
"""


@pytest.fixture
def quadratic_param_est_code():
    return """
import numpy as np

def parameter_estimator(data):
    x_min, x_max = np.min(data['x']), np.max(data['x'])
    a = x_min
    b = x_max
    c = 0.0
    return {'a': float(a), 'b': float(b), 'c': float(c)}
"""


@pytest.fixture
def wrong_entrypoint_code():
    return """
def not_a_model(data, params):
    return params['a'] * data['x']

not_a_model.DEFAULT_PARAMS = {'a': 1.0}
"""


@pytest.fixture
def make_program(linear_model_code, linear_param_est_code):
    """Call with no args for a default program using the linear_model_code and linear_param_est_code above, or override model_code/param_est_code."""

    def _factory(model_code=..., param_est_code=...):
        return Program(
            birth=BirthCertificate(generation=0, island=0, batch_index=0),
            code_jax=Code(
                model=linear_model_code if model_code is ... else model_code,
                param_est=linear_param_est_code
                if param_est_code is ...
                else param_est_code,
            ),
        )

    return _factory

@pytest.fixture
def make_population():
    p1 = make_program(linear_model_code, linear_param_est_code)
    p1.name = "LinearModel"
    p2 = make_program(quadratic_model_code, quadratic_param_est_code)
    p2.name = "QuadraticModel"
    pop = Population()
    pop.add(p1)
    pop.add(p2)
    return pop