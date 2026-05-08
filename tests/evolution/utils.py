from src.evolution.population import Population
from src.evolution.program import BirthCertificate, Code, Program

def linear_model_code():
    return """
import numpy as np

def model(data, params):
    return params['a'] * data['x'] + params['b']

model.DEFAULT_PARAMS = {'a': 1.0, 'b': 0.0}
"""

def linear_param_est_code():
    return """
import numpy as np

def parameter_estimator(data):
    x_min, x_max = np.min(data['x']), np.max(data['x'])
    a = x_min
    b = x_max
    return {'a': float(a), 'b': float(b)}
"""

def quadratic_model_code():
    return """
import numpy as np

def model(data, params):
    return params['a'] * data['x']**2 + params['b'] * data['x'] + params['c']


model.DEFAULT_PARAMS = {'a': 1.0, 'b': 0.0, 'c': 0.0}
"""


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

def wrong_entrypoint_code():
    return """
def not_a_model(data, params):
    return params['a'] * data['x']
"""

def make_empty_program(number: int = 0):
    return Program(
        birth=BirthCertificate(generation=-1, island=-1, batch_index=-1),
        name = f"Program{number}"
    )

def make_empty_population(num_programs: int = 2):
    pop = Population()
    for i in range(num_programs):
        pop.add(make_empty_program(number=i))
    return pop

def make_program(model_code = linear_model_code(), param_est_code = linear_param_est_code(), number = 0, default_params = None):
    return Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        code_jax=Code(
            model=model_code,
            param_est=param_est_code
        ),
        name = f"Program{number}",
        _default_params = default_params
    )

def make_population():
    p1 = make_program(linear_model_code(), linear_param_est_code(), number=1)
    p2 = make_program(quadratic_model_code(), quadratic_param_est_code(), number=2)
    pop = Population()
    pop.add(p1)
    pop.add(p2)
    return pop