import numpy as np
from src.evolution.population import Population
from src.evolution.program import BirthCertificate, Code, Program
from tests.llm.programs import Seed1, Seed2, Program1, Program2, ProgramSolution, InvalidProgram

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
        code=Code(
            model=model_code,
            param_est=param_est_code,
            model_jax=model_code
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

def make_fingerprint_program(
    fingerprint: list[float],
    loss: float | None = None,
    n_params: int = 2,
    number: int = 0,
) -> Program:
    """Create a Program with n_params and eval_fingerprint set for deduplication tests."""
    p = make_empty_program(number)
    p.n_params = n_params
    p.eval_fingerprint = np.asarray(fingerprint, dtype=float)
    p.program_losses.discover.final = loss
    return p


def make_fingerprint_population(specs: list[tuple]) -> Population:
    """Create a population from a list of (fingerprint, loss, n_params) tuples.

    Each spec is (fingerprint, loss) or (fingerprint, loss, n_params).
    n_params defaults to 2 when omitted.
    """
    pop = Population()
    for i, spec in enumerate(specs):
        fp, loss = spec[0], spec[1]
        n_params = spec[2] if len(spec) > 2 else 2
        pop.add(make_fingerprint_program(fp, loss, n_params, number=i))
    return pop


def make_seeds():
    seed1 = Program(
        birth = BirthCertificate(generation=-1, island=-1, batch_index=0),
        code = Code(model=Seed1.model, param_est=Seed1.param_est,model_jax=Seed1.model_jax),
        name = "Seed1",
        _default_params = Seed1.default_params,
    )
    seed2 = Program(
        birth = BirthCertificate(generation=-1, island=-1, batch_index=1),
        code = Code(model=Seed2.model, param_est=Seed2.param_est,model_jax=Seed2.model_jax),
        name = "Seed2",
        _default_params = Seed2.default_params,
    )
    return [seed1, seed2]

    
