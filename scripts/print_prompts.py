"""Print all LLM prompts (model, param_est, jax) for a given project.

Constructs mock programs to verify prompts include the expected context.
num_parents is read from the project's config.yaml (supports up to 3 parents).

Usage:
    uv run python scripts/print_prompts.py <project>
    uv run python scripts/print_prompts.py orientation_tuning
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgar.io.config import Config
from edgar.evolution.program import Program, BirthCertificate, Code, Losses, LossStats


def make_parent(
    name: str, loss: float, model_code: str, param_est_code: str
) -> Program:
    p = Program(birth=BirthCertificate(generation=0, island=0, batch_index=0))
    p.name = name
    p.code = Code(model=model_code, param_est=param_est_code)
    p.program_losses = Losses(discover=LossStats(init=loss + 0.5, final=loss))
    return p


def make_current(name: str, model_code: str) -> Program:
    p = Program(birth=BirthCertificate(generation=1, island=0, batch_index=0))
    p.name = name
    p.code = Code(model=model_code)
    return p


PARENT1_MODEL = """\
def model(data, params):
    \"\"\"Parent1 Model: Simple cosine tuning.\"\"\"
    import numpy as np
    theta = data['stimulus']
    return params['a'] * np.cos(theta - params['theta_pref']) + params['b']
"""

PARENT1_PARAM_EST = """\
def parameter_estimator(data):
    \"\"\"Parent1 Parameter Estimator: Estimate amplitude and preferred angle from mean response.\"\"\"
    import numpy as np
    r = data['response']
    return {'a': float(r.max() - r.min()), 'theta_pref': 0.0, 'b': float(r.mean())}
"""

PARENT2_MODEL = """\
def model(data, params):
    \"\"\"Parent2 Model: Von Mises tuning.\"\"\"
    import numpy as np
    theta = data['stimulus']
    return params['a'] * np.exp(params['kappa'] * (np.cos(theta - params['theta_pref']) - 1)) + params['b']
"""

PARENT2_PARAM_EST = """\
def parameter_estimator(data):
    \"\"\"Parent2 Parameter Estimator: Estimate amplitude, kappa and preferred angle from circular mean.\"\"\"
    import numpy as np
    r, theta = data['response'], data['stimulus']
    theta_pref = float(np.arctan2((r * np.sin(theta)).mean(), (r * np.cos(theta)).mean()))
    return {'a': float(r.max()), 'kappa': 2.0, 'theta_pref': theta_pref, 'b': float(r.min())}
"""

PARENT3_MODEL = """\
def model(data, params):
    \"\"\"Parent3 Model: Wrapped Cauchy tuning.\"\"\"
    import numpy as np
    theta = data['stimulus']
    rho = params['rho']
    return params['a'] * (1 - rho**2) / (1 + rho**2 - 2*rho*np.cos(theta - params['theta_pref'])) + params['b']
"""

PARENT3_PARAM_EST = """\
def parameter_estimator(data):
    \"\"\"Parent3 Parameter Estimator: Estimate from peak and baseline.\"\"\"
    import numpy as np
    r = data['response']
    return {'a': float(r.max() - r.min()), 'rho': 0.5, 'theta_pref': 0.0, 'b': float(r.min())}
"""

CURRENT_MODEL = """\
def model(data, params):
    \"\"\"Current Model: Asymmetric Von Mises tuning.\"\"\"
    import numpy as np
    theta = data['stimulus']
    delta = theta - params['theta_pref']
    kappa = params['kappa'] * (1 + params['asym'] * np.cos(delta))
    return params['a'] * np.exp(kappa * (np.cos(delta) - 1)) + params['b']
"""

SEP = "=" * 70

ALL_PARENTS = [
    ("Cosine Tuning", 1.23, PARENT1_MODEL, PARENT1_PARAM_EST),
    ("Von Mises Tuning", 0.87, PARENT2_MODEL, PARENT2_PARAM_EST),
    ("Wrapped Cauchy Tuning", 0.71, PARENT3_MODEL, PARENT3_PARAM_EST),
]


def print_prompts_for_project(project: str) -> None:
    config_path = Path(f"projects/{project}/config.yaml")
    config = Config.from_yaml(config_path)
    prompts = config.prompts
    num_parents = config.llms.num_parents

    parents = [make_parent(*p) for p in ALL_PARENTS[:num_parents]]
    current = make_current("Asymmetric Von Mises", model_code=CURRENT_MODEL)

    flat_config = {
        "num_parents": num_parents,
        "max_lines": 30,
        "swear_words": "lstsq, scipy.optimize, curve_fit",
        "idea_probability": 1.0,
    }

    mock_rng = np.random.default_rng(42)
    prompts.model.select_ideas(flat_config, mock_rng)

    print(f"\n{SEP}")
    print(f"PROJECT: {project} — MODEL PROMPT (explore, {num_parents} parents)")
    print(SEP)
    print(
        prompts.model.build_prompt(
            "explore", parent_programs=parents, config=flat_config
        )
    )

    print(f"\n{SEP}")
    print(f"PROJECT: {project} — MODEL PROMPT (exploit, {num_parents} parents)")
    print(SEP)
    print(
        prompts.model.build_prompt(
            "exploit", parent_programs=parents, config=flat_config
        )
    )

    print(f"\n{SEP}")
    print(f"PROJECT: {project} — PARAM_EST PROMPT ({num_parents} parents)")
    print(SEP)
    print(
        prompts.parameter_estimator.build_prompt(
            "explore",
            parent_programs=parents,
            config=flat_config,
            current_program=current,
        )
    )

    print(f"\n{SEP}")
    print(f"PROJECT: {project} — JAX TRANSLATION PROMPT")
    print(SEP)
    print(
        prompts.jax_translator_model.build_prompt(
            "explore", current_program=current, config=flat_config
        )
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <project>")
        print(
            "Available projects:",
            [
                p.name
                for p in Path("projects").iterdir()
                if (p / "config.yaml").exists()
            ],
        )
        sys.exit(1)
    print_prompts_for_project(sys.argv[1])
