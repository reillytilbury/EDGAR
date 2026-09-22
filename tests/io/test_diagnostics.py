import numpy as np
from edgar.projects.diagnostics import BaseDiagnostics
from edgar.evolution.program import Program, BirthCertificate
from edgar.evolution.population import Population
from edgar.llm.prompt_schema import PromptSchema, _get_nested_attr
from edgar.io.task_spec import TaskSpec
from edgar.io.config import Config


def test_base_diagnostics_defaults():
    diag = BaseDiagnostics()
    metrics = diag.compute_metrics({}, np.array([1.0, 2.0]))
    assert metrics == {}


def test_prompt_schema_nested_dict_attr():
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        diagnostics={"r2_overall": 0.85, "r2_signal": 0.92, "nested": {"val": 42}},
    )
    assert _get_nested_attr(program, "diagnostics.r2_overall") == 0.85
    assert _get_nested_attr(program, "diagnostics.r2_signal") == 0.92
    assert _get_nested_attr(program, "diagnostics.nested.val") == 42
    assert _get_nested_attr(program, "diagnostics.nonexistent", default="N/A") == "N/A"

    schema = PromptSchema(
        base="Base prompt",
        explore="Explore",
        code_guidelines="Code guidelines",
        docstring_guidelines="Docstring guidelines",
        parent_program_template="Model {name}: R2={diagnostics.r2_overall}",
    )
    prompt = schema.build_prompt("explore", parent_programs=[program])
    assert "R2=0.85" in prompt


def test_population_save_load_diagnostics(tmp_path):
    pop = Population()
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        name="TestModel",
        diagnostics={"r2_overall": 0.88, "r2_signal": 0.95},
    )
    pop.add(program)
    save_file = tmp_path / "population.jsonl"
    pop.save(str(save_file))

    loaded_pop = Population.load(str(save_file))
    assert len(loaded_pop) == 1
    assert loaded_pop[0].diagnostics == {"r2_overall": 0.88, "r2_signal": 0.95}


def test_trial_variability_diagnostics_loading():
    config = Config.from_yaml("projects/trial_variability/config.yaml")
    spec = TaskSpec.from_config(config)
    assert spec.diagnostics is not None
    assert hasattr(spec.diagnostics, "compute_metrics")
    assert hasattr(spec.diagnostics, "plot_model_fits")

    # Test compute_metrics on mock data
    n_trials = 20
    n_cells = 10
    stim = np.linspace(0, np.pi, n_trials)
    resp = np.ones((n_trials, n_cells))
    data = {"stimulus": stim, "response": resp}
    pred = np.ones((n_trials, n_cells)) * 0.9

    metrics = spec.diagnostics.compute_metrics(data, pred)
    assert "r2_overall" in metrics
    assert "r2_signal" in metrics
    assert "r2_noise" in metrics
    assert "fano_slope_data" in metrics
    assert "fano_slope_pred" in metrics


def test_diagnostics_feedback_image_worker(tmp_path):
    from edgar.io.plotting import generate_feedback_image
    from types import SimpleNamespace

    created_files = []

    class CustomDiagnostics(BaseDiagnostics):
        def generate_feedback_image(self, data, parents, rng, save_path="", **kwargs):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([1, 2], [3, 4])
            fig.savefig(save_path)
            plt.close(fig)
            created_files.append(save_path)

    spec = SimpleNamespace(
        output_dir=str(tmp_path),
        diagnostics=CustomDiagnostics(),
        rng=np.random.default_rng(0),
    )
    program = Program(birth=BirthCertificate(generation=1, island=0, batch_index=2))
    img_bytes = generate_feedback_image(
        spec=spec, data={"x": np.array([1, 2])}, parents=[], program=program
    )
    assert img_bytes is not None
    assert len(img_bytes) > 0
    assert len(program.image_path) > 0
