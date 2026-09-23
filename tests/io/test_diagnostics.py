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


def test_generate_program_fits_skips_already_plotted(tmp_path):
    from edgar.io.plotting import generate_program_fits
    from edgar.evolution.program import Losses, LossStats
    from types import SimpleNamespace

    call_count = 0

    class CountingDiagnostics(BaseDiagnostics):
        def plot_model_fits(self, data, programs, save_path="", **kwargs):
            nonlocal call_count
            call_count += 1
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([1, 2], [3, 4])
            fig.savefig(save_path)
            plt.close(fig)

    spec = SimpleNamespace(
        output_dir=str(tmp_path),
        diagnostics=CountingDiagnostics(),
        rng=np.random.default_rng(0),
    )

    p1 = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        params={"w": np.array([1.0])},
        params_init={"w": np.array([0.0])},
        program_losses=Losses(discover=LossStats(init=1.0, final=0.5)),
    )
    p1.idx = 0

    p2 = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=1),
        params={"w": np.array([2.0])},
        params_init={"w": np.array([0.0])},
        program_losses=Losses(discover=LossStats(init=2.0, final=1.0)),
    )
    p2.idx = 1

    # First run: should plot both p1 and p2
    generate_program_fits(spec, {"x": np.array([1, 2])}, [p1, p2])
    assert p1.fit_image_path is not None
    assert p2.fit_image_path is not None

    # Add a third program p3
    p3 = Program(
        birth=BirthCertificate(generation=1, island=0, batch_index=0),
        params={"w": np.array([3.0])},
        params_init={"w": np.array([0.0])},
        program_losses=Losses(discover=LossStats(init=3.0, final=1.5)),
    )
    p3.idx = 2

    # Second run passing all 3 programs: should only plot p3
    generate_program_fits(spec, {"x": np.array([1, 2])}, [p1, p2, p3])
    assert p3.fit_image_path is not None


def test_generate_trajectory_image_skips_already_plotted(tmp_path):
    from edgar.io.plotting import generate_trajectory_image
    from edgar.evolution.program import Losses, LossStats
    from types import SimpleNamespace

    spec = SimpleNamespace(
        output_dir=str(tmp_path),
    )

    trajs = np.array([[10.0, 5.0, 2.0], [12.0, 6.0, 1.0]])
    p1 = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        program_losses=Losses(
            discover=LossStats(init=10.0, final=1.0, trajectories=trajs)
        ),
    )
    p1.idx = 0
    p1.best_estimator_idx = 1

    generate_trajectory_image(spec, [p1])
    assert p1.trajectory_image_path is not None
    _ = p1.trajectory_image_path

    p2 = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=1),
        program_losses=Losses(
            discover=LossStats(init=2.0, final=1.0, trajectories=trajs)
        ),
    )
    p2.idx = 1
    p2.best_estimator_idx = 1

    # Overwrite property with custom marker to ensure it is not re-executed
    p1.trajectory_image_path = "already_done"
    generate_trajectory_image(spec, [p1, p2])
    assert p1.trajectory_image_path == "already_done"
    assert p2.trajectory_image_path is not None
