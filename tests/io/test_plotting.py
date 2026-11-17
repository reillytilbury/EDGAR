import json
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest

from edgar.evolution.program import BirthCertificate, Losses, LossStats, Program
from edgar.io.plotting import generate_program_fits
from edgar.projects.diagnostics import BaseDiagnostics


def test_generate_program_fits_uses_split_specific_results(tmp_path):
    records_path = tmp_path / "plot-inputs.jsonl"

    class RecordingDiagnostics(BaseDiagnostics):
        def plot_model_fits(self, data, programs, save_path="", **kwargs):
            record = {
                "data": {
                    key: np.asarray(value).tolist() for key, value in data.items()
                },
                "params": [
                    {key: np.asarray(value).tolist() for key, value in params.items()}
                    for params in kwargs["params"]
                ],
                "losses": [
                    None if value is None else float(value)
                    for value in kwargs["losses"]
                ],
                "sample_losses": [
                    None if value is None else np.asarray(value).tolist()
                    for value in kwargs["sample_losses"]
                ],
                "names": kwargs["program_names"],
            }
            with records_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    spec = SimpleNamespace(
        output_dir=str(tmp_path),
        diagnostics=RecordingDiagnostics(),
        rng=np.random.default_rng(0),
    )
    program = Program(
        birth=BirthCertificate(generation=0, island=0, batch_index=0),
        name="split-test",
        params={"w": np.array([2.0])},
        params_init={"w": np.array([1.0])},
        params_validate={"w": np.array([20.0])},
        params_init_validate={"w": np.array([10.0])},
        sample_losses=np.array([0.25, 0.5]),
        sample_losses_init=np.array([1.0, 2.0]),
        program_losses=Losses(
            discover=LossStats(init=4.0, final=2.0),
            validate=LossStats(init=40.0, final=20.0),
        ),
    )
    program.idx = 7

    discover_data = {"x": np.array([1.0, 2.0])}
    validate_data = {"x": np.array([9.0, 10.0, 11.0])}
    generate_program_fits(spec, discover_data, [program], split="discover")
    generate_program_fits(spec, validate_data, [program], split="validate")

    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert records[0] == {
        "data": {"x": [1.0, 2.0]},
        "params": [{"w": [1.0]}, {"w": [2.0]}],
        "losses": [4.0, 2.0],
        "sample_losses": [[1.0, 2.0], [0.25, 0.5]],
        "names": ["split-test (Init)", "split-test (Final)"],
    }
    assert records[1] == {
        "data": {"x": [9.0, 10.0, 11.0]},
        "params": [{"w": [10.0]}, {"w": [20.0]}],
        "losses": [40.0, 20.0],
        "sample_losses": [None, None],
        "names": ["split-test (Init)", "split-test (Final)"],
    }
    assert program.fit_image_path is not None
    assert program.fit_image_path_validate is not None
    assert program.fit_image_path != program.fit_image_path_validate
    np.testing.assert_array_equal(program.params["w"], [2.0])
    np.testing.assert_array_equal(program.params_validate["w"], [20.0])


@pytest.mark.parametrize("split", ["discover", "validate"])
def test_generate_program_fits_skips_already_plotted(tmp_path, monkeypatch, split):
    import edgar.io.plotting as plotting

    calls = []

    def fake_run_in_subprocess(worker, args, timeout):
        _spec_bytes, _data, programs_bytes, call_split = args
        programs = cloudpickle.loads(programs_bytes)
        calls.append((call_split, [program.idx for program in programs]))
        return [
            (program.idx, str(tmp_path / f"{call_split}-P{program.idx:04d}.png"))
            for program in programs
        ]

    monkeypatch.setattr(plotting, "run_in_subprocess", fake_run_in_subprocess)
    spec = SimpleNamespace(
        output_dir=str(tmp_path),
        diagnostics=BaseDiagnostics(),
        rng=np.random.default_rng(0),
    )
    programs = []
    for idx in range(3):
        program = Program(
            birth=BirthCertificate(generation=0, island=0, batch_index=idx),
            params={"w": np.array([float(idx + 1)])},
            params_init={"w": np.array([0.0])},
            params_validate={"w": np.array([float(idx + 11)])},
            params_init_validate={"w": np.array([float(idx + 10)])},
            program_losses=Losses(
                discover=LossStats(init=1.0, final=0.5),
                validate=LossStats(init=2.0, final=1.0),
            ),
        )
        program.idx = idx
        programs.append(program)

    data = {"x": np.array([1.0])}
    generate_program_fits(spec, data, programs[:2], split=split)
    assert calls == [(split, [0, 1])]
    first_paths = [
        p.fit_image_path_validate if split == "validate" else p.fit_image_path
        for p in programs[:2]
    ]
    assert all(first_paths)

    generate_program_fits(spec, data, programs, split=split)
    assert calls == [(split, [0, 1]), (split, [2])]
    assert (
        programs[2].fit_image_path_validate
        if split == "validate"
        else programs[2].fit_image_path
    )

    # A further call has no pending programs and should not launch another worker.
    generate_program_fits(spec, data, programs, split=split)
    assert len(calls) == 2
