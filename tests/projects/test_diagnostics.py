import cloudpickle
import pytest

from edgar.projects.diagnostics import BaseDiagnostics, is_compute_metrics_defined


class DefaultDiagnostics(BaseDiagnostics):
    pass


class CustomDiagnostics(BaseDiagnostics):
    def compute_metrics(self, data, y_pred, params=None, program=None):
        return {"metric": 1.0}


class InheritedDiagnostics(CustomDiagnostics):
    pass


class DisabledDiagnostics(CustomDiagnostics):
    compute_metrics = None


@pytest.mark.parametrize(
    "diagnostics, expected",
    [
        (None, False),
        (object(), False),
        (BaseDiagnostics(), False),
        (DefaultDiagnostics(), False),
        (CustomDiagnostics(), True),
        (InheritedDiagnostics(), True),
        (DisabledDiagnostics(), False),
    ],
)
@pytest.mark.parametrize("roundtrip", [False, True])
def test_is_compute_metrics_defined(diagnostics, expected, roundtrip):
    if roundtrip:
        diagnostics = cloudpickle.loads(cloudpickle.dumps(diagnostics))

    assert is_compute_metrics_defined(diagnostics) is expected
