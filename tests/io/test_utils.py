import json

import numpy as np
import pytest

from edgar.io.utils import EDGARJSONEncoder


def test_json_encoder_converts_numpy_arrays_and_scalars_to_lists_and_values():
    payload = {
        "array": np.array([[1, 2], [3, 4]]),
        "scalar": np.int64(5),
    }

    encoded = json.dumps(payload, cls=EDGARJSONEncoder)

    assert json.loads(encoded) == {"array": [[1, 2], [3, 4]], "scalar": 5}


def test_json_encoder_delegates_unsupported_objects_to_standard_encoder():
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(object(), cls=EDGARJSONEncoder)
