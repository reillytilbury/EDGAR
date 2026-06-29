import json


class EDGARJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts numpy/jax arrays to lists."""

    def default(self, o):
        if hasattr(o, "tolist"):
            return o.tolist()
        return super().default(o)
