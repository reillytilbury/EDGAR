## Setup

After cloning, install dev dependencies and register the pre-commit hook:

```bash
uv sync --group dev
pre-commit install
```

Verify your environment is correctly set up:
```bash
bash scripts/check_env.sh   # checks edgar imports + fake pipeline
bash scripts/check_api_keys.sh  # checks all LLM API keys work
```

Before committing, stage the Python files you want to check, then run:
```bash
git add <files>
make commit-check
```
`make commit-check` runs Ruff checks and formatting on staged Python files. Review any
changes it makes and stage them before committing. The pre-commit hook also runs Ruff
on files included in a commit.

GitHub Actions runs tests and code-quality checks on pushes and pull requests. The
default pytest configuration excludes tests marked `live` and `slow`; see
[Testing and Validation](#testing-and-validation) for commands to select other test sets.


## Branch Structure

Development is based on the [main branch](https://github.com/reillytilbury/EDGAR/tree/main).
Code on this branch should pass pre-commit checks and the `uv run pytest` unit tests.

## Testing and Validation

The repository has a suite of unit, integration and system tests, as well as validation scripts.
- `tests/` contains unit, integration tests:
    - `uv run pytest` uses the configured default markers and excludes tests marked `live` or `slow`.
    - To include live tests while excluding slow tests: `uv run pytest -m "not slow"`.
    - To run all tests, including slow and live tests: `uv run pytest -m ""`.
- `scripts/` contains useful scripts for debugging and validating project behavior.

- To run the full algorithm without real LLM calls on a simplified synthetic_data problem:
```bash
uv run edgar test-fake
```
Output files are saved in `test_output`.

- To run a project in test mode which does a run of reduced size (1 generation, batch size 2, 2 islands,...):
```bash
uv run edgar test projects/{project_name}/config.yaml
```
Additional configuration overrides can be included by appending e.g `--evolution.n_generations=2`.
Output files are saved in `test_output`.

To view the output on the dashboard use:
```bash
uv run edgar dashboard test_output
```
and navigate to the relevant run.

## Documentation

We use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).
See `documentation_guidelines.md` for the repository's documentation requirements.
