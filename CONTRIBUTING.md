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

When making a `git commit`, do the following
```bash
git add -u
make commit-check #runs ruff check --fix (autofixes) and ruff format, these lint and format the code ensuring a consistent style
# Returns status of pre-commit, files may need to be modified
git add -u
git commit -m 'a commit message'
```

Upon pushing to remote the following tests are run, and status displayed on github:
- All unit and integration pytests in `tests/` except those with live llm calls.
- Pings google and anthropic LLMs to check they can be called.


## Branch Structure

At the moment, we are actively working and developing off the [main branch](https://github.com/reillytilbury/EDGAR/tree/main).
Code on this branch should pass pre-commit checks and the `uv run pytest` unit tests.

## Testing and Validation

The repository has a suite of unit, integration and system tests, as well as validation scripts.
- `tests/` contains unit, integration tests:
    - They are run with `uv run pytest`, which excludes all tests with live LLM calls.
    - To include live LLM call tests: `uv run pytest -m "not slow"`, this runs all tests, excluding a slow system test with real LLM calls.
    - To run all tests: `uv run pytest -m ""`
- `scripts/`contains useful scripts for debugging / validating the behaviour of a project.

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

We use [Google style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).
A file with guidelines (useful for coding agents) is in `documentation_guidelines.md`.

Documentation can be automatically generated using the `.docbot`.
The workflow is:
1. In the Github Actions tab select 'Documentation Bot'.
2. Click Run workflow: Use the workflow from and run the docbot on the branch you want documentation added to (must be derived from `main`).
3. The docbot will look at which files have been modified since it was last called and add documentation to modified files, creating a new branch `docbot-updates` and a pull request onto your branch.
4. Verify the output of the docbot (it will report test status and have checked that the precommit passes), make any changes you want and merge into your branch.
   
Alternatively it can be run locally via
```bash
uv run .docbot/run_bot.py
```
