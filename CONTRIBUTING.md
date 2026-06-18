# Contributing

Thanks for your interest in `hrp-portfolio`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with every extra + dev tooling.
uv sync --all-extras --dev
# (Equivalent: uv pip install -e ".[dev]" inside an activated venv.)
```

`uv sync` creates `.venv/` and installs the locked dependency set from
`uv.lock`. Prefix commands with `uv run` to use that env without activating it.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src tests                                  # lint
uv run ruff format --check src tests                         # formatting
uv run mypy src                                              # types (strict)
uv run pytest -q --cov=hrp --cov-report=term --cov-fail-under=85  # tests + coverage
```

- **Lint** (`ruff`) and **formatting** (`ruff format --check`) must pass.
- **Types** (`mypy --strict`) must pass on `src`.
- **Tests** (`pytest`) must pass with **coverage of at least 85%** (the gate also
  lives in `[tool.coverage.report] fail_under` in `pyproject.toml`).

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## Commit hygiene

- Use clear, present-tense commit messages.
- **Do not** add AI-attribution trailers: no `Co-Authored-By:` author lines,
  no "generated with" attributions, no robot-emoji attribution lines. The
  `.github/workflows/no-ai-attribution.yml` guard fails any PR that contains them.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
