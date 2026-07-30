# Contributing to StratStat

## Development setup

```bash
git clone git@github.com:aineurog/stratstat.git
cd stratstat
pip install -e ".[fast,report,dev]"
```

## Running tests

```bash
pytest
```

## Running linting and type checking

```bash
ruff check src/ tests/
mypy src/
```

## Architecture principles

See `stratstat_build_instructions.md` for the full design document. Key rules:

1. Every metric is implemented exactly once.
2. Vectorization first — avoid Python-level loops over columns.
3. Every formula must cite its source in the docstring.
4. `core` must never import from `report` — this boundary is enforced by tests.
5. New metrics use the `@register_metric` decorator; never edit a central dispatch block.

## Before submitting

- [ ] All tests pass across Python 3.10–3.13.
- [ ] `ruff check` and `mypy` are clean.
- [ ] New metrics have tests validated against known values (not just "runs without error").
- [ ] The formula reference doc (`docs/formula-reference.md`) is updated.
- [ ] Numba-accelerated and pure-numpy paths agree within tolerance (where applicable).
