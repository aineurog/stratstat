"""Tests for column mapping: the Schema class and its application.

Covers the contract locked in the plan: exact matching only, never inferred;
the key is the canonical name and the value is the column in the user's data;
columns already using the canonical name need no entry; one schema serves many
strategies, so a mapped column that is genuinely absent warns rather than
raising, while a column named by a single-series field that is missing raises
because selecting from the wrong column would produce a confident wrong number.
"""

import numpy as np
import pandas as pd
import pytest

import stratstat as ss
from stratstat.schema import Schema, clear_schema, get_schema, set_schema


@pytest.fixture(autouse=True)
def _clear_session_schema():
    yield
    clear_schema()


def _trade_df(n=80):
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "profit": rng.normal(0.002, 0.02, size=n),
            "direction": np.where(rng.random(n) > 0.5, 1, -1),
            "bars_held": rng.integers(1, 10, size=n),
        }
    )


# ----------------------------------------------------------------------
# Schema object
# ----------------------------------------------------------------------


def test_schema_trades_maps_canonical_to_user_column():
    schema = Schema(trades={"side": "direction"})
    raw = {"pnl": np.array([1.0]), "direction": np.array([1])}
    out = schema.apply_to_trades(raw)
    assert "side" in out and "direction" not in out
    assert "pnl" in out


def test_schema_passes_through_unmapped_canonical_columns():
    schema = Schema(trades={"side": "direction"})
    raw = {"pnl": np.array([1.0]), "side": np.array([1])}
    out = schema.apply_to_trades(raw)
    assert out["side"] is raw["side"]  # untouched


def test_schema_explicit_mapping_wins_over_canonical_column():
    schema = Schema(trades={"side": "direction"})
    raw = {"side": np.array([-1]), "direction": np.array([1])}
    out = schema.apply_to_trades(raw)
    assert out["side"] is raw["direction"]


def test_schema_rejects_unknown_canonical_key():
    with pytest.raises(ValueError, match="canonical"):
        Schema(trades={"sides": "direction"})


def test_schema_rejects_unknown_key_with_suggestion():
    with pytest.raises(ValueError, match="did you mean"):
        Schema(trades={"pnl_": "profit"})


def test_schema_merge_overlays_non_empty_parts_only():
    base = Schema(trades={"pnl": "profit"}, returns="r")
    override = Schema(trades={"side": "direction"})
    merged = base.merge(override)
    assert merged.returns == "r"
    assert merged.trades == {"pnl": "profit", "side": "direction"}


def test_schema_merge_none_is_identity():
    base = Schema(trades={"pnl": "profit"})
    assert base.merge(None) is base


def test_schema_is_empty():
    assert Schema().is_empty
    assert not Schema(trades={"pnl": "profit"}).is_empty
    assert not Schema(returns="r").is_empty


# ----------------------------------------------------------------------
# Trade log mapping
# ----------------------------------------------------------------------


def test_compute_trades_flat_columns_mapping():
    df = _trade_df()
    mapped = ss.compute_trades(
        df, columns={"pnl": "profit", "side": "direction", "duration": "bars_held"}
    )
    canonical = pd.DataFrame(
        {"pnl": df["profit"], "side": df["direction"], "duration": df["bars_held"]}
    )
    direct = ss.compute_trades(canonical)
    assert {m.name for m in mapped.results} == {m.name for m in direct.results}


def test_trade_input_schema_route_matches_columns_route():
    df = _trade_df()
    by_columns = ss.compute_trades(df, columns={"pnl": "profit", "side": "direction"})
    by_schema = ss.compute_trades(df, schema=Schema(trades={"pnl": "profit", "side": "direction"}))
    assert len(by_columns.results) == len(by_schema.results)


def test_missing_mapped_trade_column_warns_and_continues():
    schema = Schema(trades={"side": "direction"})  # no direction in this log
    with pytest.warns(UserWarning, match="not in the data"):
        inp = ss.TradeInput(trades={"pnl": np.array([1.0, -1.0])}, schema=schema)
    assert inp.has_side is False
    assert inp.n_trades == 2


def test_unmapped_non_canonical_column_fails_loudly():
    """A trade log whose pnl column is neither canonical nor mapped cannot
    compute anything, and failing loudly beats producing a confident wrong
    number from zero trades."""
    df = _trade_df()
    with pytest.raises(ValueError, match="pnl"):
        ss.compute_trades(df)  # 'profit', not 'pnl', and no schema


def test_unmapped_alias_column_warns_with_hint():
    """A column that would be recognised under a different canonical name
    draws a hint naming the mapping to add, rather than dropping silently."""
    df = pd.DataFrame({"pnl": [1.0, -1.0], "direction": [1, -1]})
    with pytest.warns(UserWarning, match="columns=\\{'side': 'direction'\\}"):
        ss.TradeInput(trades=df)


def test_unmapped_close_match_column_warns():
    df = pd.DataFrame({"pnl": [1.0, -1.0], "duraton": [2, 3]})
    with pytest.warns(UserWarning, match="duraton"):
        ss.TradeInput(trades=df)


# ----------------------------------------------------------------------
# Single-series column selection
# ----------------------------------------------------------------------


def test_returns_column_selection_matches_direct():
    rng = np.random.default_rng(42)
    r = rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=504)
    frame = pd.DataFrame({"pct": r, "noise": rng.normal(size=504)})
    mapped = ss.compute_returns(frame, columns={"returns": "pct"})
    direct = ss.compute_returns(r)
    assert np.isclose(mapped["omega_ratio"].value, direct["omega_ratio"].value)


def test_returns_column_selection_missing_raises():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"pct": rng.normal(size=504)})
    with pytest.raises(KeyError, match="not in the data"):
        ss.compute_returns(frame, columns={"returns": "nope"})


def test_benchmark_column_selection():
    rng = np.random.default_rng(42)
    r = rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=504)
    b = rng.normal(0.08 / 252, 0.18 / np.sqrt(252), size=504)
    frame = pd.DataFrame({"strat": r, "bench": b})
    mapped = ss.compute_benchmark(
        frame["strat"], frame["bench"], schema=Schema(returns="strat", benchmark="bench")
    )
    direct = ss.compute_benchmark(r, b)
    assert np.isclose(mapped["beta"].value, direct["beta"].value)


def test_returns_input_schema_argument():
    rng = np.random.default_rng(42)
    r = rng.normal(0.10 / 252, 0.20 / np.sqrt(252), size=504)
    frame = pd.DataFrame({"pct": r})
    inp = ss.ReturnsInput(frame, periods_per_year=252, schema=Schema(returns="pct"))
    assert inp.values.shape == (504, 1)


# ----------------------------------------------------------------------
# Session default
# ----------------------------------------------------------------------


def test_session_schema_applies_to_subsequent_calls():
    df = _trade_df()
    set_schema(Schema(trades={"pnl": "profit", "side": "direction", "duration": "bars_held"}))
    assert get_schema() is not None
    mapped = ss.compute_trades(df)
    canonical = pd.DataFrame(
        {"pnl": df["profit"], "side": df["direction"], "duration": df["bars_held"]}
    )
    direct = ss.compute_trades(canonical)
    assert len(mapped.results) == len(direct.results)


def test_per_call_schema_overrides_session_default():
    df = _trade_df()
    set_schema(Schema(trades={"pnl": "profit"}))
    # a narrower per-call mapping overlays, not replaces
    result = ss.compute_trades(df, schema=Schema(trades={"side": "direction"}))
    assert len(result.results) > 0


def test_set_schema_none_clears():
    set_schema(Schema(trades={"pnl": "profit"}))
    set_schema(None)
    assert get_schema() is None


def test_set_schema_accepts_mapping():
    set_schema({"trades": {"pnl": "profit"}})
    assert get_schema().trades == {"pnl": "profit"}


# ----------------------------------------------------------------------
# Error cases
# ----------------------------------------------------------------------


def test_schema_and_columns_both_raises():
    df = _trade_df()
    with pytest.raises(TypeError, match="not both"):
        ss.compute_trades(df, schema=Schema(trades={"pnl": "profit"}), columns={"pnl": "profit"})


def test_compute_routes_columns_through_build_input():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"pct": rng.normal(size=504)})
    result = ss.compute(frame, "omega_ratio", columns={"returns": "pct"})
    assert np.isfinite(result.value)


# ----------------------------------------------------------------------
# describe_columns
# ----------------------------------------------------------------------


def test_describe_columns_reports_recognized_ignored_missing():
    df = pd.DataFrame({"pnl": [1.0, -1.0], "extra": [1, 2], "direction": [1, -1]})
    report = ss.describe_columns(df)
    assert report["recognized"] == ["pnl"]
    assert report["ignored"] == ["direction", "extra"]
    assert "side" in report["missing"]


def test_describe_columns_missing_side_lists_side_metrics():
    report = ss.describe_columns({"pnl": [1.0, -1.0]})
    metrics = report["missing"]["side"]
    assert "win_rate_long" in metrics
    assert "long_short_avg_duration" in metrics
    assert len(metrics) == 12


def test_describe_columns_missing_pnl_lists_all_trade_metrics():
    report = ss.describe_columns({"side": [1, -1]})
    assert "pnl" in report["missing"]
    metrics = report["missing"]["pnl"]
    assert "win_rate" in metrics
    assert len(metrics) > 10  # the full trade tier, not a fixed few


def test_describe_columns_schema_maps_aliases():
    df = pd.DataFrame(
        {"profit": [1.0, -1.0], "direction": [1, -1], "bars_held": [2, 3]}
    )
    report = ss.describe_columns(
        df, schema={"pnl": "profit", "side": "direction", "duration": "bars_held"}
    )
    assert set(report["recognized"]) == {"pnl", "side", "duration"}
    assert report["ignored"] == []
    assert "side" not in report["missing"]
    assert "duration" not in report["missing"]


def test_describe_columns_duration_derived_from_entry_exit():
    report = ss.describe_columns({"pnl": [1.0], "entry_time": [0.0], "exit_time": [2.0]})
    assert "duration" in report["recognized"]
    assert "duration" not in report["missing"]


def test_describe_columns_rejects_unsupported_input():
    with pytest.raises(TypeError, match="DataFrame"):
        ss.describe_columns([1.0, -1.0])
