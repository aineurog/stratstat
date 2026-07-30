"""Tests enforcing the core/report dependency boundary.

Per design principle 2.9: core must never import from report, matplotlib,
or any heavy visualization dependency. This test fails if that boundary
is violated.
"""

import sys


def test_core_does_not_import_matplotlib():
    """core must not import matplotlib directly or transitively."""
    # If matplotlib is already imported, remove it so we can detect a fresh import
    sys.modules.pop("matplotlib", None)

    # Check that importing core submodules does not pull in matplotlib
    from stratstat.core import _utils  # noqa: F401
    from stratstat.core.returns import (
        descriptive,  # noqa: F401
        inference,  # noqa: F401
        risk,  # noqa: F401
        risk_adjusted,  # noqa: F401
    )

    assert "matplotlib" not in sys.modules, (
        "core module imported matplotlib — this violates the core/report boundary"
    )


def test_core_does_not_import_plotly():
    """core must not import plotly."""
    sys.modules.pop("plotly", None)

    import stratstat.core  # noqa: F401
    import stratstat.core.returns  # noqa: F401

    assert "plotly" not in sys.modules, (
        "core module imported plotly — this violates the core/report boundary"
    )


def test_core_does_not_import_report():
    """core must not import anything from stratstat.report."""
    sys.modules.pop("stratstat.report", None)

    import stratstat.core  # noqa: F401
    import stratstat.core.returns  # noqa: F401

    assert "stratstat.report" not in sys.modules, (
        "core module imported stratstat.report — this violates the core/report boundary"
    )
