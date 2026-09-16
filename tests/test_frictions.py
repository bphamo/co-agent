"""The friction sensitivity study's verdict, which is the part that gets quoted."""

from __future__ import annotations

from co_agent.cycle.frictions import Row, format_rows, verdict


def row(gap: float, label: str = "x", truncated: bool = False) -> Row:
    # strategy - buy_hold == gap, with the benchmark held fixed.
    return Row(label, 10.0, 1.0, 0.10 + gap, 0.10, 4, False, truncated)


def test_a_gap_negative_everywhere_is_not_a_friction_artifact():
    text = verdict([row(-0.10), row(-0.05), row(-0.02)])
    assert "negative at every setting" in text
    assert "does not depend on what real fills turn out to cost" in text


def test_a_gap_positive_everywhere_survives_the_punitive_end():
    assert "positive at every setting" in verdict([row(0.02), row(0.09)])


def test_a_sign_change_inside_the_grid_is_called_an_artifact():
    """The failure mode the study exists to catch."""
    text = verdict([row(-0.04, "slippage 0bps"), row(0.03, "commission free")])
    assert "CHANGES SIGN" in text
    assert "artifact of the cost assumption" in text
    assert "commission free" in text


def test_the_default_row_is_marked_so_the_quoted_number_is_identifiable():
    default = Row("commission x1", 10.0, 1.0, 0.05, 0.10, 4, True)
    assert "<- default" in format_rows([default])


def test_a_truncated_benchmark_is_flagged_rather_than_compared_silently():
    """A benchmark that ran out of cash is no longer 'owning the universe'."""
    assert "ran out of cash" in format_rows([row(-0.1, "commission x4", truncated=True)])
    assert "ran out of cash" not in format_rows([row(-0.1)])
