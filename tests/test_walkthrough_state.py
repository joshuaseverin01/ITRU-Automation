"""Tests for first-time walkthrough session-state helpers."""

from __future__ import annotations

import unittest

from app import (
    ISO_METRIC_KEY,
    ISO_PREVIOUS_TIME_VIEW_MODE_KEY,
    WALKTHROUGH_STATE_KEY,
    _apply_animation_metric_default,
    _close_walkthrough_and_rerun,
    _dismiss_walkthrough,
    _ensure_walkthrough_state,
    _reopen_walkthrough,
)


class WalkthroughStateTests(unittest.TestCase):
    def test_walkthrough_defaults_to_visible(self) -> None:
        session_state: dict[str, object] = {}

        visible = _ensure_walkthrough_state(session_state)

        self.assertTrue(visible)
        self.assertTrue(session_state[WALKTHROUGH_STATE_KEY])

    def test_walkthrough_dismissal_sets_flag_false(self) -> None:
        session_state: dict[str, object] = {WALKTHROUGH_STATE_KEY: True}

        _dismiss_walkthrough(session_state)

        self.assertFalse(session_state[WALKTHROUGH_STATE_KEY])
        self.assertFalse(_ensure_walkthrough_state(session_state))

    def test_walkthrough_close_sets_flag_false_and_reruns(self) -> None:
        session_state: dict[str, object] = {WALKTHROUGH_STATE_KEY: True}
        rerun_calls: list[bool] = []

        _close_walkthrough_and_rerun(session_state, lambda: rerun_calls.append(True))

        self.assertFalse(session_state[WALKTHROUGH_STATE_KEY])
        self.assertEqual(rerun_calls, [True])

    def test_walkthrough_reopen_sets_flag_true(self) -> None:
        session_state: dict[str, object] = {WALKTHROUGH_STATE_KEY: False}

        _reopen_walkthrough(session_state)

        self.assertTrue(session_state[WALKTHROUGH_STATE_KEY])
        self.assertTrue(_ensure_walkthrough_state(session_state))

    def test_animation_mode_defaults_metric_to_cumulative_on_entry(self) -> None:
        session_state: dict[str, object] = {
            ISO_PREVIOUS_TIME_VIEW_MODE_KEY: "Snapshot",
            ISO_METRIC_KEY: "Monthly Revenue",
        }

        _apply_animation_metric_default(
            session_state,
            current_mode="Animation",
            metric_options=["Monthly Revenue", "Cumulative Revenue", "Revenue per kW"],
        )

        self.assertEqual(session_state[ISO_METRIC_KEY], "Cumulative Revenue")
        self.assertEqual(session_state[ISO_PREVIOUS_TIME_VIEW_MODE_KEY], "Animation")

    def test_animation_mode_manual_metric_override_persists_while_in_animation(self) -> None:
        session_state: dict[str, object] = {
            ISO_PREVIOUS_TIME_VIEW_MODE_KEY: "Animation",
            ISO_METRIC_KEY: "Revenue per kW",
        }

        _apply_animation_metric_default(
            session_state,
            current_mode="Animation",
            metric_options=["Monthly Revenue", "Cumulative Revenue", "Revenue per kW"],
        )

        self.assertEqual(session_state[ISO_METRIC_KEY], "Revenue per kW")

    def test_non_animation_mode_does_not_force_metric(self) -> None:
        session_state: dict[str, object] = {
            ISO_PREVIOUS_TIME_VIEW_MODE_KEY: "Animation",
            ISO_METRIC_KEY: "Monthly Revenue",
        }

        _apply_animation_metric_default(
            session_state,
            current_mode="Snapshot",
            metric_options=["Monthly Revenue", "Cumulative Revenue"],
        )

        self.assertEqual(session_state[ISO_METRIC_KEY], "Monthly Revenue")
        self.assertEqual(session_state[ISO_PREVIOUS_TIME_VIEW_MODE_KEY], "Snapshot")


if __name__ == "__main__":
    unittest.main()
