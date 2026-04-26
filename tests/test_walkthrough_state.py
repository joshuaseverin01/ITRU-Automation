"""Tests for first-time walkthrough session-state helpers."""

from __future__ import annotations

import unittest

from app import (
    WALKTHROUGH_STATE_KEY,
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

    def test_walkthrough_reopen_sets_flag_true(self) -> None:
        session_state: dict[str, object] = {WALKTHROUGH_STATE_KEY: False}

        _reopen_walkthrough(session_state)

        self.assertTrue(session_state[WALKTHROUGH_STATE_KEY])
        self.assertTrue(_ensure_walkthrough_state(session_state))


if __name__ == "__main__":
    unittest.main()
