from pathlib import Path

import pytest

from poker_vision.logic.invariants import InvariantViolationError
from poker_vision.logic.replay import run_replay_scenario

SCENARIO_FILES = [
    "01_all_folds.yaml",
    "02_raise_and_call.yaml",
    "03_preflop_3bet.yaml",
    "04_side_pot_all_in.yaml",
    "05_full_showdown.yaml",
]


@pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
def test_replay_scenarios_run_without_invariant_violations(scenario_file: str) -> None:
    scenario_path = Path("tests") / "scenarios" / scenario_file

    try:
        result = run_replay_scenario(scenario_path)
    except InvariantViolationError as exc:
        pytest.fail(f"InvariantViolationError in {scenario_file}: {exc}")

    assert result.needs_review is False
