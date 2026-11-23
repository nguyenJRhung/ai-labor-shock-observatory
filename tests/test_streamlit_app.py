"""Headless Streamlit AppTest verifying all 3 observatory tabs render cleanly without errors."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_streamlit_app_renders_cleanly_across_three_tabs() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exceptions: {[e.value for e in at.exception]}"
    assert not at.error, f"App displayed errors: {[e.value for e in at.error]}"
    assert len(at.tabs) == 3
    labels = [t.label for t in at.tabs]
    assert "Macro AI Shock Observatory (LMICs)" in labels[0]
    assert "Vietnam & Gender Deep-Dive" in labels[1]
    assert "Enterprise Reskilling ROI Simulator" in labels[2]
