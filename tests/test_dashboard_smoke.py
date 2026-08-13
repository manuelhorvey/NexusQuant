"""
Smoke tests: the Streamlit dashboard and helper scripts must stay
parseable/importable. The unit suite never exercised dashboard.py, which is
how a syntax error (indentation broken during a divergence-formatting edit)
slipped through a fully green run. These tests close that gap cheaply.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent


class TestParseable(unittest.TestCase):
    def test_dashboard_parses(self):
        ast.parse((ROOT / "dashboard.py").read_text(encoding="utf-8"))

    def test_scripts_parse(self):
        # Every helper under scripts/ (incl. run_watchlist.py) must stay
        # parseable too.
        scripts = ROOT / "scripts"
        if scripts.is_dir():
            for f in sorted(scripts.glob("*.py")):
                ast.parse(f.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
