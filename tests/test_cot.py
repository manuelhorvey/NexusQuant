"""
Tests for the COT positioning loader (src.model.cot): graceful offline
behavior, CSV parsing, currency mapping, and symbol lookup.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.model.cot import (
    ALL_MARKETS,
    CCY_MAP,
    SYMBOL_MARKET,
    _cache_fresh,
    _cache_summary,
    cot_for_symbol,
    cot_status,
    expanding_percentile,
    fetch_cot_raw,
    load_cot,
    percentile_series,
    update_cot,
)


def _write_cot(tmp: Path, rows: str, name: str = "EUR_cot.csv") -> Path:
    d = tmp / "cot"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(rows)
    return d


def _canned_rows():
    """Canned CFTC-style JSON rows for 2 currencies + a cross-rate market
    that must be excluded by the exact-name match. > 52 weeks so the
    MIN_ROWS guard passes."""
    rows = []
    n = 60
    # Weekly report dates (Tuesdays) - strictly increasing with i.
    dates = [
        f"{d.date().isoformat()}T00:00:00.000"
        for d in pd.date_range("2024-01-02", periods=n, freq="W-TUE")
    ]
    for i in range(n):
        # EUR net rises 50 -> 640 (monotonic-ish: +10 each week).
        rows.append(
            {
                "report_date_as_yyyy_mm_dd": dates[i],
                "contract_market_name": "EURO FX",
                "futonly_or_combined": "FutOnly",
                "noncomm_positions_long_all": 100 + 10 * i,
                "noncomm_positions_short_all": 50 - 10 * i,
                "open_interest_all": 1000,
            }
        )
        rows.append(
            {
                "report_date_as_yyyy_mm_dd": dates[i],
                "contract_market_name": "BRITISH POUND",
                "futonly_or_combined": "FutOnly",
                "noncomm_positions_long_all": 500 - i,
                "noncomm_positions_short_all": 400,
                "open_interest_all": 2000,
            }
        )
    # A cross-rate market that must NOT match the exact "EURO FX" name.
    rows.append(
        {
            "report_date_as_yyyy_mm_dd": dates[0],
            "contract_market_name": "EURO FX/JAPANESE YEN XRATE",
            "futonly_or_combined": "FutOnly",
            "noncomm_positions_long_all": 999,
            "noncomm_positions_short_all": 1,
            "open_interest_all": 100,
        }
    )
    return rows


class TestPercentiles(unittest.TestCase):
    def test_expanding_percentile_causal_and_bounds(self):
        # Strictly increasing series -> each new point is the new max (100),
        # and early points are unaffected by later values.
        x = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        pct = expanding_percentile(x)
        self.assertEqual(float(pct.iloc[0]), 100.0)
        self.assertEqual(float(pct.iloc[-1]), 100.0)
        # A flat prefix keeps the first point at 100 (only itself in window).
        x2 = pd.Series([5.0, 5.0, 5.0, 100.0])
        pct2 = expanding_percentile(x2)
        self.assertEqual(float(pct2.iloc[1]), 100.0)  # ties count as <=
        self.assertEqual(float(pct2.iloc[-1]), 100.0)
        self.assertTrue(((pct2 >= 0) & (pct2 <= 100)).all())

    def test_expanding_percentile_no_future_leak(self):
        # A spike later must not change the early ranks.
        x1 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        x2 = pd.Series([10.0, 20.0, 30.0, 40.0, 500.0])
        self.assertTrue(
            expanding_percentile(x1).iloc[:4].equals(expanding_percentile(x2).iloc[:4])
        )


class TestFetch(unittest.TestCase):
    def test_fetch_parses_net_and_excludes_crosses(self):
        from unittest.mock import patch
        import src.model.cot as cotmod

        with patch.object(cotmod, "_fetch_json", return_value=_canned_rows()):
            raw = fetch_cot_raw()
        self.assertIsNotNone(raw)
        self.assertEqual(sorted(raw.keys()), ["EUR", "GBP"])
        eur = raw["EUR"]
        self.assertEqual(len(eur), 60)
        # net = long - short, rising by 20/week from 50 to 1230.
        self.assertAlmostEqual(float(eur["net"].iloc[0]), 50.0)
        self.assertAlmostEqual(float(eur["net"].iloc[-1]), 1230.0)

    def test_fetch_failure_returns_none(self):
        from unittest.mock import patch
        import src.model.cot as cotmod

        with patch.object(cotmod, "_fetch_json", side_effect=OSError("down")):
            self.assertIsNone(fetch_cot_raw())

    def test_fetch_requires_min_rows(self):
        from unittest.mock import patch
        import src.model.cot as cotmod

        rows = _canned_rows()[:8]  # only 4 EUR weeks -> below MIN_ROWS
        with patch.object(cotmod, "_fetch_json", return_value=rows):
            self.assertIsNone(fetch_cot_raw())

    def test_percentile_series_on_canned(self):
        from unittest.mock import patch
        import src.model.cot as cotmod

        with patch.object(cotmod, "_fetch_json", return_value=_canned_rows()):
            raw = fetch_cot_raw()
        s = percentile_series(raw["EUR"])
        self.assertEqual(len(s), 60)
        # Monotonic net -> every new week is a fresh high -> percentile 100.
        self.assertTrue((s == 100.0).all())
        self.assertTrue(s.index.is_monotonic_increasing)


class TestCache(unittest.TestCase):
    def _write_all(self, tmp, n_days_ago):
        d = Path(tmp) / "cot"
        d.mkdir(parents=True, exist_ok=True)
        last = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days_ago)
        for key in ALL_MARKETS:
            dates = pd.date_range(
                last - pd.Timedelta(days=700), periods=100, freq="W-TUE"
            )
            pcts = np.arange(100.0) % 101
            pd.DataFrame(
                {"date": dates.strftime("%Y-%m-%d"), "percentile": pcts[:100]}
            ).to_csv(d / f"{key}_cot.csv", index=False)
        return d

    def test_cache_fresh_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._write_all(tmp, n_days_ago=5)
            self.assertTrue(_cache_fresh(d))
            self.assertFalse(_cache_fresh(d, max_stale_days=2))

    def test_cache_fresh_missing_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._write_all(tmp, n_days_ago=5)
            (d / "SP500_cot.csv").unlink()
            self.assertFalse(_cache_fresh(d))

    def test_update_cot_skips_when_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_all(tmp, n_days_ago=3)
            res = update_cot(tmp, force=False)
            self.assertFalse(res["fetched"])
            self.assertEqual(res["reason"], "cache fresh")

    def test_update_cot_fetches_when_missing(self):
        from unittest.mock import patch
        import src.model.cot as cotmod

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                cotmod, "fetch_cot_raw", return_value=_fetch_raw_for_tests()
            ):
                res = update_cot(tmp, force=True)
            self.assertTrue(res["fetched"])
            self.assertIn("EUR", res["currencies"])
            self.assertIn("SP500", res["currencies"])
            # The written CSVs must round-trip through the loader.
            cot = load_cot(f"{tmp}/cot")
            self.assertIsNotNone(cot)
            self.assertIn("EUR", cot)
            self.assertAlmostEqual(float(cot["EUR"].iloc[-1]), 100.0)
            # Fresh now (all markets present and current).
            self.assertTrue(cot_status(tmp)["fresh"])

    def test_symbol_market_resolution(self):
        # Non-FX symbols resolve to their positioning market through
        # SYMBOL_MARKET; cot_for_symbol serves the right series.
        with tempfile.TemporaryDirectory() as tmp:
            self._write_all(tmp, n_days_ago=3)
            sp = cot_for_symbol("US500", f"{tmp}/cot")
            self.assertIsNotNone(sp)
            self.assertEqual(len(sp), 100)
            # FX symbols still resolve via CCY_MAP.
            eur = cot_for_symbol("EURUSD", f"{tmp}/cot")
            self.assertIsNotNone(eur)
            self.assertTrue(eur.equals(cot_for_symbol("EURUSD", f"{tmp}/cot")))

    def test_update_cot_graceful_on_failure(self):
        from unittest.mock import patch
        import src.model.cot as cotmod

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cotmod, "fetch_cot_raw", return_value=None):
                res = update_cot(tmp, force=True)
            self.assertFalse(res["fetched"])
            self.assertIsNotNone(res["error"])
            self.assertEqual(res["currencies"], {})

    def test_update_cot_prunes_missing_currencies(self):
        # A partial fetch (only EUR) must remove the other currencies' stale
        # files so load_cot never serves stale data and freshness recovers.
        from unittest.mock import patch
        import src.model.cot as cotmod

        with tempfile.TemporaryDirectory() as tmp:
            self._write_all(tmp, n_days_ago=30)  # full set, but stale
            partial = _fetch_raw_for_tests()
            partial = {k: v for k, v in partial.items() if k == "EUR"}
            with patch.object(cotmod, "fetch_cot_raw", return_value=partial):
                res = update_cot(tmp, force=True)
            self.assertTrue(res["fetched"])
            d = Path(tmp) / "cot"
            remaining = sorted(p.name for p in d.glob("*_cot.csv"))
            self.assertEqual(remaining, ["EUR_cot.csv"])
            # load_cot reflects the pruned state.
            cot = load_cot(str(d))
            self.assertEqual(sorted(cot.keys()), ["EUR"])

    def test_cache_summary_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._write_all(tmp, n_days_ago=3)
            summary = _cache_summary(d)
            self.assertEqual(sorted(summary.keys()), sorted(ALL_MARKETS))
            for info in summary.values():
                self.assertIn("rows", info)
                self.assertIn("last_date", info)
                self.assertIn("last_percentile", info)

    def test_cot_status_missing_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = cot_status(tmp)  # empty dir -> nothing cached
            self.assertFalse(st["fresh"])
            self.assertEqual(sorted(st["missing"]), sorted(ALL_MARKETS))
            self._write_all(tmp, n_days_ago=3)
            st = cot_status(tmp)
            self.assertTrue(st["fresh"])
            self.assertEqual(st["missing"], [])


def _fetch_raw_for_tests():
    """Local stand-in for fetch_cot_raw (no network in unit tests); ends
    on the most recent Tuesday so the cache reads as fresh."""

    out = {}
    today = pd.Timestamp.now().normalize()
    # Walk back to the most recent Tuesday, then 100 weeks before that.
    end = today - pd.Timedelta(days=(today.dayofweek - 1) % 7)
    dates = pd.date_range(end - pd.Timedelta(weeks=100), periods=100, freq="W-TUE")
    for key in ALL_MARKETS:
        net = pd.Series(np.linspace(-500, 500, 100) + np.sin(np.arange(100)))
        out[key] = pd.DataFrame(
            {
                "date": dates,
                "net": net,
                "noncomm_positions_long_all": 0,
                "noncomm_positions_short_all": 0,
                "open_interest_all": 0,
            }
        )
    return out


class TestCOT(unittest.TestCase):
    def test_load_cot_none_when_missing(self):
        self.assertIsNone(load_cot("/nonexistent/cot/dir"))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_cot(tmp))

    def test_load_cot_parses_csv(self):
        rows = "date,percentile\n" + "\n".join(
            f"2024-01-{d:02d},{v}"
            for d, v in [
                (1, 20),
                (2, 25),
                (3, 30),
                (4, 35),
                (5, 40),
                (6, 45),
                (7, 50),
                (8, 55),
                (9, 60),
                (10, 65),
                (11, 70),
                (12, 75),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            d = _write_cot(Path(tmp), rows)
            cot = load_cot(str(d))
            self.assertIsNotNone(cot)
            self.assertIn("EUR", cot)
            s = cot["EUR"]
            self.assertEqual(len(s), 12)
            self.assertEqual(float(s.iloc[-1]), 75.0)
            # Sorted by date ascending.
            self.assertTrue(s.index.is_monotonic_increasing)

    def test_load_cot_multiple_currencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = "date,percentile\n" + "\n".join(
                f"2024-01-{d:02d},{v}"
                for d, v in [
                    (1, 10),
                    (2, 20),
                    (3, 30),
                    (4, 40),
                    (5, 50),
                    (6, 60),
                    (7, 70),
                    (8, 80),
                    (9, 90),
                    (10, 99),
                ]
            )
            d = _write_cot(Path(tmp), rows, "EUR_cot.csv")
            _write_cot(Path(tmp), rows, "JPY_cot.csv")
            cot = load_cot(str(d))
            self.assertEqual(sorted(cot.keys()), ["EUR", "JPY"])

    def test_ccy_map_covers_fx_universe(self):
        for sym in (
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "NZDUSD",
            "USDCAD",
            "XAUUSD",
            "EURJPY",
            "GBPJPY",
            "NZDJPY",
            "AUDJPY",
            "CADJPY",
        ):
            self.assertIn(sym, CCY_MAP, sym)

    def test_symbol_market_covers_new_instruments(self):
        for sym in ("XAGUSD", "US500", "USTEC", "US30", "XTIUSD", "USOIL", "XBRUSD"):
            self.assertIn(sym, SYMBOL_MARKET, sym)
        # Every SYMBOL_MARKET key must exist in ALL_MARKETS so the downloader
        # actually fetches it.
        for key in SYMBOL_MARKET.values():
            self.assertIn(key, ALL_MARKETS, key)
        # XAUUSD maps through the currency map (GOLD), not SYMBOL_MARKET.
        self.assertNotIn("XAUUSD", SYMBOL_MARKET)

    def test_cot_for_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = "date,percentile\n" + "\n".join(
                f"2024-01-{d:02d},{v}"
                for d, v in [
                    (1, 10),
                    (2, 20),
                    (3, 30),
                    (4, 40),
                    (5, 50),
                    (6, 60),
                    (7, 70),
                    (8, 80),
                    (9, 90),
                    (10, 99),
                ]
            )
            d = _write_cot(Path(tmp), rows)
            s = cot_for_symbol("EURUSD", str(d))
            self.assertIsNotNone(s)
            self.assertEqual(float(s.iloc[-1]), 99.0)
            # Unknown currency / missing data -> None.
            self.assertIsNone(cot_for_symbol("ZZZUSD", str(d)))


if __name__ == "__main__":
    unittest.main()
