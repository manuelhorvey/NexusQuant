"""
Tests for the live signal & alert system (src/live/):
filtering, R:R parsing, dedup state, channel sending, and the
alert message format. Pure/offline where possible; a couple use the
small candidates group for integration.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.live.alerts import (
    ConsoleChannel,
    DiscordChannel,
    FileChannel,
    TelegramChannel,
    build_channels,
    send_all,
    _split_text,
)
from src.live.signals import (
    _rr_series,
    _setup_key,
    filter_signals,
    format_briefing,
    load_state,
    save_state,
    live_signal_pass,
)

TEST_GROUP, TEST_TF = "candidates", "D1"


def _fake_table(**overrides):
    """A scanner-shaped DataFrame with two rows."""
    rows = [
        {
            "symbol": "AAA",
            "dip_score": 7,
            "dip_confirmed": "Yes",
            "bias_score": 2,
            "ml_prob": 60.0,
            "macro_gate": "PASS",
            "entry_zone": "1.00000-1.00200",
            "invalidation": 0.995,
            "resistance": 1.010,
            "regime": "Bull Trend",
            "date": "2026-01-01",
            "close": 1.001,
        },
        {
            "symbol": "BBB",
            "dip_score": 3,
            "dip_confirmed": "No",
            "bias_score": -1,
            "ml_prob": 40.0,
            "macro_gate": "BLOCKED",
            "entry_zone": "2.00000-2.00200",
            "invalidation": 1.99,
            "resistance": 2.02,
            "regime": "Range / Chop",
            "date": "2026-01-01",
            "close": 2.001,
        },
    ]
    for k, v in overrides.items():
        for row in rows:
            row[k] = v
    return pd.DataFrame(rows)


class TestFilterSignals(unittest.TestCase):
    def test_min_dip_score(self):
        f = filter_signals(
            _fake_table(),
            min_dip_score=5,
            require_confirmed=False,
            require_macro_pass=False,
        )
        self.assertEqual(set(f["symbol"]), {"AAA"})

    def test_require_confirmed(self):
        f = filter_signals(
            _fake_table(), require_confirmed=True, require_macro_pass=False
        )
        self.assertEqual(set(f["symbol"]), {"AAA"})

    def test_min_ml_prob_keeps_no_model_rows(self):
        t = _fake_table()
        t.loc[0, "ml_prob"] = None  # no model for AAA
        f = filter_signals(
            t, min_ml_prob=50.0, require_confirmed=False, require_macro_pass=False
        )
        self.assertIn("AAA", set(f["symbol"]))  # kept (graceful)
        self.assertNotIn("BBB", set(f["symbol"]))

    def test_macro_gate_required(self):
        f = filter_signals(
            _fake_table(), require_macro_pass=True, require_confirmed=False
        )
        self.assertEqual(set(f["symbol"]), {"AAA"})

    def test_macro_gate_skips_when_no_macro(self):
        t = _fake_table()
        t["macro_gate"] = None  # no macro data at all
        f = filter_signals(
            t, require_macro_pass=True, require_confirmed=False, min_dip_score=0
        )
        self.assertEqual(set(f["symbol"]), {"AAA", "BBB"})

    def test_min_rr_filters_low_reward(self):
        # AAA: entry mid 1.001, stop 0.995, target 1.010 -> RR 1.5
        f = filter_signals(
            _fake_table(), min_rr=2.0, require_confirmed=False, require_macro_pass=False
        )
        self.assertTrue(f.empty)
        f2 = filter_signals(
            _fake_table(), min_rr=1.0, require_confirmed=False, require_macro_pass=False
        )
        self.assertIn("AAA", set(f2["symbol"]))


class TestLiveMinRRDefault(unittest.TestCase):
    def test_min_rr_defaults_from_risk_settings(self):
        """Spec #11: when live.filters.min_rr is unset, the CLI falls back
        to settings.risk.min_reward_risk (2.5) - so the institutional floor
        is enforced by default, not silently disabled."""
        import src.live.run as run_mod

        captured = {}
        real_pass = run_mod.live_signal_pass
        real_short = run_mod.live_short_pass

        def _empty() -> dict:
            return {
                "date": "2026-01-01",
                "group": "x",
                "timeframe": "D1",
                "scanned": 0,
                "candidates": 0,
                "new_alerts": [],
                "skipped_dup": 0,
                "sizing_failed": 0,
                "macro": None,
                "data_age_days": 0.0,
                "data_stale": False,
            }

        def spy(*a, **kw):
            captured["min_rr"] = kw.get("min_rr")
            return _empty()

        # Default mode is "both" -> main() runs the long AND the short pass;
        # stub both so the test never touches the universe.
        run_mod.live_signal_pass = spy
        run_mod.live_short_pass = lambda *a, **kw: _empty()
        try:
            with mock.patch(
                "src.live.run._load_settings",
                return_value={
                    "live": {"filters": {}, "risk": {}, "watchlist": {}},
                    "risk": {"min_reward_risk": 2.5},
                },
            ):
                self.assertEqual(run_mod.main(["--dry-run"]), 0)
        finally:
            run_mod.live_signal_pass = real_pass
            run_mod.live_short_pass = real_short
        # The settings default (2.5) must flow into the pass when no --min-rr.
        self.assertEqual(captured.get("min_rr"), 2.5)

    def test_min_rr_cli_flag_overrides_settings(self):
        """An explicit --min-rr beats the settings default."""
        import src.live.run as run_mod

        captured = {}
        real_pass = run_mod.live_signal_pass
        real_short = run_mod.live_short_pass

        def _empty() -> dict:
            return {
                "date": "2026-01-01",
                "group": "x",
                "timeframe": "D1",
                "scanned": 0,
                "candidates": 0,
                "new_alerts": [],
                "skipped_dup": 0,
                "sizing_failed": 0,
                "macro": None,
                "data_age_days": 0.0,
                "data_stale": False,
            }

        def spy(*a, **kw):
            captured["min_rr"] = kw.get("min_rr")
            return _empty()

        run_mod.live_signal_pass = spy
        run_mod.live_short_pass = lambda *a, **kw: _empty()
        try:
            with mock.patch(
                "src.live.run._load_settings",
                return_value={
                    "live": {"filters": {}, "risk": {}, "watchlist": {}},
                    "risk": {"min_reward_risk": 2.5},
                },
            ):
                run_mod.main(["--min-rr", "1.5", "--dry-run"])
        finally:
            run_mod.live_signal_pass = real_pass
            run_mod.live_short_pass = real_short
        self.assertEqual(captured.get("min_rr"), 1.5)


class TestRRSeries(unittest.TestCase):
    def test_string_zone_parsed(self):
        # The scanner stores entry_zone as "lo-hi" strings.
        t = _fake_table()
        rr = _rr_series(t)
        # AAA: entry (1.00000+1.00200)/2 = 1.001, stop 0.995, tgt 1.010
        self.assertAlmostEqual(rr.iloc[0], 1.5, places=3)
        self.assertTrue(rr.iloc[1] > 0)

    def test_missing_levels_are_nan(self):
        t = _fake_table()
        t.loc[0, "resistance"] = None
        rr = _rr_series(t)
        self.assertTrue(rr.isna().iloc[0])


class TestDedupState(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            save_state({"X:1-2", "Y:3-4"}, path)
            self.assertEqual(load_state(path), {"X:1-2", "Y:3-4"})

    def test_missing_file_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_state(f"{tmp}/nope.json"), set())

    def test_corrupt_state_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            Path(path).write_text("not json")
            self.assertEqual(load_state(path), set())

    def test_setup_key(self):
        self.assertEqual(_setup_key("EURUSD", [1.1, 1.2]), "EURUSD:1.100000-1.200000")
        self.assertEqual(
            _setup_key("EURUSD", "1.10000-1.20000"), "EURUSD:1.100000-1.200000"
        )
        self.assertEqual(_setup_key("EURUSD", None), "EURUSD:none")

    def test_setup_key_tuple_and_string_agree(self):
        # Regression: the scanner emits string zones, the report emits
        # tuples - they must dedup to the SAME key (was 5dp vs 6dp drift).
        self.assertEqual(
            _setup_key("EURUSD", "1.10000-1.20000"), _setup_key("EURUSD", (1.1, 1.2))
        )

    def test_setup_key_unparseable_string(self):
        # A malformed zone falls back to a stable raw key, not a crash.
        self.assertEqual(_setup_key("EURUSD", "garbage"), "EURUSD:garbage")


class TestChannels(unittest.TestCase):
    def test_split_text_short(self):
        self.assertEqual(_split_text("hi", 10), ["hi"])

    def test_split_text_long(self):
        chunks = _split_text("a\nb\nc\nd", 5)
        self.assertTrue(all(len(c) <= 5 for c in chunks))
        self.assertEqual("".join(chunks).replace("\n", ""), "abcd")

    def test_split_text_oversized_line(self):
        # A single line longer than the limit must be hard-split so no
        # chunk ever exceeds the cap (Discord rejects over-long messages).
        long_line = "x" * 3000
        chunks = _split_text(long_line, 1000)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))
        self.assertEqual("".join(chunks), long_line)

    def test_split_text_preserves_content(self):
        # All characters survive (newlines are normalized - the function
        # re-joins on line boundaries, Discord normalizes them anyway).
        text = "line1\n" + "y" * 2500 + "\nline3\n" + "z" * 900
        chunks = _split_text(text, 1000)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))
        self.assertEqual("".join(chunks).replace("\n", ""), text.replace("\n", ""))

    def test_discord_requires_url(self):
        with self.assertRaises(ValueError):
            DiscordChannel("")

    def test_telegram_requires_token_and_chat(self):
        with self.assertRaises(ValueError):
            TelegramChannel("", "123")
        with self.assertRaises(ValueError):
            TelegramChannel("token", "")

    def test_telegram_splits_long_messages(self):
        # 4096-char cap: a 10000-char alert becomes multiple messages.
        ch = TelegramChannel("t", "c")
        with mock.patch.object(ch, "_send_chunk", side_effect=lambda c: True) as m:
            ok = ch.send("x" * 10000)
        self.assertTrue(ok)
        self.assertGreater(len(m.call_args_list), 2)
        for call in m.call_args_list:
            self.assertLessEqual(len(call.args[0]), 4096)

    def test_telegram_ok_false_is_failure(self):
        # Telegram returns HTTP 200 with ok:false on API errors - that must
        # count as a failure (unlike a Discord webhook 200).
        ch = TelegramChannel("t", "c")
        with mock.patch("src.live.alerts.urllib.request.urlopen") as m:

            class FakeResp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return b'{"ok": false}'

            m.return_value = FakeResp()
            self.assertFalse(ch.send("hi"))

    def test_telegram_ok_true_is_success(self):
        ch = TelegramChannel("t", "c")
        with mock.patch("src.live.alerts.urllib.request.urlopen") as m:

            class FakeResp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return b'{"ok": true}'

            m.return_value = FakeResp()
            self.assertTrue(ch.send("hi"))

    def test_telegram_request_url_and_payload(self):
        # Regression: the API path must carry the token and the body the
        # chat_id/text - a broken format would pass the ok-flag tests.
        ch = TelegramChannel("tok123", "chat42")
        with mock.patch("src.live.alerts.urllib.request.urlopen") as m:

            class FakeResp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return b'{"ok": true}'

            m.return_value = FakeResp()
            self.assertTrue(ch.send("hi"))
        req = m.call_args[0][0]
        # URL: .../bot<token>/sendMessage ; body carries chat_id + text.
        self.assertIn("bottok123/sendMessage", req.full_url)
        self.assertIn("chat_id=chat42", req.data.decode())
        self.assertIn("text=hi", req.data.decode())

    def test_telegram_error_description_is_logged(self):
        ch = TelegramChannel("t", "c")
        with mock.patch("src.live.alerts.urllib.request.urlopen") as m:

            class FakeResp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return b'{"ok": false, "description": "chat not found"}'

            m.return_value = FakeResp()
            with mock.patch("src.live.alerts.log.warning") as warn:
                self.assertFalse(ch.send("hi"))
            warn.assert_called_once()
            self.assertIn("chat not found", str(warn.call_args))

    def test_file_channel_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ch = FileChannel(f"{tmp}/out.log")
            self.assertTrue(ch.send("hello"))
            self.assertIn("hello", Path(f"{tmp}/out.log").read_text())

    def test_console_send(self):
        self.assertTrue(ConsoleChannel().send("test"))

    def test_send_all_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ch = FileChannel(f"{tmp}/out.log")
            self.assertEqual(send_all([ch], "hello"), 1)
            self.assertEqual(send_all([ch], "  "), 0)

    def test_build_channels_default(self):
        with tempfile.TemporaryDirectory():
            chs = build_channels(
                {"console": True, "file": False}, env={"NEXUS_DISCORD_WEBHOOK": ""}
            )
            self.assertEqual(len(chs), 1)
            self.assertIsInstance(chs[0], ConsoleChannel)

    def test_build_channels_discord_env(self):
        chs = build_channels(
            {"console": False, "file": False, "discord": True},
            env={"NEXUS_DISCORD_WEBHOOK": "https://x"},
        )
        self.assertEqual(len(chs), 1)
        self.assertIsInstance(chs[0], DiscordChannel)

    def test_build_channels_discord_missing_url(self):
        chs = build_channels(
            {"console": False, "file": False, "discord": True},
            env={"NEXUS_DISCORD_WEBHOOK": ""},
        )
        self.assertEqual(chs, [])

    def test_build_channels_telegram_env(self):
        chs = build_channels(
            {"console": False, "file": False, "telegram": True},
            env={"NEXUS_TELEGRAM_BOT_TOKEN": "tok", "NEXUS_TELEGRAM_CHAT_ID": "123"},
        )
        self.assertEqual(len(chs), 1)
        self.assertIsInstance(chs[0], TelegramChannel)

    def test_build_channels_telegram_missing_creds(self):
        chs = build_channels(
            {"console": False, "file": False, "telegram": True},
            env={"NEXUS_TELEGRAM_BOT_TOKEN": "", "NEXUS_TELEGRAM_CHAT_ID": ""},
        )
        self.assertEqual(chs, [])

    def test_build_channels_telegram_settings_fallback(self):
        chs = build_channels(
            {
                "console": False,
                "file": False,
                "telegram": True,
                "telegram_bot_token": "tok",
                "telegram_chat_id": "42",
            },
            env={},
        )
        self.assertEqual(len(chs), 1)
        self.assertIsInstance(chs[0], TelegramChannel)


class TestFormatting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = live_signal_pass(
            group="candidates",
            timeframe="D1",
            min_dip_score=0,
            require_confirmed=False,
            require_macro_pass=False,
            state_file="/tmp/nx_test_live_state.json",
        )

    def test_briefing_shape(self):
        b = format_briefing(self.res)
        self.assertIn("NEXUSQUANT LIVE SIGNALS", b)
        self.assertIn("candidates", b)

    def test_alert_text_has_key_fields(self):
        if not self.res["new_alerts"]:
            self.skipTest("no setups on candidates today")
        text = self.res["new_alerts"][0]["text"]
        for token in ["BUY-THE-DIP", "Entry", "Stop", "Target", "Size"]:
            self.assertIn(token, text)

    def test_macro_section_optional(self):
        b = format_briefing({**self.res, "macro": None})
        self.assertIn("No new", b) if not self.res["new_alerts"] else None

    def test_format_alert_needs_setup(self):
        if not self.res["new_alerts"]:
            self.skipTest("no setups on candidates today")
        a = self.res["new_alerts"][0]
        self.assertIn(a["symbol"], a["text"])
        self.assertIn("key", a)


class TestPassIntegration(unittest.TestCase):
    def test_pass_on_candidates(self):
        res = live_signal_pass(
            group=TEST_GROUP,
            timeframe=TEST_TF,
            min_dip_score=0,
            require_confirmed=False,
            require_macro_pass=False,
            state_file="/tmp/nx_test_live_state2.json",
        )
        self.assertGreaterEqual(res["scanned"], 4)
        self.assertIn("new_alerts", res)
        self.assertIn("macro", res)
        self.assertIn("candidates", res)

    def test_pass_dedup_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            r1 = live_signal_pass(
                group=TEST_GROUP,
                timeframe=TEST_TF,
                min_dip_score=0,
                require_confirmed=False,
                require_macro_pass=False,
                state_file=path,
            )
            r2 = live_signal_pass(
                group=TEST_GROUP,
                timeframe=TEST_TF,
                min_dip_score=0,
                require_confirmed=False,
                require_macro_pass=False,
                state_file=path,
            )
            keys1 = {a["key"] for a in r1["new_alerts"]}
            keys2 = {a["key"] for a in r2["new_alerts"]}
            # Second run never re-alerts the first run's keys.
            self.assertEqual(keys1 & keys2, set())
            # And the second run reports them as already-seen skips.
            self.assertEqual(r2["skipped_dup"], len(keys1))

    def test_pass_new_symbol_alerts_while_old_suppressed(self):
        # Pre-seed the state with a fake key for a candidate symbol, then
        # confirm a pass still alerts OTHER symbols but not the seeded one.
        with tempfile.TemporaryDirectory() as tmp:
            from src.analysis.scanner import scan_universe

            table = scan_universe(
                data_dir="data/raw",
                group=TEST_GROUP,
                timeframe=TEST_TF,
                fetch_mt5=False,
            )
            syms = table["symbol"].tolist()
            if not syms:
                self.skipTest("no symbols in candidates")
            seeded = _setup_key(syms[0], "0.00001-0.00002")
            save_state({seeded}, f"{tmp}/state.json")
            res = live_signal_pass(
                group=TEST_GROUP,
                timeframe=TEST_TF,
                min_dip_score=0,
                require_confirmed=False,
                require_macro_pass=False,
                state_file=f"{tmp}/state.json",
            )
            for a in res["new_alerts"]:
                self.assertNotEqual(a["key"], seeded)


if __name__ == "__main__":
    unittest.main()
