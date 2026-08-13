"""
Tests for the ensemble model package: label logic, feature causality,
chronological split integrity, and model save/load/predict roundtrip.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from tests.test_dip import make_downtrend, make_uptrend_dip, with_indicators
from src.features.regime import detect_regime

from src.model.features import (
    CROSS_FEATURES,
    FEATURE_COLUMNS,
    MACRO_FEATURES,
    MTF_FEATURES,
    TIME_FEATURES,
    build_dataset,
    build_features,
    build_labels,
    build_meta_labels,
    dip_signal_series,
)
from src.model.model import (
    _cross_proxies,
    apply_calibrator,
    decile_report,
    dip_filter_gate,
    evaluate,
    feature_importance,
    fit_calibrator,
    load_model,
    predict_series,
    save_model,
    top_decile_lift,
    train_model,
)
from src.model.run import (
    _chrono_val_split,
    _group_model_path,
    _symbol_groups,
    search_hyperparams,
    split_chronological,
    stack_oos,
    walk_forward,
)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return detect_regime(with_indicators(df))


class TestLabels(unittest.TestCase):
    def test_uptrend_labels_mostly_wins(self):
        df = _prep(make_uptrend_dip())
        labels = build_labels(df, horizon=10)
        y = labels["label"].dropna()
        self.assertGreater(y.mean(), 0.5, "uptrend should be mostly 1R wins")

    def test_downtrend_labels_mostly_losses(self):
        df = _prep(make_downtrend())
        labels = build_labels(df, horizon=10)
        y = labels["label"].dropna()
        self.assertLess(y.mean(), 0.5, "downtrend should be mostly losses")

    def test_last_bars_have_no_label(self):
        df = _prep(make_uptrend_dip())
        labels = build_labels(df, horizon=10)
        self.assertTrue(labels["label"].iloc[-10:].isna().any())
        self.assertEqual(labels["label"].notna().sum() <= len(df), True)

    def test_label_values_are_01(self):
        df = _prep(make_uptrend_dip())
        y = build_labels(df, horizon=10)["label"].dropna()
        self.assertTrue(set(y.unique()).issubset({0.0, 1.0}))

    def test_stop_checked_before_target(self):
        # A bar followed by a big down move then up: conservative stop-first
        # rule means the label must be a loss if the stop was touched.
        prices = [100.0] * 60 + [100, 99, 98, 97, 96, 95, 94, 110, 111, 112]
        idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
        close = pd.Series(prices, index=idx)
        open_ = close.shift(1).fillna(close.iloc[0])
        high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
        low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": 1000}
        )
        df = _prep(df)
        labels = build_labels(df, horizon=10)
        atr = df["atr_14"]
        # Find the first bar whose forward low breaches close - ATR
        y = labels["label"]
        idx_atr = atr.dropna().index
        self.assertEqual(y.loc[idx_atr[0]], 0.0)


class TestFeatures(unittest.TestCase):
    def test_feature_columns_complete(self):
        df = _prep(make_uptrend_dip())
        X = build_features(df)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        # After the indicator warm-up (~200 bars for SMA200) no NaNs remain.
        self.assertTrue(
            X.tail(20).notna().all().all(), "features should have no NaN after warm-up"
        )

    def test_time_features_present_and_bounded(self):
        df = _prep(make_uptrend_dip())
        X = build_features(df)
        self.assertTrue(set(TIME_FEATURES).issubset(X.columns))
        self.assertTrue(X["day_of_week"].between(0, 6).all())
        self.assertTrue(X["month"].between(1, 12).all())
        self.assertTrue(set(X["month_end"].unique()).issubset({0.0, 1.0}))
        self.assertTrue(set(X["mid_month"].unique()).issubset({0.0, 1.0}))

    def test_interaction_features_finite(self):
        df = _prep(make_uptrend_dip())
        X = build_features(df)
        # After the SMA200 warm-up (same window as the other feature tests).
        tail = X.tail(50)
        for col in ("vol_x_momentum", "vol_x_trend", "adx_x_slope"):
            self.assertTrue(np.isfinite(tail[col]).all(), col)

    def test_mtf_features_default_to_zero(self):
        df = _prep(make_uptrend_dip())
        X = build_features(df)
        for col in MTF_FEATURES:
            self.assertTrue((X[col] == 0.0).all(), col)

    def test_mtf_features_aligned_and_causal(self):
        # H4 frame: 300 days of 4 bars/day with a late trend regime change.
        df = _prep(make_uptrend_dip())
        base = df.index[0]
        h4_idx = pd.date_range(base, periods=1200, freq="6h")
        up = np.linspace(100.0, 200.0, 1200)
        h4 = pd.DataFrame(
            {
                "open": up,
                "high": up * 1.01,
                "low": up * 0.99,
                "close": up,
                "volume": 1000,
            },
            index=h4_idx,
        )
        X = build_features(df, mtf=h4)
        # Non-zero momentum after warm-up.
        self.assertTrue((X["h4_mom20"].iloc[40:] != 0.0).any())
        # Corrupting future H4 bars must not change early features.
        h4_bad = h4.copy()
        h4_bad.loc[h4_bad.index[600] :, "close"] *= 2.0
        X2 = build_features(df, mtf=h4_bad)
        self.assertTrue(X.iloc[:50][MTF_FEATURES].equals(X2.iloc[:50][MTF_FEATURES]))

    def test_cross_features_default_to_zero(self):
        df = _prep(make_uptrend_dip())
        X = build_features(df)
        for col in CROSS_FEATURES:
            self.assertTrue((X[col] == 0.0).all(), col)

    def test_cross_features_lagged_one_day(self):
        df = _prep(make_uptrend_dip())
        idx = df.index
        flat = pd.Series(100.0, index=idx)
        gold = flat.copy()
        gold.iloc[50] = 1000.0  # one-day gold spike
        X = build_features(df, cross={"gold": gold})
        g5 = X["gold_mom5"]
        # The spike must first reach a bar on day 51 (1-day lag), not day 50.
        self.assertEqual(g5.iloc[50], 0.0)
        self.assertNotEqual(g5.iloc[51], 0.0)

    def test_cot_features_default_to_zero_and_lagged(self):
        df = _prep(make_uptrend_dip())
        X = build_features(df, symbol="EURUSD")
        # No COT data -> neutral 50 (a positioning percentile's baseline).
        self.assertTrue((X["cot_percentile"] == 50.0).all())
        idx = df.index
        cot = pd.Series(50.0, index=idx)
        cot.iloc[40] = 99.0
        X2 = build_features(df, symbol="EURUSD", cot={"EUR": cot})
        # 1-day lag: day 40's reading first reaches the bar on day 41.
        self.assertEqual(X2["cot_percentile"].iloc[40], 50.0)
        self.assertEqual(X2["cot_percentile"].iloc[41], 99.0)

    def test_cot_features_unknown_symbol_neutral(self):
        # A symbol with no CCY_MAP entry (or no cot dict) gets the neutral
        # 50 baseline, never zeros-as-missing or a crash.
        df = _prep(make_uptrend_dip())
        idx = df.index
        cot = pd.Series(99.0, index=idx)
        X = build_features(df, symbol="ZZZUSD", cot={"EUR": cot})
        self.assertTrue((X["cot_percentile"] == 50.0).all())
        # cot=None -> neutral 50 too (default branch).
        X2 = build_features(df, symbol="EURUSD", cot=None)
        self.assertTrue((X2["cot_percentile"] == 50.0).all())

    def test_symbol_is_categorical(self):
        df = _prep(make_uptrend_dip())
        X = build_features(df, symbol="EURUSD")
        self.assertEqual(str(X["symbol"].dtype), "category")

    def test_features_causal(self):
        # Features at bar t must not change when future bars are altered.
        df = _prep(make_uptrend_dip())
        X1 = build_features(df)
        df2 = df.copy()
        # Corrupt the future after bar 100 in an unrelated column set.
        df2.loc[df2.index[100] :, ["close", "high", "low", "open"]] *= 2.0
        X2 = build_features(df2)
        self.assertTrue(
            X1.iloc[:50].equals(X2.iloc[:50]),
            "early features must be unaffected by future data",
        )

    def test_macro_features_default_to_zero(self):
        df = _prep(make_uptrend_dip())
        X = build_features(df)
        for col in MACRO_FEATURES:
            self.assertTrue((X[col] == 0.0).all(), f"{col} should default to neutral 0")

    def _zero_volume_frame(self):
        # MT5 FX tick volume is often 0: make_uptrend_dip -> prepare with
        # volume zeroed BEFORE indicator computation (like the FX parquet).
        df = make_uptrend_dip()
        df["volume"] = 0.0
        return _prep(df)

    def test_zero_volume_fx_gets_neutral_relative_volume(self):
        """Regression: MT5 FX tick volume is often 0, which makes
        relative_volume 0/0 = NaN on every bar. An all-NaN column must not
        poison the feature matrix (it silently disabled the ML probability
        for every FX pair while XAUUSD kept working)."""
        X = build_features(self._zero_volume_frame(), symbol="EURUSD")
        self.assertTrue(
            (X["relative_volume"] == 1.0).all(),
            "zero-volume frames must read neutral 1.0",
        )
        self.assertTrue(
            X.tail(20).notna().all().all(), "no all-NaN column after warm-up"
        )

    def test_zero_volume_fx_predict_series_not_none(self):
        """End-to-end: predict_series must produce a probability on a
        zero-volume FX frame instead of returning None."""
        from src.model.model import load_model, predict_series

        bundle = load_model()
        if bundle is None:
            self.skipTest("no ensemble model saved")
        df = self._zero_volume_frame()
        prob = predict_series(df, bundle=bundle, symbol="EURUSD")
        self.assertIsNotNone(prob)
        last = prob.dropna()
        self.assertGreater(len(last), 0)
        self.assertTrue((last >= 0).all() and (last <= 1).all())

    def test_macro_features_aligned_causally(self):
        # Macro state known at day D must only reach bars on day D+1 or later.
        df = _prep(make_uptrend_dip())
        idx = df.index
        macro = pd.DataFrame({"dxy": 100.0 * np.ones(len(idx))}, index=idx)
        # One distinct USD move on a single day.
        move_day = idx[50]
        macro.loc[move_day, "dxy"] = 150.0
        X = build_features(df, symbol="EURUSD", macro=macro)
        mom = X["dxy_mom20"]
        # The spike must not leak into features before the 1-day alignment lag.
        move_day = idx[50]
        self.assertTrue(
            (mom[mom != 0.0].index > move_day).all(),
            "macro features must respect the 1-day alignment lag",
        )

    def test_dataset_builds(self):
        df = _prep(make_uptrend_dip())
        ds = build_dataset(df, horizon=10)
        # 260-bar frame minus ~200-bar SMA200 warm-up minus label horizon.
        self.assertGreater(len(ds["X"]), 40)
        self.assertEqual(len(ds["X"]), len(ds["y"]))
        self.assertIn("confirmed", ds)
        self.assertIn("censored", ds)
        self.assertIn("weight", ds)
        self.assertTrue((ds["weight"] > 0).all())


class TestAsymmetricLabels(unittest.TestCase):
    def test_censored_dropped_when_requested(self):
        df = _prep(make_uptrend_dip())
        l1 = build_labels(df, horizon=10, drop_censored=False)
        l2 = build_labels(df, horizon=10, drop_censored=True)
        dropped = l1["censored"] & l2["label"].isna()
        self.assertGreater(int(dropped.sum()), 0)
        # Non-censored rows keep identical labels.
        same = ~l1["censored"] & l1["label"].notna() & l2["label"].notna()
        self.assertTrue(l1.loc[same, "label"].equals(l2.loc[same, "label"]))

    def test_asymmetric_barriers_bias_wins_up(self):
        # target closer (0.75 ATR) than stop (1.25 ATR) -> more wins on an
        # uptrend than the old symmetric geometry.
        df = _prep(make_uptrend_dip())
        asym = build_labels(df, horizon=10)["label"].dropna()
        sym = build_labels(df, horizon=10, stop_mult=1.0, target_mult=1.0)[
            "label"
        ].dropna()
        self.assertGreater(asym.mean(), sym.mean())

    def test_meta_labels_only_on_confirmed_dips(self):
        df = _prep(make_uptrend_dip())
        sig = dip_signal_series(df)
        labels = build_meta_labels(df, sig)
        non_confirmed = labels[~sig["confirmed"]]
        self.assertTrue(
            non_confirmed.isna().all(), "non-confirmed bars must have no meta label"
        )
        self.assertTrue(set(labels.dropna().unique()).issubset({0.0, 1.0}))


class TestMacroBias(unittest.TestCase):
    def test_vectorized_matches_scalar(self):
        from src.macro.overlay import macro_bias_for_symbol, macro_bias_series

        aligned = pd.DataFrame(
            {
                "dxy_score": [2.0, -1.0, 0.0],
                "vix_score": [0.0, 1.0, -2.0],
                "tnx_score": [-1.0, 0.5, 1.0],
            }
        )
        for sym in ("EURUSD", "USDJPY", "XAUUSD"):
            vb = macro_bias_series(sym, aligned)
            rb = [
                macro_bias_for_symbol(sym, r.to_dict())["bias"]
                for _, r in aligned.iterrows()
            ]
            self.assertTrue(np.allclose(vb.values, rb), sym)

    def test_gold_is_anti_dollar(self):
        from src.macro.overlay import macro_bias_for_symbol

        bias = macro_bias_for_symbol(
            "XAUUSD", {"dxy_score": 2, "vix_score": 0, "tnx_score": 0}
        )["bias"]
        self.assertLess(bias, 0)


class TestSplit(unittest.TestCase):
    def _ds(self):
        df = _prep(make_uptrend_dip())
        return build_dataset(df, horizon=10)

    def test_split_is_chronological(self):
        ds = self._ds()
        sp = split_chronological(ds, "2024-06-01", 10)
        if len(sp["X_train"]) and len(sp["X_test"]):
            t_tr, t_te = sp["time_train"].max(), sp["time_test"].min()
            self.assertLess(t_tr, t_te)
        # The embargo drops the horizon bars before the split from training,
        # so train + test may be less than the full dataset.
        self.assertLessEqual(len(sp["X_train"]) + len(sp["X_test"]), len(ds["X"]))
        self.assertGreaterEqual(len(sp["X_train"]), 0)

    def test_no_columns_leak(self):
        ds = self._ds()
        sp = split_chronological(ds, "2024-06-01", 10)
        if len(sp["X_train"]):
            self.assertTrue(ds["X"].columns.equals(sp["X_train"].columns))


class TestModelRoundtrip(unittest.TestCase):
    def test_train_save_load_predict(self):
        dfs = [_prep(make_uptrend_dip()), _prep(make_downtrend())]
        Xs, ys = [], []
        for df in dfs:
            ds = build_dataset(df, horizon=10)
            Xs.append(ds["X"])
            ys.append(ds["y"])
        X, y = pd.concat(Xs), pd.concat(ys)
        model = train_model(X, y)
        prob = model.predict_proba(X)[:, 1]
        self.assertTrue((prob >= 0).all() and (prob <= 1).all())

        imp = feature_importance(model, X.columns.tolist())
        self.assertEqual(len(imp), len(X.columns))
        self.assertAlmostEqual(imp["gain_pct"].sum(), 100.0, places=1)

        metrics = evaluate(y.values, prob)
        self.assertGreater(metrics["auc"], 0.5)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_model(
                model,
                X.columns.tolist(),
                {"auc_oos": metrics["auc"], "symbols": ["A"]},
                f"{tmp}/m.joblib",
            )
            bundle = load_model(path)
            self.assertIsNotNone(bundle)
            self.assertEqual(bundle["features"], list(X.columns))
            self.assertIsNone(load_model("/nonexistent/x.joblib"))

            prob2 = predict_series(_prep(make_uptrend_dip()), bundle=bundle)
            self.assertIsNotNone(prob2)
            self.assertTrue((prob2.dropna() >= 0).all())
            self.assertTrue((prob2.dropna() <= 1).all())


class TestCategoricalAndCalibration(unittest.TestCase):
    def _pooled(self):
        dfs = [_prep(make_uptrend_dip()), _prep(make_downtrend())]
        X = pd.concat(
            [
                build_features(d, symbol=s)
                for d, s in zip(dfs, ["EURUSD", "XAUUSD"], strict=True)
            ]
        )
        X["symbol"] = X["symbol"].astype("category")
        y = pd.Series(np.tile([0.0, 1.0], len(X) // 2), index=X.index)
        return X, y

    def test_train_with_categorical_symbol(self):
        X, y = self._pooled()
        model = train_model(X, y)
        p = model.predict_proba(X)[:, 1]
        self.assertTrue((p >= 0).all() and (p <= 1).all())

    def test_predict_with_unseen_symbol(self):
        X, y = self._pooled()
        model = train_model(X, y)
        Xu = build_features(_prep(make_uptrend_dip()), symbol="NEWPAIR")
        p = model.predict_proba(Xu)[:, 1]
        self.assertTrue((p >= 0).all() and (p <= 1).all())

    def test_calibrator_roundtrip_in_bundle(self):
        X, y = self._pooled()
        model = train_model(X, y)
        p = model.predict_proba(X)[:, 1]
        calib = fit_calibrator(y.values, p)
        self.assertTrue((apply_calibrator(calib, p) >= 0).all())
        self.assertTrue((apply_calibrator(calib, p) <= 1).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = save_model(
                model,
                list(X.columns),
                {"auc_oos": 0.55},
                f"{tmp}/m.joblib",
                calibrator=calib,
            )
            bundle = load_model(path)
            self.assertIsNotNone(bundle["calibrator"])
            prob = predict_series(
                _prep(make_uptrend_dip()), bundle=bundle, symbol="EURUSD"
            )
            self.assertIsNotNone(prob)
            self.assertTrue((prob.dropna() >= 0).all())
            self.assertTrue((prob.dropna() <= 1).all())


class TestDecilesAndGate(unittest.TestCase):
    def test_decile_report_separates_perfect_model(self):
        y = np.array([0.0] * 50 + [1.0] * 50)
        p = np.linspace(0.01, 0.99, 100)
        rows = decile_report(y, p, n_bins=10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["win_rate"], 0.0)
        self.assertEqual(rows[-1]["win_rate"], 1.0)

    def test_top_decile_lift(self):
        y = np.array([0.0] * 50 + [1.0] * 50)
        p = np.linspace(0.01, 0.99, 100)
        lift = top_decile_lift(y, p)
        self.assertAlmostEqual(lift["base"], 0.5)
        self.assertEqual(lift["top_decile"], 1.0)
        self.assertAlmostEqual(lift["lift"], 2.0)

    def test_coin_flip_has_no_lift(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 400).astype(float)
        p = rng.random(400)
        lift = top_decile_lift(y, p)
        self.assertLess(lift["lift"], 1.5)

    def test_dip_filter_gate(self):
        rng = np.random.default_rng(3)
        # 100 confirmed dips: high-prob half all winners, low half all losers.
        y = np.array([1.0] * 50 + [0.0] * 50)
        p = np.concatenate([rng.uniform(0.6, 0.9, 50), rng.uniform(0.1, 0.4, 50)])
        conf = np.ones(100, dtype=bool)
        g = dip_filter_gate(y, p, conf)
        self.assertIsNotNone(g)
        self.assertEqual(g["n_dips"], 100)
        self.assertGreater(g["top_half_win_rate"], 0.9)
        self.assertLess(g["bottom_half_win_rate"], 0.1)
        self.assertGreater(g["lift"], 1.5)
        # Too few dips -> None.
        self.assertIsNone(dip_filter_gate(y[:10], p[:10], conf[:10]))

    def test_dip_filter_gate_tied_median_no_nan(self):
        # All-equal probabilities (e.g. heavily tied calibrated probs) must
        # not produce NaN win rates - the gate degrades to "no separation".
        y = np.array([1.0] * 30 + [0.0] * 30)
        p = np.full(60, 0.55)
        conf = np.ones(60, dtype=bool)
        g = dip_filter_gate(y, p, conf)
        self.assertIsNotNone(g)
        self.assertEqual(g["top_half_win_rate"], g["bottom_half_win_rate"])
        self.assertEqual(g["lift"], 1.0)
        self.assertFalse(np.isnan(g["top_half_win_rate"]))


class TestEarlyStopping(unittest.TestCase):
    def test_train_with_chronological_val_slice(self):
        rng = np.random.default_rng(11)
        n = 900
        X = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
                "trend": np.linspace(-2, 2, n),
            }
        )
        y = pd.Series((X["a"] + X["trend"] > 0).astype(float))
        X_tr, y_tr, w_tr = (
            X.iloc[:600],
            y.iloc[:600],
            pd.Series(1.0, index=X.index[:600]),
        )
        X_val, y_val, w_val = (
            X.iloc[600:],
            y.iloc[600:],
            pd.Series(1.0, index=X.index[600:]),
        )
        model = train_model(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=w_val,
        )
        p = model.predict_proba(X_val)[:, 1]
        self.assertTrue((p >= 0).all() and (p <= 1).all())
        self.assertGreater(evaluate(y_val.values, p)["auc"], 0.6)

    def test_train_model_kwargs_applied(self):
        rng = np.random.default_rng(13)
        n = 500
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = pd.Series((X["a"] + X["b"] > 0).astype(float))
        cfg = {"num_leaves": 7, "learning_rate": 0.05, "min_child_samples": 5}
        model = train_model(X, y, model_kwargs=cfg)
        # The searched config must actually reach the model.
        self.assertEqual(model.num_leaves, 7)
        self.assertEqual(model.learning_rate, 0.05)

    def test_early_stopping_kwargs_legacy_branch(self):
        from unittest.mock import patch
        from src.model.model import _early_stopping_kwargs

        Xv = pd.DataFrame({"a": [1.0, 2.0]})
        yv = pd.Series([0.0, 1.0])
        with patch("lightgbm.__version__", "4.4.0"):
            kwargs = _early_stopping_kwargs(Xv, yv, None, 40)
        self.assertIn("eval_set", kwargs)
        self.assertIn("early_stopping_rounds", kwargs)
        self.assertEqual(kwargs["early_stopping_rounds"], 40)
        # Modern branch uses eval_X/eval_y and callbacks.
        with patch("lightgbm.__version__", "4.7.0"):
            kwargs = _early_stopping_kwargs(Xv, yv, None, 40)
        self.assertIn("eval_X", kwargs)
        self.assertNotIn("eval_set", kwargs)
        self.assertIn("callbacks", kwargs)

    def test_chrono_val_split_guards(self):
        X = pd.DataFrame({"a": np.arange(100.0)})
        y = pd.Series(np.zeros(100))
        w = pd.Series(np.ones(100))
        out = _chrono_val_split(X, y, w, min_val=200)
        self.assertIsNone(out[3])  # too small to reserve a slice
        X2 = pd.DataFrame({"a": np.arange(1500.0)})
        y2 = pd.Series(np.zeros(1500))
        w2 = pd.Series(np.ones(1500))
        out2 = _chrono_val_split(X2, y2, w2)
        self.assertEqual(len(out2[0]) + len(out2[3]), 1500)


class TestWalkForward(unittest.TestCase):
    def _ds(self):
        dfs = [_prep(make_uptrend_dip()), _prep(make_downtrend())]
        X = pd.concat(
            [
                build_features(d, symbol=s)
                for d, s in zip(dfs, ["EURUSD", "XAUUSD"], strict=True)
            ]
        )
        X["symbol"] = X["symbol"].astype("category")
        y = pd.Series(np.tile([0.0, 1.0], len(X) // 2), index=X.index)
        times = pd.concat([d.index.to_series() for d in dfs])
        return {
            "X": X,
            "y": y,
            "weight": pd.Series(1.0, index=X.index),
            "time": times,
            "confirmed": pd.Series(False, index=X.index),
        }

    def test_walk_forward_pooled_metrics(self):
        ds = self._ds()
        out = walk_forward(ds, horizon=10, n_folds=2, min_train=50, min_test=20)
        self.assertIn("metrics", out)
        self.assertIn("oos_y", out)
        self.assertIn("oos_p", out)
        self.assertEqual(len(out["oos_y"]), len(out["oos_p"]))
        self.assertGreater(len(out["fold_metrics"]), 0)

    def test_walk_forward_returns_oos_idx(self):
        ds = self._ds()
        out = walk_forward(ds, horizon=10, n_folds=2, min_train=50, min_test=20)
        idx = out["oos_idx"]
        self.assertEqual(len(idx), len(out["oos_y"]))
        # The OOS positions must map back to real dataset rows.
        self.assertLess(int(idx.max()), len(ds["X"]))


class TestCrossProxyCache(unittest.TestCase):
    def _valid_ohlc(self, base: float, n: int = 30) -> pd.DataFrame:
        """Synthetic OHLC that passes clean_data's integrity checks."""
        o = np.linspace(base, base + 10, n)
        return pd.DataFrame(
            {
                "open": o,
                "high": o + 1.0,
                "low": o - 1.0,
                "close": o + 0.5,
                "volume": 1000,
            },
            index=pd.date_range("2024-01-01", periods=n),
        )

    def test_cross_proxies_busts_on_file_change(self):
        """The mtime-aware key means a data refresh is picked up without a
        restart (important for the long-lived API server)."""
        import os
        import shutil

        tmp = tempfile.mkdtemp()
        try:
            self._valid_ohlc(100.0).to_parquet(f"{tmp}/AUDJPY_D1.parquet")
            p1 = _cross_proxies(None, tmp)
            self.assertIn("risk", p1)
            v1 = float(p1["risk"]["AUDJPY"].iloc[0])
            # Overwrite the file (new mtime) with different prices.
            self._valid_ohlc(200.0).to_parquet(f"{tmp}/AUDJPY_D1.parquet")
            os.utime(f"{tmp}/AUDJPY_D1.parquet", None)
            p2 = _cross_proxies(None, tmp)
            v2 = float(p2["risk"]["AUDJPY"].iloc[0])
            self.assertNotEqual(v1, v2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestServeContext(unittest.TestCase):
    """predict_series must load the same MTF / cross-asset / COT context at
    serve time that training used (and skip it when the model doesn't need
    it) - otherwise live probabilities silently use neutral features."""

    def _bundle(self, tmp: str, features=None):
        df = _prep(make_uptrend_dip())
        X_full = build_features(df, symbol="EURUSD")
        features = features or list(X_full.columns)
        X = X_full[features]
        y = pd.Series([0.0, 1.0] * (len(X) // 2) + [0.0] * (len(X) % 2), index=X.index)
        model = train_model(X, y)
        path = save_model(model, features, {"auc_oos": 0.5}, f"{tmp}/m.joblib")
        return path, df

    def test_predict_series_loads_serve_context(self):
        from unittest.mock import patch
        import src.model.cot as cotmod
        import src.model.model as modelmod

        with tempfile.TemporaryDirectory() as tmp:
            path, df = self._bundle(tmp)
            with (
                patch.object(modelmod, "_mtf_frame", return_value=None) as m_mtf,
                patch.object(modelmod, "_cross_proxies", return_value={}) as m_cross,
                patch.object(cotmod, "load_cot", return_value=None) as m_cot,
                patch(
                    "src.macro.overlay.macro_for_model", return_value=None
                ) as m_macro,
            ):
                prob = predict_series(
                    df, model_path=path, symbol="EURUSD", group="full_fx", data_dir=tmp
                )
            self.assertIsNotNone(prob)
            p = prob.dropna()
            self.assertTrue((p >= 0).all() and (p <= 1).all())
            m_mtf.assert_called_once_with("EURUSD", "full_fx", tmp)
            m_cross.assert_called_once_with("full_fx", tmp)
            m_cot.assert_called_once_with(f"{tmp}/cot")
            m_macro.assert_called_once_with(tmp)

    def test_predict_series_skips_unneeded_context(self):
        from unittest.mock import patch
        import src.model.cot as cotmod
        import src.model.model as modelmod

        basic = ["rsi_14", "adx", "atr_pct", "ret_5"]
        with tempfile.TemporaryDirectory() as tmp:
            path, df = self._bundle(tmp, features=basic)
            with (
                patch.object(modelmod, "_mtf_frame") as m_mtf,
                patch.object(modelmod, "_cross_proxies") as m_cross,
                patch.object(cotmod, "load_cot") as m_cot,
                patch("src.macro.overlay.macro_for_model") as m_macro,
            ):
                prob = predict_series(
                    df, model_path=path, symbol="EURUSD", group="full_fx", data_dir=tmp
                )
            self.assertIsNotNone(prob)
            m_mtf.assert_not_called()
            m_cross.assert_not_called()
            m_cot.assert_not_called()
            m_macro.assert_not_called()


class TestGroupDispatch(unittest.TestCase):
    def test_symbol_groups_split(self):
        syms = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "XAUUSD",
            "EURJPY",
            "GBPJPY",
            "AUDJPY",
            "USDCAD",
            "AUDUSD",
        ]
        groups = _symbol_groups(syms)
        self.assertEqual(groups["gold"], ["XAUUSD"])
        self.assertEqual(
            sorted(groups["jpy"]), ["AUDJPY", "EURJPY", "GBPJPY", "USDJPY"]
        )
        self.assertIn("EURUSD", groups["majors"])
        self.assertNotIn("XAUUSD", groups["majors"])
        # Empty groups are dropped.
        self.assertNotIn("gold", _symbol_groups(["EURUSD"]))

    def test_group_model_path(self):
        self.assertIn("jpy", _group_model_path("jpy"))
        self.assertIn("dip_lgbm", _group_model_path("gold"))


class TestSearchAndStack(unittest.TestCase):
    def test_search_hyperparams_returns_config(self):
        rng = np.random.default_rng(5)
        n = 3500
        X = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
                "trend": np.linspace(-2, 2, n),
            }
        )
        X["symbol"] = pd.Series("EURUSD", index=X.index, dtype="category")
        y = pd.Series((X["a"] + X["trend"] > 0).astype(float))
        ds = {"X": X, "y": y, "weight": pd.Series(1.0, index=X.index)}
        cfg = search_hyperparams(ds, horizon=5, max_rows=3500)
        self.assertIsInstance(cfg, dict)
        self.assertIn("num_leaves", cfg)

    def test_stack_oos_guards_and_runs(self):
        rng = np.random.default_rng(9)
        n = 700
        X = pd.DataFrame(
            {
                "dip_score": rng.normal(size=n),
                "macro_bias": rng.normal(size=n),
                "adx": rng.uniform(10, 50, n),
            }
        )
        y = pd.Series((X["dip_score"] + X["macro_bias"] > 0).astype(float))
        ds = {"X": X, "y": y}
        oos_idx = np.arange(n)
        p = 1.0 / (1.0 + np.exp(-(X["dip_score"] + X["macro_bias"]).values))
        self.assertIsNone(stack_oos(ds, np.arange(100), p[:100]))
        res = stack_oos(ds, oos_idx, p)
        self.assertIsNotNone(res)
        self.assertIn("stacked_auc", res)
        self.assertIn("delta", res)


class TestIntegration(unittest.TestCase):
    def test_scanner_ml_graceful(self):
        # The scanner must never crash on the ML step: ml_prob is a valid
        # percentage when a model is saved, None otherwise.
        from src.model.model import load_model
        from src.analysis.scanner import scan_symbol

        has_model = load_model() is not None
        row = scan_symbol("US500", group="candidates", fetch_mt5=False)
        self.assertIn("ml_prob", row)
        if has_model:
            self.assertIsInstance(row["ml_prob"], float)
            self.assertTrue(0.0 <= row["ml_prob"] <= 100.0)
        else:
            self.assertIsNone(row["ml_prob"])


if __name__ == "__main__":
    unittest.main()
