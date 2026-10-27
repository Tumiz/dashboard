from argparse import ArgumentParser
from datetime import date, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "dashboard_monthly.csv"
PREDICTIONS_PATH = Path(__file__).resolve().parent / "predictions.json"
d = pd.read_csv(DATA_PATH, index_col="month")

MINUS = "\N{MINUS SIGN}"
PREDICTION_INTERVAL_COVERAGE = 0.90
MIN_CALIBRATION_SAMPLES = 20
MAX_CALIBRATION_SAMPLES = 120
HISTORICAL_BASELINE_LOOKBACK = 20


def normalize_month(month):
    try:
        year, mm = str(month).split("-")
        period = pd.Period(year=int(year), month=int(mm), freq="M")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid month {month!r}, expected YYYY-MM") from exc
    return f"{period.year:04d}-{period.month:02d}"


def data_before(month):
    return d[d.index < normalize_month(month)]


def add_months(month, k):
    period = pd.Period(month, freq="M") + k
    return f"{period.year:04d}-{period.month:02d}"


def ln_return(series):
    return np.log(series / series.shift(1))


def _quarterly_windows(f):
    """Rolling three-month windows with the gold lows and highs for each window."""
    gpr = f["gpr"]
    obs = gpr.notna().to_numpy()
    last_obs = np.where(obs, np.arange(len(obs)), np.nan)
    if obs.any():
        last_obs = pd.Series(last_obs, index=gpr.index).ffill().fillna(-4).to_numpy()
    age = np.arange(len(obs)) - last_obs
    gpr_window = gpr.ffill().where(pd.Series(age < 3, index=gpr.index))
    return pd.DataFrame({
        "gold_low_return": np.log(f["gold_low"].rolling(3).min() / f["gold"].shift(3)),
        "gold_high_return": np.log(f["gold_high"].rolling(3).max() / f["gold"].shift(3)),
        "dxy_return": f["dxy_return"].rolling(3).sum(),
        "real_yield_change": f["ryc"].rolling(3).sum(),
        "gpr": gpr_window,
    }, index=f.index)


def actual_price(month):
    if month in d.index and pd.notna(d.loc[month, "xau_usd"]):
        return float(d.loc[month, "xau_usd"])
    return None


def _r2(y, y_hat):
    ss_total = np.sum((y - y.mean()) ** 2)
    if len(y) < 2 or not np.isfinite(ss_total) or ss_total == 0:
        return float("nan")
    return float(1 - np.sum((y - y_hat) ** 2) / ss_total)


def _fit_linear_model(reg, target, predictors):
    if len(reg):
        X = reg[predictors].to_numpy()
        y = reg[target].to_numpy()
        X_design = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        r2 = _r2(y, X_design @ coef)
    else:
        coef = np.full(len(predictors) + 1, np.nan)
        r2 = float("nan")

    return {
        "alpha": float(coef[0]),
        **{
            f"beta_{predictor}": float(value)
            for predictor, value in zip(predictors, coef[1:])
        },
        "r2": float(r2),
        "n": int(len(reg)),
    }


def gold_dxy_linear_regression(data, verbose=True):
    df = data[
        ["xau_usd", "xau_usd_low", "xau_usd_high", "twexb", "real_yield_pct"]
    ].dropna()
    real_yield_change = df["real_yield_pct"].diff()
    reg = pd.DataFrame({
        "gold_low_return": np.log(df["xau_usd_low"] / df["xau_usd"].shift(1)),
        "gold_high_return": np.log(df["xau_usd_high"] / df["xau_usd"].shift(1)),
        "dxy_return": ln_return(df["twexb"]),
        "real_yield_change": real_yield_change,
    }).dropna()

    monthly_predictors = ["dxy_return", "real_yield_change"]
    monthly = {
        bound: _fit_linear_model(reg, f"gold_{bound}_return", monthly_predictors)
        for bound in ("low", "high")
    }

    if verbose:
        for bound, params in monthly.items():
            print(
                f"Monthly {bound} model: Gold_{bound.title()}_Return_t = "
                f"{_alpha_txt(params['alpha'])} "
                f"{_term_txt(params['beta_dxy_return'])} * DXY_Return_t "
                f"{_term_txt(params['beta_real_yield_change'])} * RealYield_Change_t"
            )
            print(f"Monthly {bound} R-squared = {params['r2']:.4f}, n = {params['n']}")

    quarterly = _quarterly_windows(_features(data)).dropna()
    quarterly_predictors = ["dxy_return", "real_yield_change", "gpr"]
    quarterly_models = {
        bound: _fit_linear_model(
            quarterly, f"gold_{bound}_return", quarterly_predictors
        )
        for bound in ("low", "high")
    }

    if verbose:
        for bound, params in quarterly_models.items():
            print(
                f"Quarterly {bound} model: Gold_{bound.title()}_Return_q = "
                f"{_alpha_txt(params['alpha'])} "
                f"{_term_txt(params['beta_dxy_return'])} * DXY_Return_q "
                f"{_term_txt(params['beta_real_yield_change'])} * RealYield_Change_q "
                f"{_term_txt(params['beta_gpr'], 6)} * GPR_q"
            )
            print(
                f"Quarterly {bound} R-squared = {params['r2']:.4f}, "
                f"n = {params['n']}"
            )

    return {
        "monthly": monthly,
        "quarterly": quarterly_models,
    }


def _features(data):
    gold = data["xau_usd"].astype(float)
    twexb = data["twexb"].astype(float)
    ry = data["real_yield_pct"].astype(float)
    gpr = data["gpr"].astype(float)
    labels = pd.Series(data.index, index=data.index, dtype=object)

    prev_gold = gold.ffill().shift(1)
    prev_twexb = twexb.ffill().shift(1)
    prev_ry = ry.ffill().shift(1)
    prev_twexb_month = labels.where(twexb.notna()).ffill().shift(1)
    prev_ry_month = labels.where(ry.notna()).ffill().shift(1)
    dxy_return = np.log(twexb / prev_twexb).where(twexb.gt(0) & prev_twexb.gt(0))

    return pd.DataFrame({
        "gold": gold,
        "gold_low": data["xau_usd_low"].astype(float),
        "gold_high": data["xau_usd_high"].astype(float),
        "twexb": twexb,
        "ry": ry,
        "gpr": gpr,
        "prev_gold": prev_gold,
        "prev_twexb": prev_twexb,
        "prev_ry": prev_ry,
        "prev_twexb_month": prev_twexb_month,
        "prev_ry_month": prev_ry_month,
        "dxy_return": dxy_return,
        "ryc": ry - prev_ry,
    }, index=data.index)


def _alpha_txt(value, digits=4):
    text = f"{abs(value):.{digits}f}"
    return MINUS + text if value < 0 else text


def _term_txt(value, digits=4):
    return f"{MINUS if value < 0 else '+'} {abs(value):.{digits}f}"


def _tex_number(value, digits=8):
    rounded = float(f"{value:.{digits}f}")
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def _latex_alpha(value, digits=4):
    return f"{value:.{digits}f}"


def _latex_op(value, digits=4):
    return f"{'-' if value < 0 else '+'}{abs(value):.{digits}f}"


def _model_equation(params, kind):
    """LaTeX display equation for the fitted model (rendered by the dashboard)."""
    if kind == "monthly":
        lhs = lambda bound: (
            "\\ln\\left(\\frac{X_{" + bound + ",t}}{X_{t-1}}\\right)"
        )
        dxy = "\\ln\\left(\\frac{\\mathrm{TWEXB}_t}{\\mathrm{TWEXB}_{t-1}}\\right)"
        ry = "\\,\\Delta y_t"
        digits = 4
    else:
        lhs = lambda bound: (
            "\\ln\\left(\\frac{X_{" + bound + ",q}}{X_{q-1}}\\right)"
        )
        dxy = "\\ln\\left(\\frac{\\mathrm{TWEXB}_q}{\\mathrm{TWEXB}_{q-1}}\\right)"
        ry = "\\,\\Delta y_q"
        digits = 4

    equations = []
    for bound in ("low", "high"):
        coefficients = params[bound]
        equation = (
            lhs(bound)
            + "=" + _latex_alpha(coefficients["alpha"])
            + _latex_op(coefficients["beta_dxy_return"], digits)
            + dxy
            + _latex_op(coefficients["beta_real_yield_change"], digits)
            + ry
        )
        if kind == "quarterly":
            equation += (
                _latex_op(coefficients["beta_gpr"], 6)
                + "\\,\\mathrm{GPR}_q"
            )
        equations.append("\\[" + equation + "\\]")
    return " ".join(equations)


def _forecast_equation(xau, params, predictors, kind):
    endpoint = "t+1" if kind == "monthly" else "q+1"
    equations = []
    for bound in ("low", "high"):
        coefficients = params[predictors[bound + "_model"]]
        terms = [
            _latex_alpha(coefficients["alpha"]),
            _latex_op(coefficients["beta_dxy_return"])
            + "\\left(" + _tex_number(predictors["dxy"]) + "\\right)",
            _latex_op(coefficients["beta_real_yield_change"])
            + "\\left(" + _tex_number(predictors["ryc"]) + "\\right)",
        ]
        if kind == "quarterly":
            terms.append(
                _latex_op(coefficients["beta_gpr"], 6)
                + "\\left(" + _tex_number(predictors["gpr"]) + "\\right)"
            )
        estimate = predictors[bound + "_price"]
        equations.append(
            "\\[\\widehat{X}_{" + bound + "," + endpoint + "}="
            + _tex_number(xau, 2)
            + "\\exp\\left(" + "".join(terms) + "\\right)\\approx "
            + _tex_number(estimate, 2) + "\\]"
        )
    return " ".join(equations)


def _order_bounds(predictors):
    endpoints = sorted(
        (
            (bound, predictors[bound + "_return"], predictors[bound + "_price"])
            for bound in ("low", "high")
        ),
        key=lambda endpoint: endpoint[2],
    )
    for bound, (model, predicted_return, predicted_price) in zip(
        ("low", "high"), endpoints
    ):
        predictors[bound + "_model"] = model
        predictors[bound + "_return"] = predicted_return
        predictors[bound + "_price"] = predicted_price
    return predictors


def _forecast_base_price(f, month, horizon):
    start = month if horizon == "monthly" else add_months(month, -2)
    prior = f["gold"][f.index < start].dropna()
    return float(prior.iloc[-1]) if len(prior) else None


def _actual_log_range(f, month, horizon):
    actual = _actual_range(f, month, horizon)
    anchor = _forecast_base_price(f, month, horizon)
    if actual is None or anchor is None or anchor <= 0:
        return None
    return float(np.log(actual[0] / anchor)), float(np.log(actual[1] / anchor))


def _range_calibration_score(forecast, actual_low_return, actual_high_return):
    return max(
        0.0,
        forecast["predicted_low_return"] - actual_low_return,
        actual_high_return - forecast["predicted_high_return"],
    )


def _actual_range(f, month, horizon):
    if horizon == "monthly":
        if month not in f.index:
            return None
        low = f.at[month, "gold_low"]
        high = f.at[month, "gold_high"]
    else:
        window = [add_months(month, offset) for offset in (-2, -1, 0)]
        if any(window_month not in f.index for window_month in window):
            return None
        low = f.loc[window, "gold_low"].min()
        high = f.loc[window, "gold_high"].max()
    if pd.isna(low) or pd.isna(high):
        return None
    return float(low), float(high)


def _calibration_scores(f, series, horizon, cutoff):
    scores = []
    for month, forecast in series.items():
        if month >= cutoff:
            continue
        actual = _actual_log_range(f, month, horizon)
        if actual is not None:
            scores.append(_range_calibration_score(forecast, *actual))
    return scores


def _conformal_adjustment(scores):
    if len(scores) < MIN_CALIBRATION_SAMPLES:
        return None
    ordered = np.sort(np.asarray(scores, dtype=float))
    rank = min(
        int(np.ceil((len(ordered) + 1) * PREDICTION_INTERVAL_COVERAGE)),
        len(ordered),
    )
    return float(ordered[rank - 1])


def _prediction_interval(f, forecast, series, horizon, cutoff, month):
    if (
        forecast["predicted_low_price"] is None
        or forecast["predicted_high_price"] is None
    ):
        return None
    scores = _calibration_scores(f, series, horizon, cutoff)
    adjustment = _conformal_adjustment(scores[-MAX_CALIBRATION_SAMPLES:])
    if adjustment is None:
        return None
    base_price = _forecast_base_price(f, month, horizon)
    if base_price is None:
        return None
    return {
        "coverage": PREDICTION_INTERVAL_COVERAGE,
        "lower": float(base_price * np.exp(forecast["predicted_low_return"] - adjustment)),
        "upper": float(base_price * np.exp(forecast["predicted_high_return"] + adjustment)),
        "log_return_adjustment": adjustment,
        "calibration_samples": min(len(scores), MAX_CALIBRATION_SAMPLES),
    }


def _calibrated_series(f, series, horizon, backtest_start_month=None):
    calibrated = {}
    coverage_hits = []
    boundary_errors = []
    interval_widths = []
    current_month = date.today().strftime("%Y-%m")
    for month, forecast in series.items():
        cutoff = month if horizon == "monthly" else add_months(month, -2)
        interval = _prediction_interval(
            f, forecast, series, horizon, cutoff, month
        )
        calibrated[month] = {**forecast, "prediction_interval": interval}
        actual = _actual_log_range(f, month, horizon)
        matched_period = (
            backtest_start_month is None or month >= backtest_start_month
        )
        if month < current_month and matched_period and actual is not None:
            boundary_errors.append((
                abs(forecast["predicted_low_return"] - actual[0]),
                abs(forecast["predicted_high_return"] - actual[1]),
            ))
        if (
            month < current_month
            and matched_period
            and interval is not None
            and actual is not None
        ):
            coverage_hits.append(
                forecast["predicted_low_return"]
                - interval["log_return_adjustment"] <= actual[0]
                and actual[1]
                <= forecast["predicted_high_return"]
                + interval["log_return_adjustment"]
            )
            interval_widths.append(
                forecast["predicted_high_return"]
                - forecast["predicted_low_return"]
                + 2 * interval["log_return_adjustment"]
            )
    summary = {
        "coverage": PREDICTION_INTERVAL_COVERAGE,
        "minimum_calibration_samples": MIN_CALIBRATION_SAMPLES,
        "maximum_calibration_samples": MAX_CALIBRATION_SAMPLES,
        "backtest_months": len(coverage_hits),
        "backtest_coverage": (
            float(np.mean(coverage_hits)) if coverage_hits else None
        ),
        "boundary_mae_low_log_return": (
            float(np.mean([error[0] for error in boundary_errors]))
            if boundary_errors else None
        ),
        "boundary_mae_high_log_return": (
            float(np.mean([error[1] for error in boundary_errors]))
            if boundary_errors else None
        ),
        "mean_interval_width_log_return": (
            float(np.mean(interval_widths)) if interval_widths else None
        ),
    }
    return calibrated, summary


def _historical_range_baseline(f, horizon):
    """Evaluate trailing-mean range forecasts against the same past-only calibration."""
    outcomes = []
    current_month = date.today().strftime("%Y-%m")
    for month in f.index:
        actual = _actual_log_range(f, month, horizon)
        if actual is not None:
            outcomes.append((month, actual))

    baseline = {}
    coverage_hits = []
    boundary_errors = []
    interval_widths = []
    backtest_months = []
    for month, actual in outcomes:
        cutoff = month if horizon == "monthly" else add_months(month, -2)
        history = [value for past_month, value in outcomes if past_month < cutoff]
        if len(history) < HISTORICAL_BASELINE_LOOKBACK:
            continue

        recent = history[-HISTORICAL_BASELINE_LOOKBACK:]
        low_return = float(np.mean([value[0] for value in recent]))
        high_return = float(np.mean([value[1] for value in recent]))
        low_return, high_return = min(low_return, high_return), max(low_return, high_return)
        forecast = {
            "month": month,
            "predicted_low_return": low_return,
            "predicted_high_return": high_return,
        }
        score = _range_calibration_score(forecast, *actual)
        baseline[month] = (forecast, score)

        if month >= current_month:
            continue
        past_scores = [
            past_score
            for past_month, (_, past_score) in baseline.items()
            if past_month < cutoff
        ][-MAX_CALIBRATION_SAMPLES:]
        adjustment = _conformal_adjustment(past_scores)
        if adjustment is None:
            continue
        boundary_errors.append((
            abs(low_return - actual[0]),
            abs(high_return - actual[1]),
        ))
        coverage_hits.append(
            low_return - adjustment <= actual[0]
            and actual[1] <= high_return + adjustment
        )
        interval_widths.append(high_return - low_return + 2 * adjustment)
        backtest_months.append(month)

    return {
        "lookback_periods": HISTORICAL_BASELINE_LOOKBACK,
        "calibration_window_periods": MAX_CALIBRATION_SAMPLES,
        "backtest_months": len(coverage_hits),
        "backtest_start_month": backtest_months[0] if backtest_months else None,
        "backtest_coverage": (
            float(np.mean(coverage_hits)) if coverage_hits else None
        ),
        "boundary_mae_low_log_return": (
            float(np.mean([error[0] for error in boundary_errors]))
            if boundary_errors else None
        ),
        "boundary_mae_high_log_return": (
            float(np.mean([error[1] for error in boundary_errors]))
            if boundary_errors else None
        ),
        "mean_interval_width_log_return": (
            float(np.mean(interval_widths)) if interval_widths else None
        ),
    }


def _actual_returns(f):
    """Trailing actual XAU/USD log returns: 1-month (monthly) and 3-month (quarterly)."""
    gold = f["gold"]
    monthly = np.log(gold / gold.shift(1)).dropna()
    quarterly = np.log(gold / gold.shift(3)).dropna()
    return {
        "monthly": {m: float(v) for m, v in monthly.items()},
        "quarterly": {m: float(v) for m, v in quarterly.items()},
    }


def _monthly_series(f):
    """Walk-forward monthly predictions: refit on data before each month, like the headline forecast."""
    series = {}
    for month in f.index:
        before = data_before(month)
        if len(before) < 2 or before["xau_usd"].dropna().empty:
            continue
        params = gold_dxy_linear_regression(before, verbose=False)["monthly"]
        forecast = _monthly_forecast(f, params, month, "Target month")
        low_price = forecast["predicted_low_price"]
        high_price = forecast["predicted_high_price"]
        if (
            low_price is not None
            and high_price is not None
            and np.isfinite(low_price)
            and np.isfinite(high_price)
        ):
            series[month] = {
                "predicted_low_price": float(low_price),
                "predicted_high_price": float(high_price),
                "predicted_low_return": float(forecast["predicted_low_return"]),
                "predicted_high_return": float(forecast["predicted_high_return"]),
            }
    return series


def _monthly_forecast(f, params, target, label):
    """Forecast `target` month; predictors come from the latest usable inputs."""
    gold = f["gold"]
    has_macro = f["twexb"].notna() & f["ry"].notna()

    before = gold[gold.index < target].dropna()
    base_month = before.index[-1] if len(before) else None
    base_price = float(before.iloc[-1]) if base_month is not None else None

    input_candidates = f.index[(f.index < target) & has_macro]
    input_month = input_candidates[-1] if len(input_candidates) else None
    prior_candidates = (
        f.index[(f.index < input_month) & has_macro] if input_month is not None else pd.Index([])
    )
    prior_month = prior_candidates[-1] if len(prior_candidates) else None

    breakdown = [
        f"Forecast target month: {target}; "
        f"predictor inputs from {input_month or 'an unavailable month'}."
    ]
    if base_price is not None and np.isfinite(base_price):
        breakdown.append(
            f"X_{{t-1}} = last XAU/USD value before the forecast month, "
            f"from {base_month} = {base_price:.2f} USD/oz."
        )
        target_gold = gold[target] if target in gold.index else np.nan
        if pd.notna(target_gold):
            breakdown.append(
                f"The current {target} XAU/USD reading ({float(target_gold):.2f} USD/oz) "
                f"is not used as the forecast starting price; it may be month-to-date, "
                f"not the final monthly close."
            )
    else:
        breakdown.append(f"X_{{t-1}} is unavailable: no XAU/USD value before {target}.")

    dxy = ryc = None
    predicted_low_return = predicted_high_return = None
    predicted_low_price = predicted_high_price = None
    equation = None
    if input_month is not None and prior_month is not None:
        current_twexb = float(f.at[input_month, "twexb"])
        previous_twexb = float(f.at[prior_month, "twexb"])
        current_ry = float(f.at[input_month, "ry"])
        previous_ry = float(f.at[prior_month, "ry"])

        if current_twexb > 0 and previous_twexb > 0:
            dxy = float(np.log(current_twexb / previous_twexb))
            breakdown.append(
                f"DXYReturn_t = ln(TWEXB_{input_month} / TWEXB_{prior_month}) = "
                f"ln({current_twexb:.4f} / {previous_twexb:.4f}) = {dxy:.8f}."
            )
        else:
            breakdown.append("DXYReturn_t is unavailable: TWEXB must be positive in both months.")

        ryc = current_ry - previous_ry
        breakdown.append(
            f"\u0394y_t = real yield in {input_month} \u2212 real yield in {prior_month} = "
            f"{current_ry:.4f}% \u2212 {previous_ry:.4f}% = {ryc:.8f} percentage points."
        )

        if (
            base_price is not None
            and np.isfinite(base_price)
            and dxy is not None
            and np.isfinite(ryc)
        ):
            predictors = {"dxy": dxy, "ryc": ryc}
            for bound in ("low", "high"):
                coefficients = params[bound]
                predicted_return = (
                    coefficients["alpha"]
                    + coefficients["beta_dxy_return"] * dxy
                    + coefficients["beta_real_yield_change"] * ryc
                )
                predictors[bound + "_return"] = float(predicted_return)
                predictors[bound + "_price"] = float(base_price * np.exp(predicted_return))
            predictors = _order_bounds(predictors)
            predicted_low_return = predictors["low_return"]
            predicted_high_return = predictors["high_return"]
            predicted_low_price = predictors["low_price"]
            predicted_high_price = predictors["high_price"]
            breakdown.append(
                f"Predicted low/high log returns = {predicted_low_return:.8f} / "
                f"{predicted_high_return:.8f}."
            )
            equation = _forecast_equation(base_price, params, predictors, "monthly")
    else:
        breakdown.append(
            f"The latest prior TWEXB and real-yield observations before {target} are incomplete."
        )

    return {
        "month": target,
        "label": label,
        "base_month": base_month,
        "base_price": base_price,
        "input_month": input_month,
        "prior_macro_month": prior_month,
        "dxy_return": dxy,
        "real_yield_change": ryc,
        "predicted_low_return": predicted_low_return,
        "predicted_high_return": predicted_high_return,
        "predicted_low_price": predicted_low_price,
        "predicted_high_price": predicted_high_price,
        "actual_price": actual_price(target),
        "breakdown": breakdown,
        "equation": equation,
    }


def _window_context(f, window):
    """Month-by-month terms and predictor metadata for a three-month window."""
    dxy_terms = []
    ryc_terms = []
    gpr = None
    gpr_month = None
    for month in window:
        row = f.loc[month]
        if pd.notna(row["dxy_return"]):
            dxy_terms.append({
                "month": month,
                "previous_month": row["prev_twexb_month"],
                "previous": float(row["prev_twexb"]),
                "current": float(row["twexb"]),
                "value": float(row["dxy_return"]),
            })
        if pd.notna(row["ryc"]):
            ryc_terms.append({
                "month": month,
                "previous_month": row["prev_ry_month"],
                "previous": float(row["prev_ry"]),
                "current": float(row["ry"]),
                "value": float(row["ryc"]),
            })
        if pd.notna(row["gpr"]):
            gpr = float(row["gpr"])
            gpr_month = month
    return {
        "dxy": sum(t["value"] for t in dxy_terms),
        "ryc": sum(t["value"] for t in ryc_terms),
        "gpr": gpr,
        "gpr_month": gpr_month,
        "dxy_terms": dxy_terms,
        "ryc_terms": ryc_terms,
    }


def _quarterly_forecast(f, params, target_month, cutoff_month, label="Quarterly"):
    """Forecast `target_month` using only data from months before `cutoff_month`.

    Features come from the rolling three-month window ending at the cutoff
    (falling back to the most recent complete window before it); the anchor
    is the last XAU/USD close before the cutoff.
    """
    windows = _quarterly_windows(f).dropna()
    windows = windows[windows.index < cutoff_month]
    predictor = windows.index[-1] if len(windows) else None
    no_window = (
        "No complete three-month window with the required DXY return, "
        "real-yield changes, and GPR value ends before the cutoff."
    )

    def result(**overrides):
        base = {
            "month": None,
            "label": label,
            "predictor_window": None,
            "window_end_month": None,
            "base_month": None,
            "base_price": None,
            "dxy_return": None,
            "real_yield_change": None,
            "gpr": None,
            "gpr_month": None,
            "predicted_low_return": None,
            "predicted_high_return": None,
            "predicted_low_price": None,
            "predicted_high_price": None,
            "actual_price": None,
            "breakdown": [],
            "equation": None,
        }
        base.update(overrides)
        return base

    if predictor is None:
        return result(breakdown=[no_window])

    window = [add_months(predictor, -2), add_months(predictor, -1), predictor]
    ctx = _window_context(f, window)
    dxy = ctx["dxy"]
    ryc = ctx["ryc"]
    gpr = ctx["gpr"]

    gold_before = f["gold"][f.index < cutoff_month].dropna()
    if not len(gold_before):
        return result(
            predictor_window=window,
            window_end_month=predictor,
            breakdown=[
                f"X_{{t-1}} is unavailable: no XAU/USD value before {cutoff_month}."
            ],
        )
    base_month = gold_before.index[-1]
    base_price = float(gold_before.iloc[-1])

    predictors = {"dxy": dxy, "ryc": ryc, "gpr": gpr}
    for bound in ("low", "high"):
        coefficients = params[bound]
        predicted_return = (
            coefficients["alpha"]
            + coefficients["beta_dxy_return"] * dxy
            + coefficients["beta_real_yield_change"] * ryc
            + coefficients["beta_gpr"] * gpr
        )
        predictors[bound + "_return"] = float(predicted_return)
        predictors[bound + "_price"] = float(base_price * np.exp(predicted_return))
    predictors = _order_bounds(predictors)

    window_label = f"{window[0]} to {window[2]}"
    dxy_terms = " + ".join(
        f"{t['month']}: ln({t['current']:.4f} / {t['previous']:.4f}) "
        f"[{t['previous_month']} \u2192 {t['month']}] = {t['value']:.8f}"
        for t in ctx["dxy_terms"]
    )
    ryc_terms = " + ".join(
        f"{t['month']}: {t['current']:.4f}% \u2212 {t['previous']:.4f}% "
        f"[{t['previous_month']} \u2192 {t['month']}] = {t['value']:.8f}"
        for t in ctx["ryc_terms"]
    )
    breakdown = [
        f"Forecast target month: {target_month}; quarterly features from the "
        f"three-month window {window_label}.",
        f"X_{{t-1}} = last XAU/USD value before the forecast month, "
        f"from {base_month} = {base_price:.2f} USD/oz.",
    ]
    target_gold = f.at[target_month, "gold"] if target_month in f.index else np.nan
    if pd.notna(target_gold):
        breakdown.append(
            f"The current {target_month} XAU/USD reading ({float(target_gold):.2f} USD/oz) "
            f"is not used as the forecast starting price; it may be month-to-date, "
            f"not the final monthly close."
        )
    breakdown.extend([
        f"Window DXYReturn = sum of monthly log returns ({dxy_terms}) = {dxy:.8f}.",
        f"Window \u0394y = sum of monthly real-yield changes ({ryc_terms}) = "
        f"{ryc:.8f} percentage points.",
        f"GPR = latest GPR observation in {ctx['gpr_month']} = {gpr:.4f}.",
        f"Predicted low/high log returns = {predictors['low_return']:.8f} / "
        f"{predictors['high_return']:.8f}.",
    ])
    return result(
        month=target_month,
        predictor_window=window,
        window_end_month=predictor,
        base_month=base_month,
        base_price=base_price,
        dxy_return=float(dxy),
        real_yield_change=float(ryc),
        gpr=float(gpr),
        gpr_month=ctx["gpr_month"],
        predicted_low_return=predictors["low_return"],
        predicted_high_return=predictors["high_return"],
        predicted_low_price=predictors["low_price"],
        predicted_high_price=predictors["high_price"],
        actual_price=actual_price(target_month),
        breakdown=breakdown,
        equation=_forecast_equation(base_price, params, predictors, "quarterly"),
    )


def _quarterly_series(f):
    """Walk-forward quarterly predictions: apply rule 5 with T = month - 2 for every month."""
    complete_windows = _quarterly_windows(f).dropna()
    first_window = complete_windows.index[0] if len(complete_windows) else None
    series = {}
    for month in f.index:
        cutoff = add_months(month, -2)
        if first_window is None or cutoff <= first_window:
            continue
        params = gold_dxy_linear_regression(data_before(cutoff), verbose=False)["quarterly"]
        forecast = _quarterly_forecast(f, params, month, cutoff)
        low_price = forecast["predicted_low_price"]
        high_price = forecast["predicted_high_price"]
        if (
            low_price is not None
            and high_price is not None
            and np.isfinite(low_price)
            and np.isfinite(high_price)
        ):
            series[month] = {
                "predicted_low_price": float(low_price),
                "predicted_high_price": float(high_price),
                "predicted_low_return": float(forecast["predicted_low_return"]),
                "predicted_high_return": float(forecast["predicted_high_return"]),
            }
    return series


def build_dashboard_predictions(target=None):
    """Fit forecasts for `target` (default: current month) for the dashboard."""
    current_month = date.today().strftime("%Y-%m")
    target = normalize_month(target) if target else current_month
    if target > current_month:
        raise ValueError(
            f"target month {target} is later than the current month {current_month}; "
            "the month before a future target may still be month-to-date."
        )

    model = gold_dxy_linear_regression(d)
    model["monthly"]["equation"] = _model_equation(model["monthly"], "monthly")
    model["quarterly"]["equation"] = _model_equation(model["quarterly"], "quarterly")
    f = _features(d)

    monthly_baseline = _historical_range_baseline(f, "monthly")
    quarterly_baseline = _historical_range_baseline(f, "quarterly")
    monthly_series, monthly_calibration = _calibrated_series(
        f,
        _monthly_series(f),
        "monthly",
        monthly_baseline["backtest_start_month"],
    )
    quarterly_series, quarterly_calibration = _calibrated_series(
        f,
        _quarterly_series(f),
        "quarterly",
        quarterly_baseline["backtest_start_month"],
    )

    monthly_label = "Current month" if target == current_month else "Target month"
    monthly_params = gold_dxy_linear_regression(data_before(target), verbose=False)["monthly"]
    forecast_next = _monthly_forecast(f, monthly_params, target, monthly_label)
    forecast_next["prediction_interval"] = _prediction_interval(
        f, forecast_next, monthly_series, "monthly", target, target
    )

    quarterly_target = add_months(target, 2)
    quarterly_params = gold_dxy_linear_regression(
        data_before(target), verbose=False
    )["quarterly"]
    forecast_three = _quarterly_forecast(f, quarterly_params, quarterly_target, target)
    forecast_three["prediction_interval"] = _prediction_interval(
        f,
        forecast_three,
        quarterly_series,
        "quarterly",
        target,
        quarterly_target,
    )

    if (
        forecast_next["predicted_low_price"] is not None
        and np.isfinite(forecast_next["predicted_low_price"])
        and forecast_next["predicted_high_price"] is not None
        and np.isfinite(forecast_next["predicted_high_price"])
    ):
        monthly_series[forecast_next["month"]] = {
            "predicted_low_price": forecast_next["predicted_low_price"],
            "predicted_high_price": forecast_next["predicted_high_price"],
            "predicted_low_return": forecast_next["predicted_low_return"],
            "predicted_high_return": forecast_next["predicted_high_return"],
            "prediction_interval": forecast_next["prediction_interval"],
        }
    if (
        forecast_three["predicted_low_price"] is not None
        and np.isfinite(forecast_three["predicted_low_price"])
        and forecast_three["predicted_high_price"] is not None
        and np.isfinite(forecast_three["predicted_high_price"])
    ):
        quarterly_series[forecast_three["month"]] = {
            "predicted_low_price": forecast_three["predicted_low_price"],
            "predicted_high_price": forecast_three["predicted_high_price"],
            "predicted_low_return": forecast_three["predicted_low_return"],
            "predicted_high_return": forecast_three["predicted_high_return"],
            "prediction_interval": forecast_three["prediction_interval"],
        }

    actual_months = d["xau_usd"].dropna()
    last_actual_month = actual_months.index[-1] if len(actual_months) else None

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current_month": current_month,
        "target_month": target,
        "last_actual_month": last_actual_month,
        "models": model,
        "prediction_interval_calibration": {
            "method": "rolling walk-forward conformal log-return calibration",
            "coverage": PREDICTION_INTERVAL_COVERAGE,
            "minimum_calibration_samples": MIN_CALIBRATION_SAMPLES,
            "maximum_calibration_samples": MAX_CALIBRATION_SAMPLES,
            "monthly": monthly_calibration,
            "quarterly": quarterly_calibration,
            "historical_range_baseline": {
                "monthly": monthly_baseline,
                "quarterly": quarterly_baseline,
            },
        },
        "monthly_series": monthly_series,
        "quarterly_series": quarterly_series,
        "actual_returns": _actual_returns(f),
        "forecasts": {
            "next_month": forecast_next,
            "three_months": forecast_three,
        },
    }


def write_predictions(predictions, path=PREDICTIONS_PATH):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def parse_args():
    parser = ArgumentParser(
        description="Predict gold prices for a target month and write predictions.json."
    )
    parser.add_argument(
        "month",
        nargs="?",
        type=normalize_month,
        default=None,
        help="Target month to predict in YYYY-MM format (default: current month).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        predictions = build_dashboard_predictions(args.month)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    write_predictions(predictions)
    for name in ("next_month", "three_months"):
        forecast = predictions["forecasts"][name]
        low = forecast["predicted_low_price"]
        high = forecast["predicted_high_price"]
        rendered = (
            f"{low:.1f}\u2013{high:.1f} USD/oz"
            if low is not None and high is not None
            else "unavailable"
        )
        print(f"{forecast['label']} ({forecast['month']}) range: {rendered}")
        interval = forecast["prediction_interval"]
        if interval is not None:
            print(
                f"  90% prediction interval: {interval['lower']:.1f}–"
                f"{interval['upper']:.1f} USD/oz "
                f"(calibrated on {interval['calibration_samples']} prior periods)"
            )
    print(f"Wrote {PREDICTIONS_PATH}")
