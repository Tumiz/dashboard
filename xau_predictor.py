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


def _quarter_label(month):
    return f"{month[:4]}-{((int(month[5:7]) - 1) // 3) + 1:02d}"


def _quarter_end(quarter):
    return f"{quarter[:4]}-{3 * int(quarter[5:7]):02d}"


def _next_quarter(quarter):
    year, q = int(quarter[:4]), int(quarter[5:7])
    return f"{year + 1}-01" if q == 4 else f"{year}-{q + 1:02d}"


def _quarterly_data(data):
    source = data[["xau_usd", "twexb", "real_yield_pct", "gpr"]].dropna()
    df = pd.DataFrame({
        "gold_return": ln_return(source["xau_usd"]),
        "dxy_return": ln_return(source["twexb"]),
        "real_yield_change": source["real_yield_pct"].diff(),
        "gpr": source["gpr"],
    }).dropna()
    df["quarter"] = df.index.map(_quarter_label)
    return df.groupby("quarter").agg({
        "gold_return": "sum",
        "dxy_return": "sum",
        "real_yield_change": "sum",
        "gpr": "last",
    })


def actual_price(month):
    if month in d.index and pd.notna(d.loc[month, "xau_usd"]):
        return float(d.loc[month, "xau_usd"])
    return None


def actual_quarter_end_price(quarter):
    q = d[d.index.map(_quarter_label) == quarter]["xau_usd"].dropna()
    return float(q.iloc[-1]) if len(q) else None


def gold_dxy_linear_regression(data):
    df = data[["xau_usd", "twexb", "real_yield_pct"]].dropna()
    real_yield_change = df["real_yield_pct"].diff()
    reg = pd.DataFrame({
        "gold_return": ln_return(df["xau_usd"]),
        "dxy_return": ln_return(df["twexb"]),
        "real_yield_change": real_yield_change,
    }).dropna()

    X = reg[["dxy_return", "real_yield_change"]].to_numpy()
    y = reg["gold_return"].to_numpy()
    X_design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    alpha, beta_dxy, beta_real_yield_change = coef

    y_hat = X_design @ coef
    r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

    print(
        f"Monthly model: Gold_Return_t = {alpha:.4f} + {beta_dxy:.4f} * DXY_Return_t + "
        f"{beta_real_yield_change:.4f} * RealYield_Change_t"
    )
    print(f"Monthly R-squared = {r2:.4f}, n = {len(reg)}")

    quarterly = _quarterly_data(data)

    X_q = quarterly[["dxy_return", "real_yield_change", "gpr"]].to_numpy()
    y_q = quarterly["gold_return"].to_numpy()
    X_q_design = np.column_stack([np.ones(len(X_q)), X_q])
    coef_q, *_ = np.linalg.lstsq(X_q_design, y_q, rcond=None)
    alpha_q, beta_dxy_q, beta_real_yield_change_q, beta_gpr_q = coef_q
    y_hat_q = X_q_design @ coef_q
    r2_q = 1 - np.sum((y_q - y_hat_q) ** 2) / np.sum((y_q - y_q.mean()) ** 2)

    print(
        f"Quarterly model: Gold_Return_q = {alpha_q:.4f} + {beta_dxy_q:.4f} * DXY_Return_q + "
        f"{beta_real_yield_change_q:.4f} * RealYield_Change_q + {beta_gpr_q:.6f} * GPR_q"
    )
    print(f"Quarterly R-squared = {r2_q:.4f}, n = {len(quarterly)}")

    return {
        "monthly": {
            "alpha": float(alpha),
            "beta_dxy": float(beta_dxy),
            "beta_real_yield_change": float(beta_real_yield_change),
            "r2": float(r2),
            "n": int(len(reg)),
        },
        "quarterly": {
            "alpha": float(alpha_q),
            "beta_dxy": float(beta_dxy_q),
            "beta_real_yield_change": float(beta_real_yield_change_q),
            "beta_gpr": float(beta_gpr_q),
            "r2": float(r2_q),
            "n": int(len(quarterly)),
        },
    }


def predict_price(data, predict_month, alpha, beta_dxy, beta_real_yield_change):
    gold = data["xau_usd"]
    dxy_ret = ln_return(data["twexb"])
    real_yield_change = data["real_yield_pct"].diff()

    last_month = gold.dropna().index[-1]
    print(f"Last month used for regression: {last_month}, gold price: {gold.loc[last_month]} USD")
    latest_dxy_ret = dxy_ret.dropna().iloc[-1]
    latest_real_yield_change = real_yield_change.dropna().iloc[-1]
    forecast_return = (
        alpha + beta_dxy * latest_dxy_ret + beta_real_yield_change * latest_real_yield_change
    )
    forecast_price = gold.loc[last_month] * np.exp(forecast_return)

    actual = actual_price(predict_month)
    actual_str = f"{actual:.1f} USD" if actual is not None else "N/A"
    print(
        f"Monthly forecast ({predict_month}): {forecast_price:.1f} USD, "
        f"return={forecast_return:.4f}, actual={actual_str}"
    )
    return forecast_price


def predict_quarterly_price(data, predict_month, alpha, beta_dxy, beta_real_yield_change, beta_gpr):
    gold = data["xau_usd"]
    quarterly = _quarterly_data(data)

    forecast_quarter = _quarter_label(predict_month)
    prev = quarterly[quarterly.index < forecast_quarter]
    if prev.empty:
        raise SystemExit(f"No completed quarter before {forecast_quarter} to build a forecast.")
    prev = prev.iloc[-1]

    quarterly_return = (
        alpha
        + beta_dxy * prev["dxy_return"]
        + beta_real_yield_change * prev["real_yield_change"]
        + beta_gpr * prev["gpr"]
    )
    last_month = gold.dropna().index[-1]
    forecast_price = gold.loc[last_month] * np.exp(quarterly_return)

    actual = actual_quarter_end_price(forecast_quarter)
    actual_str = f"{actual:.1f} USD" if actual is not None else "N/A"
    print(
        f"Quarterly forecast ({forecast_quarter} quarter-end): {forecast_price:.1f} USD, "
        f"return={quarterly_return:.4f}, actual={actual_str}"
    )
    return forecast_price


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
        "quarter": [_quarter_label(m) for m in data.index],
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


def _monthly_equation(xau, alpha, beta_dxy, beta_real_yield_change, dxy, ryc, price):
    return (
        "\\[\\widehat{X}_{t+1}=" + _tex_number(xau, 2)
        + "\\exp\\left(" + _latex_alpha(alpha)
        + _latex_op(beta_dxy) + "\\left(" + _tex_number(dxy) + "\\right)"
        + _latex_op(beta_real_yield_change) + "\\left(" + _tex_number(ryc) + "\\right)"
        + "\\right)\\approx " + _tex_number(price, 2) + "\\]"
    )


def _quarterly_equation(xau, params, dxy, ryc, gpr, price):
    return (
        "\\[\\widehat{X}_{q+1}=" + _tex_number(xau, 2)
        + "\\exp\\left(" + _latex_alpha(params["alpha"])
        + _latex_op(params["beta_dxy"]) + "\\left(" + _tex_number(dxy) + "\\right)"
        + _latex_op(params["beta_real_yield_change"]) + "\\left(" + _tex_number(ryc) + "\\right)"
        + _latex_op(params["beta_gpr"], 6) + "\\left(" + _tex_number(gpr) + "\\right)"
        + "\\right)\\approx " + _tex_number(price, 2) + "\\]"
    )


def _monthly_series(f, params):
    """In-sample predicted price/return for every month with available inputs."""
    alpha = params["alpha"]
    beta_dxy = params["beta_dxy"]
    beta_real_yield_change = params["beta_real_yield_change"]
    series = {}
    for month in f.index:
        row = f.loc[month]
        if any(pd.isna(row[k]) for k in
               ("dxy_return", "ryc", "prev_gold", "twexb", "ry", "prev_twexb", "prev_ry")):
            continue
        predicted_return = alpha + beta_dxy * row["dxy_return"] + beta_real_yield_change * row["ryc"]
        predicted_price = row["prev_gold"] * np.exp(predicted_return)
        series[month] = {
            "predicted_price": float(predicted_price),
            "predicted_return": float(predicted_return),
        }
    return series


def _monthly_forecast(f, params, target, label):
    """Forecast `target` month; predictors come from the latest usable inputs."""
    gold = f["gold"]
    has_macro = f["twexb"].notna() & f["ry"].notna()

    before = gold[gold.index < target].dropna()
    base_month = before.index[-1] if len(before) else None
    base_price = float(before.iloc[-1]) if base_month is not None else None

    input_candidates = f.index[(f.index <= target) & has_macro]
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

    alpha = params["alpha"]
    beta_dxy = params["beta_dxy"]
    beta_real_yield_change = params["beta_real_yield_change"]

    dxy = ryc = None
    predicted_return = predicted_price = None
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
            predicted_return = alpha + beta_dxy * dxy + beta_real_yield_change * ryc
            predicted_price = float(base_price * np.exp(predicted_return))
            breakdown.append(
                f"Predicted log return = {_alpha_txt(alpha)} {_term_txt(beta_dxy)} \u00d7 "
                f"({dxy:.8f}) {_term_txt(beta_real_yield_change)} \u00d7 ({ryc:.8f}) = "
                f"{predicted_return:.8f}."
            )
            equation = _monthly_equation(
                base_price, alpha, beta_dxy, beta_real_yield_change, dxy, ryc, predicted_price
            )
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
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "actual_price": actual_price(target),
        "breakdown": breakdown,
        "equation": equation,
    }


def _quarter_context(f):
    """Per-quarter aggregates, month-by-month terms and predictor metadata."""
    quarters = {}
    for month in f.index:
        row = f.loc[month]
        q = row["quarter"]
        ctx = quarters.setdefault(q, {
            "dxy": 0.0,
            "ryc": 0.0,
            "dxy_n": 0,
            "ryc_n": 0,
            "gpr": None,
            "gpr_month": None,
            "dxy_terms": [],
            "ryc_terms": [],
        })
        if pd.notna(row["dxy_return"]):
            value = float(row["dxy_return"])
            ctx["dxy"] += value
            ctx["dxy_n"] += 1
            ctx["dxy_terms"].append({
                "month": month,
                "previous_month": row["prev_twexb_month"],
                "previous": float(row["prev_twexb"]),
                "current": float(row["twexb"]),
                "value": value,
            })
        if pd.notna(row["ryc"]):
            ctx["ryc"] += float(row["ryc"])
            ctx["ryc_n"] += 1
            ctx["ryc_terms"].append({
                "month": month,
                "previous_month": row["prev_ry_month"],
                "previous": float(row["prev_ry"]),
                "current": float(row["ry"]),
                "value": float(row["ryc"]),
            })
        if pd.notna(row["gpr"]):
            ctx["gpr"] = float(row["gpr"])
            ctx["gpr_month"] = month
    return quarters


def _quarterly_forecast(f, quarters, params, label="Three months"):
    """Three-months-ahead forecast from the quarterly model."""
    full = sorted(q for q, ctx in quarters.items() if ctx["dxy_n"] >= 3)
    predictor = full[-1] if full else None
    no_quarter = (
        "No complete quarter with the required DXY return, real-yield changes, "
        "and GPR value is available."
    )

    def result(**overrides):
        base = {
            "month": None,
            "label": label,
            "predictor_quarter": predictor,
            "forecast_quarter": None,
            "quarter_end_month": None,
            "base_price": None,
            "dxy_return": None,
            "real_yield_change": None,
            "gpr": None,
            "gpr_month": None,
            "predicted_return": None,
            "predicted_price": None,
            "actual_price": None,
            "breakdown": [],
            "equation": None,
        }
        base.update(overrides)
        return base

    if predictor is None:
        return result(breakdown=[no_quarter])

    ctx = quarters[predictor]
    dxy = ctx["dxy"] if ctx["dxy_n"] else np.nan
    ryc = ctx["ryc"] if ctx["ryc_n"] else np.nan
    gpr = ctx["gpr"] if ctx["gpr"] is not None else np.nan
    if not (np.isfinite(dxy) and np.isfinite(ryc) and np.isfinite(gpr)):
        return result(breakdown=[no_quarter])

    predicted_return = (
        params["alpha"]
        + params["beta_dxy"] * dxy
        + params["beta_real_yield_change"] * ryc
        + params["beta_gpr"] * gpr
    )

    quarter_end_month = _quarter_end(predictor)
    forecast_quarter = _next_quarter(predictor)
    target = _quarter_end(forecast_quarter)
    base_price = None
    if quarter_end_month in f.index:
        value = f.at[quarter_end_month, "gold"]
        base_price = float(value) if pd.notna(value) else None

    quarter_label = f"Q{int(predictor[5:7])} {predictor[:4]}"
    if base_price is None:
        breakdown = [
            f"Forecast inputs for {quarter_label} are incomplete; "
            f"a quarter-end XAU/USD value is also required."
        ]
        return result(
            forecast_quarter=forecast_quarter,
            quarter_end_month=quarter_end_month,
            breakdown=breakdown,
        )

    predicted_price = float(base_price * np.exp(predicted_return))
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
        f"{quarter_label} DXYReturn_q = sum of monthly log returns ({dxy_terms}) = {dxy:.8f}.",
        f"{quarter_label} \u0394y_q = sum of monthly real-yield changes ({ryc_terms}) = "
        f"{ryc:.8f} percentage points.",
        f"GPR_q = latest GPR observation in {ctx['gpr_month']} = {gpr:.4f}.",
        f"X_q = quarter-end XAU/USD in {quarter_end_month} = {base_price:.2f} USD/oz.",
        f"Predicted log return = {_alpha_txt(params['alpha'])} "
        f"{_term_txt(params['beta_dxy'])} \u00d7 ({dxy:.8f}) "
        f"{_term_txt(params['beta_real_yield_change'])} \u00d7 ({ryc:.8f}) "
        f"{_term_txt(params['beta_gpr'], 6)} \u00d7 ({gpr:.4f}) = {predicted_return:.8f}.",
    ]
    return result(
        month=target,
        forecast_quarter=forecast_quarter,
        quarter_end_month=quarter_end_month,
        base_price=base_price,
        dxy_return=float(dxy),
        real_yield_change=float(ryc),
        gpr=float(gpr),
        gpr_month=ctx["gpr_month"],
        predicted_return=float(predicted_return),
        predicted_price=predicted_price,
        actual_price=actual_quarter_end_price(forecast_quarter),
        breakdown=breakdown,
        equation=_quarterly_equation(base_price, params, dxy, ryc, gpr, predicted_price),
    )


def build_dashboard_predictions():
    """Fit on all available data and produce the dashboard's predictions."""
    model = gold_dxy_linear_regression(d)
    f = _features(d)
    monthly_params = model["monthly"]
    quarterly_params = model["quarterly"]

    series = _monthly_series(f, monthly_params)
    quarters = _quarter_context(f)

    quarterly_returns = {}
    for q, ctx in quarters.items():
        if ctx["dxy_n"] == 0 or ctx["ryc_n"] == 0 or ctx["gpr"] is None:
            continue
        value = (
            quarterly_params["alpha"]
            + quarterly_params["beta_dxy"] * ctx["dxy"]
            + quarterly_params["beta_real_yield_change"] * ctx["ryc"]
            + quarterly_params["beta_gpr"] * ctx["gpr"]
        )
        quarterly_returns[_quarter_end(q)] = float(value)

    actual_months = d["xau_usd"].dropna()
    if actual_months.empty:
        raise SystemExit("No actual XAU/USD price available to anchor a forecast.")
    base_month = actual_months.index[-1]
    current_month = date.today().strftime("%Y-%m")

    next_month = base_month if base_month == current_month else add_months(base_month, 1)
    next_label = "Current month" if next_month == current_month else "Next month"
    forecast_next = _monthly_forecast(f, monthly_params, next_month, next_label)
    forecast_three = _quarterly_forecast(f, quarters, quarterly_params)

    if forecast_next["predicted_price"] is not None and np.isfinite(forecast_next["predicted_price"]):
        series[forecast_next["month"]] = {
            "predicted_price": forecast_next["predicted_price"],
            "predicted_return": forecast_next["predicted_return"],
        }
    if forecast_three["predicted_price"] is not None and np.isfinite(forecast_three["predicted_price"]):
        series[forecast_three["month"]] = {
            "predicted_price": forecast_three["predicted_price"],
            "predicted_return": None,
        }

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current_month": current_month,
        "last_actual_month": base_month,
        "models": model,
        "monthly_series": series,
        "quarterly_returns": quarterly_returns,
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
        description="Predict gold prices; without a month it writes predictions.json for the dashboard."
    )
    parser.add_argument(
        "month",
        nargs="?",
        type=normalize_month,
        default=None,
        help="Optional backtest month in YYYY-MM format; only data before it is used to regress.",
    )
    return parser.parse_args()


def backtest(month):
    data = data_before(month)
    model = gold_dxy_linear_regression(data)
    predict_price(
        data, month,
        model["monthly"]["alpha"],
        model["monthly"]["beta_dxy"],
        model["monthly"]["beta_real_yield_change"],
    )
    predict_quarterly_price(
        data, month,
        model["quarterly"]["alpha"],
        model["quarterly"]["beta_dxy"],
        model["quarterly"]["beta_real_yield_change"],
        model["quarterly"]["beta_gpr"],
    )


if __name__ == "__main__":
    args = parse_args()
    if args.month:
        backtest(args.month)
    else:
        predictions = build_dashboard_predictions()
        write_predictions(predictions)
        for name in ("next_month", "three_months"):
            forecast = predictions["forecasts"][name]
            price = forecast["predicted_price"]
            target = forecast["month"] or forecast.get("forecast_quarter")
            rendered = f"{price:.1f} USD" if price is not None else "unavailable"
            print(f"{forecast['label']} ({target}) predicted: {rendered}")
        print(f"Wrote {PREDICTIONS_PATH}")
