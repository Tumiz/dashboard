from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "dashboard_monthly.csv"
d = pd.read_csv(DATA_PATH, index_col="month")


def normalize_month(month):
    try:
        year, mm = str(month).split("-")
        period = pd.Period(year=int(year), month=int(mm), freq="M")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid month {month!r}, expected YYYY-MM") from exc
    return f"{period.year:04d}-{period.month:02d}"


def data_before(month):
    return d[d.index < normalize_month(month)]


def ln_return(series):
    return np.log(series / series.shift(1))


def _quarter_label(month):
    return f"{month[:4]}-{((int(month[5:7]) - 1) // 3) + 1:02d}"


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
            "alpha": alpha,
            "beta_dxy": beta_dxy,
            "beta_real_yield_change": beta_real_yield_change,
            "r2": r2,
        },
        "quarterly": {
            "alpha": alpha_q,
            "beta_dxy": beta_dxy_q,
            "beta_real_yield_change": beta_real_yield_change_q,
            "beta_gpr": beta_gpr_q,
            "r2": r2_q,
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


def parse_args():
    parser = ArgumentParser(description="Predict the gold price for a given month.")
    parser.add_argument(
        "month",
        type=normalize_month,
        help="Predicting month in YYYY-MM format; only data before it is used to regress.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = data_before(args.month)
    model = gold_dxy_linear_regression(data)
    predict_price(data, args.month, **{k: model["monthly"][k] for k in
                                       ("alpha", "beta_dxy", "beta_real_yield_change")})
    predict_quarterly_price(data, args.month, **{k: model["quarterly"][k] for k in
                                                 ("alpha", "beta_dxy", "beta_real_yield_change", "beta_gpr")})