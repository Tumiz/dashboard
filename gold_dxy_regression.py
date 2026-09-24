from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "dashboard_monthly.csv"
d = pd.read_csv(DATA_PATH, index_col="month")


def ln_return(series):
    return np.log(series / series.shift(1))


def next_month(m):
    y, mm = int(m[:4]), int(m[5:7])
    y2, m2 = (y + 1, 1) if mm == 12 else (y, mm + 1)
    return f"{y2:04d}-{m2:02d}"


def gold_dxy_linear_regression():
    df = d[["xau_usd", "twexb", "real_yield_pct"]].dropna()
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
        f"Gold_Return_t = {alpha:.4f} + {beta_dxy:.4f} * DXY_Return_t + "
        f"{beta_real_yield_change:.4f} * RealYield_Change_t"
    )
    print(f"R-squared = {r2:.4f}, n = {len(reg)}")

    return alpha, beta_dxy, beta_real_yield_change, r2


def generate_predicted_prices(alpha, beta_dxy, beta_real_yield_change, output=DATA_PATH):
    gold = d["xau_usd"]
    dxy_ret = ln_return(d["twexb"])
    real_yield_change = d["real_yield_pct"].diff()

    actual_ret = ln_return(gold)
    pred_ret = alpha + beta_dxy * dxy_ret + beta_real_yield_change * real_yield_change
    pred_price = (gold.shift(1) * np.exp(pred_ret)).round(1)

    output_path = Path(output)
    out = pd.read_csv(output_path, index_col="month")
    for col in ["actual_return", "pred_return", "pred_price"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out = out[~out.index.duplicated(keep="last")]

    out = out.join(actual_ret.rename("actual_return").to_frame(), how="outer")
    out = out.join(pred_ret.rename("pred_return").to_frame(), how="outer")
    out = out.join(pred_price.rename("pred_price").to_frame(), how="outer")

    last_month = gold.dropna().index[-1]
    forecast_month = next_month(last_month)
    latest_dxy_ret = dxy_ret.dropna().iloc[-1]
    latest_real_yield_change = real_yield_change.dropna().iloc[-1]
    forecast_return = (
        alpha + beta_dxy * latest_dxy_ret + beta_real_yield_change * latest_real_yield_change
    )
    forecast_price = gold.loc[last_month] * np.exp(forecast_return)
    out = out.drop(index=[forecast_month], errors="ignore")
    out = pd.concat([
        out,
        pd.DataFrame({
            "actual_return": [np.nan],
            "pred_return": [forecast_return],
            "pred_price": [forecast_price],
        }, index=pd.Index([forecast_month], name="month")),
    ])
    out = out.sort_index().reset_index()
    out = out[["month"] + [c for c in out.columns if c != "month"]]
    out.to_csv(output_path, index=False)

    print(f"Wrote actual_return/pred_return/pred_price for {actual_ret.notna().sum()} months")
    print(f"Next-month forecast ({forecast_month}): {forecast_price} USD, return={forecast_return:.4f}")


if __name__ == "__main__":
    alpha, beta_dxy, beta_real_yield_change, r2 = gold_dxy_linear_regression()
    generate_predicted_prices(alpha, beta_dxy, beta_real_yield_change, output=DATA_PATH)