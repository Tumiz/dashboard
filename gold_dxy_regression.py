import numpy as np
import pandas as pd

d = pd.read_csv("dashboard_monthly.csv", index_col="month")


def ln_return(series):
    return np.log(series / series.shift(1))


def next_month(m):
    y, mm = int(m[:4]), int(m[5:7])
    y2, m2 = (y + 1, 1) if mm == 12 else (y, mm + 1)
    return f"{y2:04d}-{m2:02d}"


def gold_dxy_linear_regression():
    df = d[["xau_usd", "twexb"]].dropna()
    reg = pd.DataFrame({
        "gold_return": ln_return(df["xau_usd"]),
        "dxy_return": ln_return(df["twexb"]),
    }).dropna()

    x = reg["dxy_return"].to_numpy()
    y = reg["gold_return"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)

    y_hat = alpha + beta * x
    r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

    print(f"Gold_Return_t = {alpha:.4f} + {beta:.4f} * DXY_Return_t")
    print(f"R-squared = {r2:.4f}, n = {len(reg)}")

    return alpha, beta, r2


def generate_predicted_prices(alpha, beta, output="dashboard_monthly.csv"):
    gold = d["xau_usd"]
    dxy_ret = ln_return(d["twexb"])

    actual_ret = ln_return(gold)
    pred_ret = alpha + beta * dxy_ret
    pred_price = (gold.shift(1) * np.exp(pred_ret)).round(1)

    out = pd.read_csv(output, index_col="month")
    for col in ["actual_return", "pred_return", "pred_price"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out = out[~out.index.duplicated(keep="last")]

    out = out.join(actual_ret.rename("actual_return").to_frame(), how="outer")
    out = out.join(pred_ret.rename("pred_return").to_frame(), how="outer")
    out = out.join(pred_price.rename("pred_price").to_frame(), how="outer")

    last_month = gold.dropna().index[-1]
    forecast_month = next_month(last_month)
    forecast_return = float(alpha)
    forecast = round(gold.loc[last_month] * np.exp(forecast_return), 1)
    out = out.drop(index=[forecast_month], errors="ignore")
    out = pd.concat([
        out,
        pd.DataFrame({
            "actual_return": [np.nan],
            "pred_return": [forecast_return],
            "pred_price": [forecast],
        }, index=pd.Index([forecast_month], name="month")),
    ])
    out = out.sort_index().reset_index()
    out = out[["month"] + [c for c in out.columns if c != "month"]]
    out.to_csv(output, index=False)

    print(f"Wrote actual_return/pred_return/pred_price for {actual_ret.notna().sum()} months")
    print(f"Next-month forecast ({forecast_month}): {forecast} USD, return={forecast_return:.4f}")


alpha, beta, r2 = gold_dxy_linear_regression()
generate_predicted_prices(alpha, beta)