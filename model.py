import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense


gold = pd.read_csv("sge_gold_monthly.csv")[["month", "close"]].rename(columns={"close": "gold"})
btc = pd.read_csv("btc_monthly.csv")[["month", "btc_usd"]]
fed = pd.read_csv("fed_rates_monthly.csv")[["month", "fed_rate_pct"]]
aux = pd.read_csv("aux_monthly.csv")[["month", "aux_usd"]]
xag = pd.read_csv("xag_monthly.csv")[["month", "xag_usd"]]

df = gold.merge(btc, on="month", how="left").merge(fed, on="month", how="left")
df = df.merge(aux, on="month", how="left").merge(xag, on="month", how="left")
df["xau_xag"] = df["aux_usd"] / df["xag_usd"]
df = df.dropna().reset_index(drop=True)

WINDOW = 12
X, y = [], []
for i in range(WINDOW, len(df) - 1):
    feat = []
    for t in range(i - WINDOW, i):
        feat.extend([df.loc[t, "btc_usd"], df.loc[t, "fed_rate_pct"], df.loc[t, "xau_xag"]])
    X.append(feat)
    y.append(df.loc[i + 1, "gold"])

X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_df = pd.DataFrame(X)

mu_X = X_df.mean()
sig_X = X_df.std().replace(0, 1)

X_df = (X_df - mu_X) / sig_X
X = X_df.values.astype(np.float32)
y = y.astype(np.float32)

mu_X, sig_X = mu_X.values, sig_X.values

np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
RX, Ry = X[shuffle_idx], y[shuffle_idx]

print("\nCorrelation between inputs and output (gold):")
feature_names = []
for t in range(WINDOW):
    feature_names.extend([f"btc_usd_t-{WINDOW - t}", f"fed_rate_t-{WINDOW - t}", f"xau_xag_t-{WINDOW - t}"])
correlations = np.corrcoef(X, y, rowvar=False)[-1, :-1]
for name, corr in zip(feature_names, correlations):
    print(f"  {name}: {corr:+.4f}")

w = WINDOW * 3
model = keras.Sequential([
    Input(shape=(w,)),
    Dense(w, activation="relu", kernel_initializer="he_normal"),
    Dense(w, activation="relu", kernel_initializer="he_normal"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

early_stop = EarlyStopping(monitor="val_mae", patience=50, restore_best_weights=True)
model.fit(RX, Ry, epochs=500, batch_size=16, validation_split=0.1, callbacks=[early_stop], verbose=1)

X_last24 = X[-24:]
y_last24_actual = y[-24:]
y_last24_pred = model.predict(X_last24).ravel()
last24_months = df["month"].iloc[-24:].values

print("\nLast 24 months prediction:")
print(f"{'Month':<10} {'Actual':>10} {'Predicted':>10} {'Diff':>10}")
print("-" * 42)
for month, actual, pred in zip(last24_months, y_last24_actual, y_last24_pred):
    print(f"{month:<10} {actual:>10.2f} {pred:>10.2f} {pred - actual:>10.2f}")
print(f"\nMAE: {np.mean(np.abs(y_last24_pred - y_last24_actual)):.2f}")

print("\nFeature windows for Jun vs Jul 2026 predictions:")
print(f"{'Month':<10} {'btc_usd':>10} {'fed_rate':>10} {'xau_xag':>10}")
print("-" * 42)
for label, x_idx in [("Jun 2026", -2), ("Jul 2026", -1)]:
    k = x_idx + len(X)
    print(f"\n{label} window:")
    for t in range(k, k + WINDOW):
        print(f"{df.loc[t, 'month']:<10} {df.loc[t, 'btc_usd']:>10.2f} {df.loc[t, 'fed_rate_pct']:>10.2f} {df.loc[t, 'xau_xag']:>10.2f}")

future_months = pd.date_range(start="2026-07", periods=1, freq="MS")

last_btc = df["btc_usd"].iloc[-1]
last_fed = df["fed_rate_pct"].iloc[-1]
last_xau_xag = df["xau_xag"].iloc[-1]

btc_vals = df["btc_usd"].tolist() + [last_btc]
fed_vals = df["fed_rate_pct"].tolist() + [last_fed]
xau_xag_vals = df["xau_xag"].tolist() + [last_xau_xag]

X_future = []
for i in range(1):
    end = len(df) + i
    start = end - WINDOW
    feat = []
    for t in range(start, end):
        feat.extend([btc_vals[t], fed_vals[t], xau_xag_vals[t]])
    X_future.append(feat)

X_future = np.array(X_future, dtype=np.float32)
X_future = ((X_future - mu_X) / sig_X).astype(np.float32)
y_future_pred = model.predict(X_future).ravel()

print("\nFuture 3 months prediction:")
print(f"{'Month':<10} {'Predicted':>10}")
print("-" * 21)
for month, pred in zip(future_months.strftime("%Y-%m"), y_future_pred):
    print(f"{month:<10} {pred:>10.2f}")