# %%
import sys
sys.stdout = open("model.log", "w")
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense

df = pd.read_csv("dashboard_monthly.csv")[["month", "gold_close", "btc_usd", "fed_rate_pct", "aux_usd", "xag_usd"]]
df = df.rename(columns={"gold_close": "gold"}).dropna(subset=["gold"]).reset_index(drop=True)
df["xau_xag"] = df["aux_usd"] / df["xag_usd"]

WINDOW = 12
OX, Oy = [], []
for i in range(WINDOW, len(df) - 1):
    feat = [df.loc[i, "gold"]]
    for t in range(i - WINDOW, i):
        feat.extend([df.loc[t, "btc_usd"], df.loc[t, "fed_rate_pct"], df.loc[t, "xau_xag"]])
    OX.append(feat)
    Oy.append(df.loc[i + 1, "gold"])

OX, Oy = np.array(OX, dtype=np.float32), np.array(Oy, dtype=np.float32)

X_df = pd.DataFrame(OX)

mu_X = X_df.mean()
sig_X = X_df.std().replace(0, 1)

X_df = (X_df - mu_X) / sig_X
X = X_df.values.astype(np.float32)
y = Oy.astype(np.float32)

mu_X, sig_X = mu_X.values, sig_X.values

np.random.seed(42)
shuffle_idx = np.random.permutation(len(X)-1)
RX, Ry = X[:-1][shuffle_idx], y[:-1][shuffle_idx]

print("\nCorrelation between inputs and output (gold):")
feature_names = []
for t in range(WINDOW):
    feature_names.extend([f"btc_usd_t-{WINDOW - t}", f"fed_rate_t-{WINDOW - t}", f"xau_xag_t-{WINDOW - t}"])
correlations = np.corrcoef(X, y, rowvar=False)[-1, :-1]
for name, corr in zip(feature_names, correlations):
    print(f"  {name}: {corr:+.4f}")
# %%
w = WINDOW * 3 + 1
model = keras.Sequential([
    Input(shape=(w,)),
    Dense(16, activation="relu", kernel_initializer="he_normal"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
early_stop = EarlyStopping(monitor="val_mae", patience=20, restore_best_weights=True)
model.fit(RX, Ry, epochs=2000, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=1)
model.save("gold_model.keras")

# %%
X_last24 = X[-24:]
y_last24_actual = y[-24:]
y_last24_pred = model.predict(X_last24).ravel()
diff_last24 = y_last24_pred - y_last24_actual
next_months = np.append(df["month"].iloc[-23:].values, ["2026-07"])

def last24_summary(months, actual, predicted, diff):
    last24 = pd.DataFrame({
        "Month": months,
        "Actual": np.round(actual, 2),
        "Predicted": np.round(predicted, 2),
        "Difference": np.round(diff, 2),
    })
    print("\nLast 24 months prediction (next month):")
    print(last24.to_string(index=False))
    print(f"\nMAE: {np.mean(np.abs(diff)):.2f}  Max Diff: {np.max(np.abs(diff)):.2f}")
    return last24

last24 = last24_summary(next_months, y_last24_actual, y_last24_pred, diff_last24)
btc_vals = df["btc_usd"]
fed_vals = df["fed_rate_pct"]
xau_xag_vals = df["xau_xag"]

X_future = []
end = len(df)
start = end - WINDOW
feat = [df["gold"].iloc[-1]]
for t in range(start, end):
    feat.extend([btc_vals[t], fed_vals[t], xau_xag_vals[t]])
X_future.append(feat)
print(X_future)

X_future = np.array(X_future, dtype=np.float32)
X_future = ((X_future - mu_X) / sig_X).astype(np.float32)
y_future_pred = model.predict(X_future).ravel()

print("\nFuture ` month prediction:")
print(f"{'Month':<10} {'Predicted':>10}")
print("-" * 21)
print(f"{'2026-08':<10} {y_future_pred.item():>10.2f}")
# %%
