import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

def standardize_fit(x):
    mu = x.mean(axis=0)
    sig = x.std(axis=0)
    if np.ndim(sig) == 0:
        sig = np.float32(1) if sig == 0 else sig
    else:
        sig[sig == 0] = 1
    return mu, sig

def standardize_apply(x, mu, sig):
    return (x - mu) / sig

def inverse_standardize(y, mu, sig):
    return y * sig + mu

gold = pd.read_csv("sge_gold_monthly.csv")[["month", "close"]].rename(columns={"close": "gold"})
btc = pd.read_csv("btc_monthly.csv")[["month", "btc_usd"]]
fed = pd.read_csv("fed_rates_monthly.csv")[["month", "fed_rate_pct"]]

df = gold.merge(btc, on="month", how="left").merge(fed, on="month", how="left")
df = df.dropna().reset_index(drop=True)

WINDOW = 15
X, y = [], []
for i in range(WINDOW, len(df) - 1):
    feat = []
    for t in range(i - WINDOW, i):
        feat.extend([df.loc[t, "btc_usd"], df.loc[t, "fed_rate_pct"]])
    X.append(feat)
    y.append(df.loc[i + 1, "gold"])

X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

mu_X, sig_X = standardize_fit(X)
mu_y, sig_y = standardize_fit(y)
X = standardize_apply(X, mu_X, sig_X)
y = standardize_apply(y, mu_y, sig_y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
w = WINDOW * 2
model = keras.Sequential([
    Input(shape=(w,)),
    Dense(w, activation="relu"),
    Dense(w, activation="relu"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=1)

loss, mae = model.evaluate(X_test, y_test)
y_pred = inverse_standardize(model.predict(X_test).ravel(), mu_y, sig_y)
y_true = inverse_standardize(y_test, mu_y, sig_y)

print("\nTest results:")
for i in range(len(y_true)):
    print(f"  Actual: {y_true[i]:.2f}  Predicted: {y_pred[i]:.2f}  Diff: {y_pred[i]-y_true[i]:.2f}")
print(f"\nMAE: {np.mean(np.abs(y_pred - y_true)):.2f}")
