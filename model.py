# %%
import sys
sys.stdout = open("model.log", "w")
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout

df = pd.read_csv("dashboard_monthly.csv")[["month", "aux_usd", "btc_usd", "fed_rate_pct", "real_yield_pct"]]
df = df.rename(columns={"aux_usd": "xau"}).dropna(subset=["xau"]).reset_index(drop=True)

def prepare_dataset(df, window=12, val=6):
    btc_chg = df["btc_usd"].pct_change()
    fed_delta = df["fed_rate_pct"].diff()
    ry_delta = df["real_yield_pct"].diff()

    y = df["xau"].pct_change().shift(-1)

    frames = []
    feature_names = []
    for lag in range(1, window + 1):
        for name, series in [("btc_chg", btc_chg), ("fed_delta", fed_delta), ("ry_delta", ry_delta)]:
            fname = f"{name}_lag{lag}"
            frames.append(series.shift(lag).rename(fname))
            feature_names.append(fname)
    X = pd.concat(frames, axis=1)

    combined = pd.concat([X, y.rename("target"), df["month"], df["xau"]], axis=1)
    combined = combined.dropna().reset_index(drop=True)

    X_all = combined[feature_names].values.astype(np.float32)
    y_all = combined["target"].values.astype(np.float32)
    months_all = combined["month"].values
    curr_all = combined["xau"].values.astype(np.float32)

    X_tr, y_tr = X_all[:-val], y_all[:-val]
    X_va, y_va = X_all[-val:], y_all[-val:]

    mu_X = X_tr.mean(axis=0)
    sig_X = X_tr.std(axis=0)
    sig_X[sig_X == 0] = 1

    X_tr = ((X_tr - mu_X) / sig_X).astype(np.float32)
    X_va = ((X_va - mu_X) / sig_X).astype(np.float32)

    return X_tr, y_tr, X_va, y_va, mu_X, sig_X, feature_names, months_all, curr_all


WINDOW = 12
VAL = 6
X_tr, y_tr, X_va, y_va, mu_X, sig_X, feature_names, months_all, curr_all = prepare_dataset(df, WINDOW, VAL)

print("\nCorrelation between inputs and output (XAU next-month return, training set):")
correlations = np.corrcoef(X_tr, y_tr, rowvar=False)[-1, :-1]
for name, corr in zip(feature_names, correlations):
    print(f"  {name}: {corr:+.4f}")
# %%
w = WINDOW * 3
N_SEEDS = 5
models = []
for seed in range(N_SEEDS):
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([
        Input(shape=(w,)),
        Dense(16, activation="relu", kernel_initializer="he_normal"),
        Dropout(0.2),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    m.summary()
    early_stop = EarlyStopping(monitor="val_mae", patience=20, restore_best_weights=True)
    m.fit(X_tr, y_tr, epochs=2000, batch_size=16, validation_data=(X_va, y_va), callbacks=[early_stop], verbose=1)
    models.append(m)
    m.save(f"xau_model_seed{seed}.keras")

# %%
y_va_actual = y_va
y_va_pred = np.mean([m.predict(X_va, verbose=0).ravel() for m in models], axis=0)
curr_va = curr_all[-VAL:]
actual_price = curr_va * (1 + y_va_actual)
pred_price = curr_va * (1 + y_va_pred)
diff_price = pred_price - actual_price
va_months = months_all[-VAL:]

def last_summary(months, actual, predicted, diff):
    last = pd.DataFrame({
        "Month": months,
        "Actual": np.round(actual, 2),
        "Predicted": np.round(predicted, 2),
        "Difference": np.round(diff, 2),
    })
    print("\nLast 6 months (held-out validation/test) prediction (next month):")
    print(last.to_string(index=False))
    print(f"\nPrice MAE: {np.mean(np.abs(diff)):.2f}  Max Diff: {np.max(np.abs(diff)):.2f}")
    print(f"Return MAE: {np.mean(np.abs(y_va_pred - y_va_actual)) * 100:.2f}%")
    return last

last = last_summary(va_months, actual_price, pred_price, diff_price)
btc_vals = df["btc_usd"]
fed_vals = df["fed_rate_pct"]
ry_vals = df["real_yield_pct"]

btc_chg = btc_vals.pct_change()
fed_delta = fed_vals.diff()
ry_delta = ry_vals.diff()

feat = []
for lag in range(1, WINDOW + 1):
    feat.extend([
        btc_chg.iloc[-lag],
        fed_delta.iloc[-lag],
        ry_delta.iloc[-lag],
    ])
X_future = np.array([feat], dtype=np.float32)
X_future = ((X_future - mu_X) / sig_X).astype(np.float32)
y_future_pred = np.mean([m.predict(X_future, verbose=0).ravel() for m in models], axis=0)
future_price = df["xau"].iloc[-1] * (1 + y_future_pred)

print("\nFuture month prediction:")
print(f"{'Month':<10} {'Return':>10} {'Predicted XAU':>16}")
print("-" * 40)
print(f"{'2026-08':<10} {y_future_pred.item() * 100:>9.2f}% {future_price.item():>15.2f}")
# %%
