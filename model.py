# %%
import sys
sys.stdout = open("model.log", "w")
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout

keras.utils.set_random_seed(42)

df = pd.read_csv("dashboard_monthly.csv")[["month", "aux_usd", "btc_usd", "fed_rate_pct", "xag_usd"]]
df = df.rename(columns={"aux_usd": "xau"}).dropna(subset=["xau"]).reset_index(drop=True)
df["xau_xag"] = df["xau"] / df["xag_usd"]

WINDOW = 12
VAL = 6
OX, Oy, O_months, O_curr = [], [], [], []
for i in range(WINDOW + 1, len(df) - 1):
    feat = []
    for t in range(i - WINDOW, i):
        feat.extend([
            df.loc[t, "btc_usd"] / df.loc[t - 1, "btc_usd"] - 1,
            df.loc[t, "fed_rate_pct"] - df.loc[t - 1, "fed_rate_pct"],
            df.loc[t, "xau_xag"] / df.loc[t - 1, "xau_xag"] - 1,
        ])
    OX.append(feat)
    Oy.append(df.loc[i + 1, "xau"] / df.loc[i, "xau"] - 1)
    O_months.append(df.loc[i + 1, "month"])
    O_curr.append(df.loc[i, "xau"])

OX, Oy = np.array(OX, dtype=np.float32), np.array(Oy, dtype=np.float32)
O_months = np.array(O_months)
O_curr = np.array(O_curr, dtype=np.float32)
keep = ~np.isnan(OX).any(axis=1)
OX, Oy, O_months, O_curr = OX[keep], Oy[keep], O_months[keep], O_curr[keep]

X_tr, y_tr = OX[:-VAL], Oy[:-VAL]
X_va, y_va = OX[-VAL:], Oy[-VAL:]

tr_df = pd.DataFrame(X_tr)
mu_X = tr_df.mean()
sig_X = tr_df.std().replace(0, 1)

X_tr = ((tr_df - mu_X) / sig_X).values.astype(np.float32)
X_va = ((pd.DataFrame(X_va) - mu_X) / sig_X).values.astype(np.float32)
y_tr = y_tr.astype(np.float32)
y_va = y_va.astype(np.float32)

mu_X, sig_X = mu_X.values, sig_X.values

print("\nCorrelation between inputs and output (XAU next-month return, training set):")
feature_names = []
for t in range(WINDOW):
    feature_names.extend([f"btc_chg_t-{WINDOW - t}", f"fed_delta_t-{WINDOW - t}", f"xau_xag_chg_t-{WINDOW - t}"])
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
curr_va = O_curr[-VAL:]
actual_price = curr_va * (1 + y_va_actual)
pred_price = curr_va * (1 + y_va_pred)
diff_price = pred_price - actual_price
va_months = O_months[-VAL:]

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
xau_xag_vals = df["xau_xag"]

X_future = []
end = len(df)
start = end - WINDOW
feat = []
for t in range(start, end):
    feat.extend([
        btc_vals[t] / btc_vals[t - 1] - 1,
        fed_vals[t] - fed_vals[t - 1],
        xau_xag_vals[t] / xau_xag_vals[t - 1] - 1,
    ])
X_future.append(feat)

X_future = np.array(X_future, dtype=np.float32)
X_future = ((X_future - mu_X) / sig_X).astype(np.float32)
y_future_pred = np.mean([m.predict(X_future, verbose=0).ravel() for m in models], axis=0)
future_price = df["xau"].iloc[-1] * (1 + y_future_pred)

print("\nFuture month prediction:")
print(f"{'Month':<10} {'Return':>10} {'Predicted XAU':>16}")
print("-" * 40)
print(f"{'2026-08':<10} {y_future_pred.item() * 100:>9.2f}% {future_price.item():>15.2f}")
# %%
