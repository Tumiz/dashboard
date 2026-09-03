# %% Data Loading
import pandas

d = pandas.read_csv("dashboard_monthly.csv", index_col="month")

train_set = d[["aux_usd"]]
for i in range(1, 7):
    train_set[f"btc_usd_{i}m"] = d["btc_usd"].shift(i)
    train_set[f"twexb_{i}m"] = d["twexb"].shift(i)
    train_set[f"real_yield_pct_{i}m"] = d["real_yield_pct"].shift(i)
    train_set[f"fed_rate_pct_{i}m"] = d["fed_rate_pct"].shift(i)
    train_set[f"gpr_{i}m"] = d["gpr"].shift(i)
train_set = train_set.dropna()
print(train_set.index[0], "~", train_set.index[-1], ",", len(train_set))


# %% Model Definition
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.Sequential([
	keras.Input(shape=(train_set.shape[1] - 1,)),
	keras.layers.Dense(4, activation="relu"),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
early_stopping = keras.callbacks.EarlyStopping(min_delta=0.0001,
                                               monitor="val_loss", 
                                               patience=20, 
                                               restore_best_weights=True)
model.fit(train_set.drop(columns=["aux_usd"]), train_set["aux_usd"], epochs=500, batch_size=4, validation_split=0.2, callbacks=[early_stopping])
model.save("predictor_model.h5")
plt.plot(model.history.history["loss"], label="Training Loss")
plt.plot(model.history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
# %% Validation
test = d[["aux_usd", "btc_usd"]].iloc[-len(train_set):].copy()
test["preds"] = model.predict(train_set.drop(columns=["aux_usd"]))
test.plot(y=["aux_usd", "preds", "btc_usd"], 
          secondary_y="btc_usd", 
          grid=True,
          title="Aux USD Prediction vs Actual")

# %%
