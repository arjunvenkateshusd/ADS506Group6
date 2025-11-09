import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# ========= Data prep =========
# replace with path.
df = pd.read_csv("/Users/user/dev/processed_ems_load_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
series = (
    df.groupby("timestamp", as_index=True)["CAISO"]
      .mean()
      .sort_index()
)

series = series.asfreq("h")
series = series.interpolate()

# ========= Train-test split (last 7 days = 168 hours) =========
train = series.iloc[:-168]
test = series.iloc[-168:]

# ========= Manual TSLM: trend + hour + day-of-week =========
train_df = pd.DataFrame({
    "CAISO": train.values,
    "hour": train.index.hour,
    "day_of_week": train.index.dayofweek,
    "time_index": np.arange(len(train))
}, index=train.index)

test_df = pd.DataFrame({
    "CAISO": test.values,
    "hour": test.index.hour,
    "day_of_week": test.index.dayofweek,
    "time_index": np.arange(len(train), len(train) + len(test))
}, index=test.index)

X_train = pd.get_dummies(
    train_df[["time_index", "hour", "day_of_week"]],
    drop_first=True
)
X_train = sm.add_constant(X_train)
y_train = train_df["CAISO"]

X_test = pd.get_dummies(
    test_df[["time_index", "hour", "day_of_week"]],
    drop_first=True
)
X_test = X_test.reindex(columns=X_train.columns.drop("const"), fill_value=0)
X_test = sm.add_constant(X_test)
y_test = test_df["CAISO"]

manual_model = sm.OLS(y_train, X_train).fit()
manual_preds = manual_model.predict(X_test)

# ========= Auto TSLM: trend + Fourier seasonality =========
def fourier_matrix(t, K, period):
    """
    Build Fourier terms up to harmonic K for time index t
    with given period (e.g., 24 hours).
    """
    t = np.asarray(t)
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=t)

t_train = np.arange(len(train))
t_test = np.arange(len(train), len(train) + len(test))

best_aic = np.inf
best_K = None
best_model = None

# Try Fourier orders 1–4 and choose the one with lowest AIC
for K in range(1, 5):
    F_train = fourier_matrix(t_train, K, period=24)  
    Xk = pd.concat([
        pd.Series(1, index=t_train, name="const"),
        pd.Series(t_train, index=t_train, name="trend"),
        F_train
    ], axis=1)
    model_k = sm.OLS(y_train.values, Xk).fit()
    if model_k.aic < best_aic:
        best_aic = model_k.aic
        best_K = K
        best_model = model_k

print(f"Selected Fourier order K = {best_K} based on AIC = {best_aic:.1f}")

# Fit final auto model with best_K and forecast
F_train_best = fourier_matrix(t_train, best_K, period=24)
X_train_auto = pd.concat([
    pd.Series(1, index=t_train, name="const"),
    pd.Series(t_train, index=t_train, name="trend"),
    F_train_best
], axis=1)

auto_model = sm.OLS(y_train.values, X_train_auto).fit()

F_test_best = fourier_matrix(t_test, best_K, period=24)
X_test_auto = pd.concat([
    pd.Series(1, index=t_test, name="const"),
    pd.Series(t_test, index=t_test, name="trend"),
    F_test_best
], axis=1)

auto_preds = auto_model.predict(X_test_auto)

# ========= Evaluation =========
manual_rmse = float(np.sqrt(mean_squared_error(y_test, manual_preds)))
auto_rmse = float(np.sqrt(mean_squared_error(y_test, auto_preds)))
print(f"Manual TSLM RMSE: {manual_rmse:.2f}")
print(f"Auto Fourier TSLM RMSE: {auto_rmse:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(test.index, y_test, label="Actual", linewidth=2, color="goldenrod")
plt.plot(test.index, manual_preds, label=f"Manual TSLM (RMSE={manual_rmse:.1f})")
plt.plot(test.index, auto_preds, label=f"Auto Fourier TSLM (RMSE={auto_rmse:.1f})")
plt.title("CAISO Load Forecast: Manual vs Auto Linear Regression")
plt.xlabel("Date")
plt.ylabel("Load (MW)")
plt.legend()
plt.tight_layout()
plt.show()
