import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/battery_dataset.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. Select Features and Target
# ==========================================

features = [
    "Cycle",
    "Voltage",
    "Current",
    "Temperature",
    "ChargeTime",
    "DischargeTime",
    "InternalResistance",
    "Capacity",
    "AmbientHumidity",
    "C_Rate"
]

X = df[features]
y = df["SOH"]


# ==========================================
# 3. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. Train Random Forest Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 5. Make Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. Evaluate Model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# 7. Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- Feature Importance ---")
print(importance.to_string(index=False))


# ==========================================
# 8. Actual vs Predicted Graph
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual SOH (%)")
plt.ylabel("Predicted SOH (%)")
plt.title("Actual vs Predicted Battery Health")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.close()

print("\nActual vs Predicted graph saved as actual_vs_predicted.png")


# ==========================================
# 9. Feature Importance Graph
# ==========================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Battery Health Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("Feature importance graph saved as feature_importance.png")


# ==========================================
# 10. Save Model
# ==========================================

joblib.dump(model, "battery_health_model.pkl")

print("Model saved successfully as battery_health_model.pkl")