import joblib
import pandas as pd


# Load the trained model
model = joblib.load("battery_health_model.pkl")

print("=" * 55)
print("           BATTERY HEALTH PREDICTOR")
print("=" * 55)

print("\nEnter the battery details:\n")

cycle = float(input("Cycle: "))
voltage = float(input("Voltage: "))
current = float(input("Current: "))
temperature = float(input("Temperature: "))
charge_time = float(input("Charge Time: "))
discharge_time = float(input("Discharge Time: "))
internal_resistance = float(input("Internal Resistance: "))
capacity = float(input("Capacity: "))
ambient_humidity = float(input("Ambient Humidity: "))
c_rate = float(input("C Rate: "))

# Create input DataFrame
input_data = pd.DataFrame([{
    "Cycle": cycle,
    "Voltage": voltage,
    "Current": current,
    "Temperature": temperature,
    "ChargeTime": charge_time,
    "DischargeTime": discharge_time,
    "InternalResistance": internal_resistance,
    "Capacity": capacity,
    "AmbientHumidity": ambient_humidity,
    "C_Rate": c_rate
}])

# Predict SOH
prediction = model.predict(input_data)[0]

# Keep SOH within 0-100%
prediction = max(0, min(100, prediction))

print("\n" + "=" * 55)
print(f"Predicted Battery Health (SOH): {prediction:.2f}%")
print("=" * 55)

# Determine battery condition
if prediction >= 80:
    condition = "Good"
elif prediction >= 60:
    condition = "Moderate"
else:
    condition = "Poor"

print(f"Battery Condition: {condition}")
print("=" * 55)