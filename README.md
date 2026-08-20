# 🔋 Battery Health Predictor

A Machine Learning based runtime application that predicts the **State of Health (SOH)** of a battery using battery operating and performance parameters.

The project uses a **Random Forest Regression** model trained on battery health data and provides the predicted battery health percentage and battery condition through a command-line interface.

---

## 📌 Project Overview

Battery health gradually decreases as a battery undergoes repeated charge and discharge cycles.

This project uses Machine Learning to estimate the **State of Health (SOH)** of a battery from parameters such as:

- Cycle
- Voltage
- Current
- Temperature
- Charge Time
- Discharge Time
- Internal Resistance
- Capacity
- Ambient Humidity
- C Rate

The system predicts the battery's SOH as a percentage and classifies the battery condition as:

- **Good** — SOH ≥ 80%
- **Moderate** — SOH ≥ 60%
- **Poor** — SOH < 60%

---

## 🎯 Objective

The main objective of this project is to develop a Machine Learning model capable of predicting battery health based on battery operating conditions and degradation-related parameters.

---

## 🧠 Machine Learning Model

The project uses:

**Random Forest Regressor**

Random Forest Regression was selected because it can model nonlinear relationships between battery parameters and battery health.

### Target Variable

```text
SOH