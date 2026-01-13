import numpy as np
import pandas as pd
import joblib

# Load trained ML model file
model = joblib.load("shelf_life_model.pkl")

# --- Constants and Parameters ---
R = 8.314  # J/mol·K
T_REF_C = 5.0
T_REF_K = T_REF_C + 273.15
OPTIMAL_RH = 90.0

# Example fruit kinetic parameters
KINETIC_DATA = {
    "apple": {"Ea": 70000.0, "A": 2.0e11, "ref_life_days": 60},
    "banana": {"Ea": 62000.0, "A": 9.0e9, "ref_life_days": 14},
    "tomato": {"Ea": 36000.0, "A": 1.5e5, "ref_life_days": 14},
    "mango": {"Ea": 46000.0, "A": 2.5e7, "ref_life_days": 12},
    "potato": {"Ea": 60000.0, "A": 4.0e10, "ref_life_days": 90},
}

# --- Physics-based Equation ---
def arrhenius_rate_constant(Ea, A, T_k):
    return A * np.exp(-Ea / (R * T_k))

def humidity_factor(rh):
    rh = max(30, min(rh, 100))
    deviation = abs(rh - OPTIMAL_RH)
    return np.exp(-0.02 * (deviation ** 1.2))

def arrhenius_shelf_life(fruit, temp_c, rh):
    data = KINETIC_DATA.get(fruit.lower())
    if not data:
        raise ValueError("Unsupported fruit type.")
    Ea, A, ref = data["Ea"], data["A"], data["ref_life_days"]
    T_k = temp_c + 273.15

    k_input = arrhenius_rate_constant(Ea, A, T_k)
    k_ref = arrhenius_rate_constant(Ea, A, T_REF_K)
    ratio = k_ref / k_input
    rh_adj = humidity_factor(rh)
    return ref * ratio * rh_adj

# --- Hybrid Prediction (weighted fusion) ---
def hybrid_prediction(fruit, temp_c, rh, alpha=0.35):
    # Machine Learning input prep
    X = pd.DataFrame({"Temperature_C": [temp_c], "Humidity_%": [rh]})
    for f in model.feature_names_in_:
        if f.startswith("Type_"):
            X[f] = 1 if f == f"Type_{fruit.capitalize()}" else 0
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    # Model and Arrhenius predictions
    ml_pred = model.predict(X)[0]
    arr_pred = arrhenius_shelf_life(fruit, temp_c, rh)

    # Weighted hybrid output
    hybrid_pred = alpha * arr_pred + (1 - alpha) * ml_pred

    print("\n================ Hybrid Shelf-Life Prediction ================")
    print(f"Fruit: {fruit.capitalize()} | Temp: {temp_c}°C | Humidity: {rh}%")
    print(f"Arrhenius Shelf Life: {arr_pred:.2f} days")
    print(f"ML Model Shelf Life: {ml_pred:.2f} days")
    print(f"Weighted Hybrid (α={alpha}): {hybrid_pred:.2f} days")
    print("==============================================================")

    return hybrid_pred

# --- Run Example (customizable) ---
if __name__ == "__main__":
    fruit = "apple"
    temp = 1
    humidity = 10
    hybrid_prediction(fruit, temp, humidity)
