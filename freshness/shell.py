import numpy as np

# --- 1. Product kinetic and reference data ---
KINETIC_DATA = {
    "apple": {"Ea": 70000.0, "A": 2.0e11, "metric": "firmness loss", "ref_life_days": 60},
    "banana": {"Ea": 62000.0, "A": 9.0e9, "metric": "softening/ripening", "ref_life_days": 14},
    "bellpepper": {"Ea": 55000.0, "A": 1.0e9, "metric": "color degradation", "ref_life_days": 18},
    "bittergourd": {"Ea": 50000.0, "A": 5.0e8, "metric": "weight loss", "ref_life_days": 10},
    "capsicum": {"Ea": 55000.0, "A": 1.0e9, "metric": "color degradation", "ref_life_days": 20},
    "carrot": {"Ea": 85000.0, "A": 5.0e13, "metric": "vitamin C loss", "ref_life_days": 30},
    "cucumber": {"Ea": 48000.0, "A": 3.5e7, "metric": "firmness loss", "ref_life_days": 10},
    "mango": {"Ea": 46000.0, "A": 2.5e7, "metric": "ripening", "ref_life_days": 12},
    "okra": {"Ea": 58000.0, "A": 7.0e9, "metric": "respiration rate", "ref_life_days": 9},
    "orange": {"Ea": 44000.0, "A": 1.0e7, "metric": "vitamin C loss", "ref_life_days": 25},
    "potato": {"Ea": 60000.0, "A": 4.0e10, "metric": "texture loss", "ref_life_days": 90},
    "strawberry": {"Ea": 32000.0, "A": 5.0e4, "metric": "color loss", "ref_life_days": 7},
    "tomato": {"Ea": 36000.0, "A": 1.5e5, "metric": "softening/respiration", "ref_life_days": 14},
}

# Constants
R = 8.314  # J/mol·K
T_REF_C = 5.0
T_REF_K = T_REF_C + 273.15
OPTIMAL_RH = 90.0  # % optimal humidity for produce

def arrhenius_rate_constant(Ea, A, T_k):
    return A * np.exp(-Ea / (R * T_k))

def humidity_factor(rh):
    """Applies humidity-related penalty to shelf-life; peaks at 90% RH."""
    rh = max(30, min(rh, 100))  # clamp RH to realistic range
    deviation = abs(rh - OPTIMAL_RH)
    factor = np.exp(-0.02 * (deviation ** 1.2))
    return factor

def predict_shelf_life_days():
    print("\n--- 🍎 Arrhenius + Humidity Shelf Life Predictor (Days) ---")

    # Choose produce item
    available = ", ".join(KINETIC_DATA.keys())
    fruit = input(f"Enter the produce ({available}): ").strip().lower()
    if fruit not in KINETIC_DATA:
        print("Error: Unsupported produce type.")
        return

    # Input temperature and humidity
    temp_c = float(input("Enter storage temperature (°C): "))
    rh = float(input("Enter relative humidity (%): "))

    product = KINETIC_DATA[fruit]
    Ea, A, ref_life = product['Ea'], product['A'], product['ref_life_days']
    T_k = temp_c + 273.15

    # Compute rate constants
    k_input = arrhenius_rate_constant(Ea, A, T_k)
    k_ref = arrhenius_rate_constant(Ea, A, T_REF_K)

    # Shelf-life scaling from Arrhenius kinetics
    sl_ratio = k_ref / k_input if k_input > 0 else 1e6
    humidity_adj = humidity_factor(rh)

    # Final predicted shelf life in days
    shelf_life_days = ref_life * sl_ratio * humidity_adj

    # Output
    print("\n==================================================")
    print(f"📦 Product: {fruit.capitalize()}")
    print(f"🌡️ Temp: {temp_c:.1f}°C | 💧 Humidity: {rh:.1f}%")
    print(f"🧪 Metric: {product['metric'].capitalize()}")
    print(f"Reference Shelf Life @ {T_REF_C}°C, {OPTIMAL_RH}%: {ref_life:.1f} days")
    print(f"Arrhenius Shelf Life Ratio (Temp-only): {sl_ratio:.2f}")
    print(f"Humidity Correction Factor: {humidity_adj:.2f}")
    print("-" * 50)
    print(f"📅 Predicted Shelf Life: {shelf_life_days:.1f} days")
    print("==================================================")

if __name__ == "__main__":
    predict_shelf_life_days()
