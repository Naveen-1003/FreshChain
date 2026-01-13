import numpy as np

# --- 1. Hardcoded Kinetic Parameters (Ea and A) ---
# Ea is in J/mol, A is in 1/day (assuming first-order kinetics)
# NOTE: Capsicum is assumed to use the same kinetics as Bellpepper,
# and Bittergourd/Okra use general estimates due to limited public data.
KINETIC_DATA = {
    "apple": {"Ea": 70000.0, "A": 2.0e11, "metric": "firmness loss"},
    "banana": {"Ea": 62000.0, "A": 9.0e9, "metric": "softening/ripening"},
    "bellpepper": {"Ea": 55000.0, "A": 1.0e9, "metric": "color degradation"},
    "bittergourd": {"Ea": 50000.0, "A": 5.0e8, "metric": "estimated weight loss"},
    "capsicum": {"Ea": 55000.0, "A": 1.0e9, "metric": "color degradation"},
    "carrot": {"Ea": 85000.0, "A": 5.0e13, "metric": "vitamin c loss"},
    "cucumber": {"Ea": 48000.0, "A": 3.5e7, "metric": "firmness loss"},
    "mango": {"Ea": 46000.0, "A": 2.5e7, "metric": "firmness/ripening"},
    "okra": {"Ea": 58000.0, "A": 7.0e9, "metric": "estimated respiration rate"},
    "orange": {"Ea": 44000.0, "A": 1.0e7, "metric": "vitamin c loss"},
    "potato": {"Ea": 60000.0, "A": 4.0e10, "metric": "texture/ascorbic acid loss"},
    "strawberry": {"Ea": 32000.0, "A": 5.0e4, "metric": "anthocyanin (color) loss"},
    "tomato": {"Ea": 36000.0, "A": 1.5e5, "metric": "softening/respiration"},
}

# Universal Gas Constant (R) in J/(mol·K)
R = 8.314

# Reference Temperature (T_ref) for comparison (5°C is a typical refrigerated temp)
T_REF_C = 5.0
T_REF_K = T_REF_C + 273.15


def arrhenius_rate_constant(Ea, A, T_k):
    """
    Calculates the reaction rate constant (k) using the Arrhenius equation:
    k = A * exp(-Ea / (R * T))
    """
    # The rate constant (k) increases as temperature (T_k) increases
    k = A * np.exp(-Ea / (R * T_k))
    return k


def predict_shelf_life():
    """Predicts relative shelf life based on user input and Arrhenius model."""
    print("\n--- 🌡️ Arrhenius Shelf Life Predictor ---")

    # --- 2. User Input ---
    available_items = ", ".join(KINETIC_DATA.keys()).replace('fresh', '')

    # Get item name
    while True:
        # Prompt user and remove "fresh" prefix if they enter it
        fruit_name = input(f"Enter the produce item ({available_items}): ").strip().lower().replace('fresh', '')
        if fruit_name in KINETIC_DATA:
            data = KINETIC_DATA[fruit_name]
            break
        else:
            print("Item not found. Please try again or choose from the list.")

    # Get temperature
    while True:
        try:
            temp_c = float(input("Enter the storage temperature in Celsius (°C): "))
            break
        except ValueError:
            print("Invalid temperature. Please enter a number.")

    # --- 3. Calculation ---

    # Convert input temperature to Kelvin
    T_input_K = temp_c + 273.15

    Ea = data["Ea"]
    A = data["A"]
    metric = data["metric"]

    # Calculate rate constant (k) at the input temperature
    k_input = arrhenius_rate_constant(Ea, A, T_input_K)

    # Calculate rate constant (k) at the reference temperature (5°C)
    k_ref = arrhenius_rate_constant(Ea, A, T_REF_K)

    # The Shelf Life Ratio (SLR) is inversely proportional to the rate constant ratio:
    # Shelf Life is proportional to 1/k.
    # SLR = (Shelf Life at T_input) / (Shelf Life at T_ref) = k_ref / k_input
    if k_input == 0:
        shelf_life_ratio = float('inf')
    else:
        shelf_life_ratio = k_ref / k_input

    # --- 4. Output Results ---
    print("\n" + "=" * 50)
    print(f"📦 Product: {fruit_name.capitalize()}")
    print(f"🌡️ Storage Temperature: {temp_c:.1f}°C ({T_input_K:.2f} K)")
    print(f"🔬 Spoilage Metric: {metric.capitalize()}")
    print(f"🧪 Activation Energy (Ea): {Ea / 1000:.1f} kJ/mol")
    print("-" * 50)

    # Rate Constant at input temperature
    print(f"Degradation Rate (k) at {temp_c}°C: {k_input:.5e} 1/day")

    # Shelf Life Ratio
    print(f"Shelf Life Ratio (vs. {T_REF_C}°C): {shelf_life_ratio:.2f}")

    if shelf_life_ratio > 1:
        print(f"Conclusion: Shelf life is {shelf_life_ratio:.2f} times **LONGER** than at {T_REF_C}°C.")
    elif shelf_life_ratio < 1:
        speed_up_factor = 1 / shelf_life_ratio
        print(f"Conclusion: Spoilage is {speed_up_factor:.2f} times **FASTER** than at {T_REF_C}°C.")
    else:
        print(f"Conclusion: Shelf life is approximately the same as at {T_REF_C}°C.")

    print("=" * 50)


if __name__ == "__main__":
    predict_shelf_life()