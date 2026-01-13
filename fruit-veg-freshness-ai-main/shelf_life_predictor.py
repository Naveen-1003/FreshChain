import numpy as np
from typing import Dict, Any, Optional

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


def arrhenius_rate_constant(Ea: float, A: float, T_k: float) -> float:
    """
    Calculates the reaction rate constant (k) using the Arrhenius equation:
    k = A * exp(-Ea / (R * T))
    
    Args:
        Ea: Activation energy in J/mol
        A: Pre-exponential factor in 1/day
        T_k: Temperature in Kelvin
    
    Returns:
        Rate constant k in 1/day
    """
    # The rate constant (k) increases as temperature (T_k) increases
    k = A * np.exp(-Ea / (R * T_k))
    return k


def predict_shelf_life_api(
    fruit_name: str,
    temp_c: float,
    baseline_shelf_life_days_at_ref: Optional[float] = None,
    current_age_days: float = 0.0,
    uncertainty_fraction: float = 0.2,
) -> Dict[str, Any]:
    """
    API version of shelf life prediction that returns structured data instead of printing.
    
    Args:
        fruit_name: Name of the fruit/vegetable (must be in KINETIC_DATA)
        temp_c: Storage temperature in Celsius
        baseline_shelf_life_days_at_ref: Optional baseline shelf life at the reference temperature (5°C), in days
        current_age_days: Optional current age of the item in days (default 0)
        uncertainty_fraction: Optional fractional uncertainty to express a range (default 0.2 => ±20%)
    
    Returns:
        Dictionary containing prediction results
    """
    # Get kinetic data for the fruit
    data = KINETIC_DATA[fruit_name]
    
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
    
    # Simple characteristic life in days using first-order assumption: tau = 1/k
    life_days = float('inf') if k_input == 0 else (1.0 / k_input)

    # Clamp uncertainty to a sensible range [0, 0.9]
    try:
        uncertainty_fraction = float(uncertainty_fraction)
    except Exception:
        uncertainty_fraction = 0.2
    uncertainty_fraction = max(0.0, min(uncertainty_fraction, 0.9))

    # Compute absolute shelf life (days) and remaining days if baseline is provided
    estimated_shelf_life_days: Optional[float] = None
    remaining_days: Optional[float] = None
    estimated_shelf_life_days_range: Optional[Dict[str, float]] = None
    remaining_days_range: Optional[Dict[str, float]] = None

    if baseline_shelf_life_days_at_ref is not None and np.isfinite(shelf_life_ratio):
        try:
            baseline = float(baseline_shelf_life_days_at_ref)
            age = float(current_age_days)
        except Exception:
            baseline = None
            age = 0.0

        if baseline is not None and baseline >= 0:
            estimated_shelf_life_days = baseline * float(shelf_life_ratio)
            remaining_days = max(0.0, estimated_shelf_life_days - max(0.0, age))

            # Ranges using uncertainty fraction
            lower_factor = 1.0 - uncertainty_fraction
            upper_factor = 1.0 + uncertainty_fraction

            est_lower = estimated_shelf_life_days * lower_factor
            est_upper = estimated_shelf_life_days * upper_factor
            estimated_shelf_life_days_range = {"lower": est_lower, "upper": est_upper}

            rem_lower = max(0.0, est_lower - max(0.0, age))
            rem_upper = max(0.0, est_upper - max(0.0, age))
            remaining_days_range = {"lower": rem_lower, "upper": rem_upper}

    # Generate conclusion message
    if shelf_life_ratio > 1:
        conclusion = f"Shelf life is {shelf_life_ratio:.2f} times LONGER than at {T_REF_C}°C."
    elif shelf_life_ratio < 1:
        speed_up_factor = 1 / shelf_life_ratio
        conclusion = f"Spoilage is {speed_up_factor:.2f} times FASTER than at {T_REF_C}°C."
    else:
        conclusion = f"Shelf life is approximately the same as at {T_REF_C}°C."
    
    return {
        "product": fruit_name.capitalize(),
        "storage_temperature": temp_c,
        "spoilage_metric": metric.capitalize(),
        "activation_energy_kj_mol": Ea / 1000,  # Convert to kJ/mol
        "degradation_rate": k_input,
        "shelf_life_ratio": shelf_life_ratio,
        "conclusion": conclusion,
        "comparison_temperature": T_REF_C,
        # Simple estimated life at given temperature (days)
        "life_days": life_days,
        # Optional absolute estimates (provided only if baseline was provided)
        "estimated_shelf_life_days": estimated_shelf_life_days,
        "estimated_shelf_life_days_range": estimated_shelf_life_days_range,
        "remaining_days": remaining_days,
        "remaining_days_range": remaining_days_range,
    }


# Original console function for backward compatibility
def predict_shelf_life():
    """Original console-based shelf life predictor from shell.py"""
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
    result = predict_shelf_life_api(fruit_name, temp_c)

    # --- 4. Output Results ---
    print("\n" + "=" * 50)
    print(f"📦 Product: {result['product']}")
    print(f"🌡️ Storage Temperature: {result['storage_temperature']:.1f}°C")
    print(f"🔬 Spoilage Metric: {result['spoilage_metric']}")
    print(f"🧪 Activation Energy (Ea): {result['activation_energy_kj_mol']:.1f} kJ/mol")
    print("-" * 50)

    # Rate Constant at input temperature
    print(f"Degradation Rate (k) at {temp_c}°C: {result['degradation_rate']:.5e} 1/day")

    # Estimated Life in Days (characteristic time 1/k)
    if np.isfinite(result.get('life_days', float('nan'))):
        print(f"Estimated Life at {temp_c}°C: {result['life_days']:.2f} days")
    else:
        print(f"Estimated Life at {temp_c}°C: N/A (rate too low)")

    # Shelf Life Ratio
    print(f"Shelf Life Ratio (vs. {T_REF_C}°C): {result['shelf_life_ratio']:.2f}")
    print(f"Conclusion: {result['conclusion']}")
    print("=" * 50)


if __name__ == "__main__":
    predict_shelf_life()