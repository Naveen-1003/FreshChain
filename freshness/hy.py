import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# Load trained ML model file
model = joblib.load("shelf_life_model.pkl")

# --- Constants and Parameters ---
R = 8.314  # J/mol·K
T_REF_C = 5.0
T_REF_K = T_REF_C + 273.15
OPTIMAL_RH = 90.0

# --- Dynamic Alpha Configuration ---
ALPHA_CONFIG = {
    "base_alpha": 0.35,  # Default starting value
    "min_alpha": 0.1,    # Minimum physics weight
    "max_alpha": 0.8,    # Maximum physics weight
    "sensor_stability_weight": 0.3,  # How much sensor stability affects alpha
    "ml_performance_weight": 0.4,    # How much ML performance affects alpha
    "history_file": "prediction_history.json"
}

# Example fruit kinetic parameters
KINETIC_DATA = {
    "apple": {"Ea": 70000.0, "A": 2.0e11, "ref_life_days": 60},
    "banana": {"Ea": 62000.0, "A": 9.0e9, "ref_life_days": 14},
    "tomato": {"Ea": 36000.0, "A": 1.5e5, "ref_life_days": 14},
    "mango": {"Ea": 46000.0, "A": 2.5e7, "ref_life_days": 12},
    "potato": {"Ea": 60000.0, "A": 4.0e10, "ref_life_days": 90},
}

# --- Sensor Stability Assessment ---
def assess_sensor_stability(temp_c, rh, history_data):
    """
    Assess sensor stability based on recent readings variance
    Returns stability score (0-1, higher = more stable)
    """
    if not history_data or len(history_data) < 3:
        return 0.5  # Neutral if insufficient data
    
    recent_temps = [entry.get('temperature', temp_c) for entry in history_data[-10:]]
    recent_rh = [entry.get('humidity', rh) for entry in history_data[-10:]]
    
    # Calculate coefficient of variation (CV) as stability metric
    temp_cv = np.std(recent_temps) / (np.mean(recent_temps) + 1e-6) if recent_temps else 0
    rh_cv = np.std(recent_rh) / (np.mean(recent_rh) + 1e-6) if recent_rh else 0
    
    # Lower CV means higher stability
    temp_stability = max(0, 1 - temp_cv * 10)  # Scale factor for temperature
    rh_stability = max(0, 1 - rh_cv * 5)       # Scale factor for humidity
    
    return (temp_stability + rh_stability) / 2

# --- ML Performance Assessment ---
def assess_ml_performance(history_data):
    """
    Assess ML model performance based on recent prediction accuracy
    Returns performance score (0-1, higher = better performance)
    """
    if not history_data or len(history_data) < 3:
        return 0.5  # Neutral if insufficient data
    
    # Look for entries with actual outcomes (if available)
    performance_scores = []
    for entry in history_data[-20:]:  # Last 20 predictions
        if 'actual_shelf_life' in entry and 'ml_prediction' in entry:
            actual = entry['actual_shelf_life']
            predicted = entry['ml_prediction']
            # Calculate relative accuracy (1 - relative error)
            if actual > 0:
                relative_error = abs(actual - predicted) / actual
                accuracy = max(0, 1 - relative_error)
                performance_scores.append(accuracy)
    
    if performance_scores:
        return np.mean(performance_scores)
    else:
        # If no actual outcomes available, use prediction consistency as proxy
        ml_preds = [entry.get('ml_prediction', 0) for entry in history_data[-10:]]
        if len(ml_preds) > 1:
            consistency = 1 - (np.std(ml_preds) / (np.mean(ml_preds) + 1e-6))
            return max(0, min(1, consistency))
    
    return 0.5  # Default neutral score

# --- Dynamic Alpha Calculation ---
def calculate_dynamic_alpha(fruit, temp_c, rh, history_data=None):
    """
    Calculate dynamic alpha based on sensor stability and ML performance
    """
    config = ALPHA_CONFIG
    base_alpha = config["base_alpha"]
    
    # Load history if not provided
    if history_data is None:
        history_data = load_prediction_history()
    
    # Assess sensor stability
    sensor_stability = assess_sensor_stability(temp_c, rh, history_data)
    
    # Assess ML performance
    ml_performance = assess_ml_performance(history_data)
    
    # Calculate alpha adjustments
    # Higher sensor stability → trust physics more (increase alpha)
    sensor_adjustment = (sensor_stability - 0.5) * config["sensor_stability_weight"]
    
    # Higher ML performance → trust ML more (decrease alpha)
    ml_adjustment = -(ml_performance - 0.5) * config["ml_performance_weight"]
    
    # Combine adjustments
    dynamic_alpha = base_alpha + sensor_adjustment + ml_adjustment
    
    # Clamp to valid range
    dynamic_alpha = max(config["min_alpha"], min(config["max_alpha"], dynamic_alpha))
    
    return dynamic_alpha, sensor_stability, ml_performance

# --- History Management ---
def load_prediction_history():
    """Load prediction history from file"""
    try:
        if os.path.exists(ALPHA_CONFIG["history_file"]):
            with open(ALPHA_CONFIG["history_file"], 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load history file: {e}")
    return []

def save_prediction_to_history(fruit, temp_c, rh, arr_pred, ml_pred, hybrid_pred, alpha, sensor_stability, ml_performance):
    """Save prediction to history for future alpha calculations"""
    try:
        history = load_prediction_history()
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "fruit": fruit,
            "temperature": temp_c,
            "humidity": rh,
            "arrhenius_prediction": arr_pred,
            "ml_prediction": ml_pred,
            "hybrid_prediction": hybrid_pred,
            "alpha_used": alpha,
            "sensor_stability": sensor_stability,
            "ml_performance": ml_performance
        }
        
        history.append(entry)
        
        # Keep only last 100 entries to prevent file from growing too large
        if len(history) > 100:
            history = history[-100:]
        
        with open(ALPHA_CONFIG["history_file"], 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save to history file: {e}")

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

# --- Hybrid Prediction (dynamic weighted fusion) ---
def hybrid_prediction(fruit, temp_c, rh, fixed_alpha=None):
    """
    Enhanced hybrid prediction with dynamic alpha adjustment
    
    Args:
        fruit: Type of fruit
        temp_c: Temperature in Celsius
        rh: Relative humidity percentage
        fixed_alpha: If provided, uses this alpha instead of dynamic calculation
    """
    # Load prediction history
    history_data = load_prediction_history()
    
    # Calculate dynamic alpha or use fixed value
    if fixed_alpha is not None:
        alpha = fixed_alpha
        sensor_stability = 0.5  # Default values when using fixed alpha
        ml_performance = 0.5
    else:
        alpha, sensor_stability, ml_performance = calculate_dynamic_alpha(fruit, temp_c, rh, history_data)
    
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

    # Display results
    print("\n================ Dynamic Hybrid Shelf-Life Prediction ================")
    print(f"Fruit: {fruit.capitalize()} | Temp: {temp_c}°C | Humidity: {rh}%")
    print(f"Sensor Stability Score: {sensor_stability:.3f}")
    print(f"ML Performance Score: {ml_performance:.3f}")
    print(f"Dynamic Alpha (α): {alpha:.3f}")
    print(f"Arrhenius Shelf Life: {arr_pred:.2f} days")
    print(f"ML Model Shelf Life: {ml_pred:.2f} days")
    print(f"Dynamic Hybrid Prediction: {hybrid_pred:.2f} days")
    print("=====================================================================")
    
    # Save to history for future alpha calculations
    save_prediction_to_history(fruit, temp_c, rh, arr_pred, ml_pred, hybrid_pred, alpha, sensor_stability, ml_performance)

    return hybrid_pred

# --- Utility Functions ---
def update_actual_outcome(prediction_index, actual_shelf_life):
    """
    Update a past prediction with actual outcome for ML performance assessment
    
    Args:
        prediction_index: Index of prediction to update (0 = most recent, -1 = oldest)
        actual_shelf_life: Actual shelf life observed in days
    """
    try:
        history = load_prediction_history()
        if abs(prediction_index) < len(history):
            history[prediction_index]["actual_shelf_life"] = actual_shelf_life
            with open(ALPHA_CONFIG["history_file"], 'w') as f:
                json.dump(history, f, indent=2)
            print(f"Updated prediction with actual shelf life: {actual_shelf_life} days")
        else:
            print("Invalid prediction index")
    except Exception as e:
        print(f"Error updating actual outcome: {e}")

def get_prediction_history_summary():
    """Get summary of recent predictions and performance"""
    history = load_prediction_history()
    if not history:
        print("No prediction history available")
        return
    
    print("\n================ Prediction History Summary ================")
    print(f"Total predictions: {len(history)}")
    
    recent = history[-10:]  # Last 10 predictions
    if recent:
        avg_alpha = np.mean([p.get('alpha_used', 0.35) for p in recent])
        avg_sensor_stability = np.mean([p.get('sensor_stability', 0.5) for p in recent])
        avg_ml_performance = np.mean([p.get('ml_performance', 0.5) for p in recent])
        
        print(f"Recent Average Alpha: {avg_alpha:.3f}")
        print(f"Recent Average Sensor Stability: {avg_sensor_stability:.3f}")
        print(f"Recent Average ML Performance: {avg_ml_performance:.3f}")
    
    # Show predictions with actual outcomes
    validated = [p for p in history if 'actual_shelf_life' in p]
    if validated:
        print(f"Predictions with actual outcomes: {len(validated)}")
        errors = []
        for p in validated:
            error = abs(p['hybrid_prediction'] - p['actual_shelf_life']) / p['actual_shelf_life']
            errors.append(error)
        avg_error = np.mean(errors)
        print(f"Average relative error: {avg_error:.3f} ({avg_error*100:.1f}%)")
    
    print("============================================================")

# --- Run Example (customizable) ---
if __name__ == "__main__":
    # Example 1: Basic dynamic prediction
    fruit = "apple"
    temp = 40
    humidity = 10
    
    print("=== Example 1: Dynamic Alpha Prediction ===")
    hybrid_prediction(fruit, temp, humidity)
    
    # Example 2: Compare with fixed alpha
    print("\n=== Example 2: Comparison with Fixed Alpha ===")
    print("Fixed Alpha (0.35):")
    hybrid_prediction(fruit, temp, humidity, fixed_alpha=0.35)
    
    # Example 3: Show prediction history summary
    print("\n=== Example 3: History Summary ===")
    get_prediction_history_summary()
    
    # Example 4: Demonstrate how to update with actual outcome
    print("\n=== Example 4: Update with Actual Outcome ===")
    print("To update the most recent prediction with actual shelf life:")
    print("update_actual_outcome(0, actual_days_observed)")
    print("Example: update_actual_outcome(0, 15.5)  # If apple lasted 15.5 days")
