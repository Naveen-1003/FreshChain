import pandas as pd
import joblib

# Load trained model
model = joblib.load('shelf_life_model.pkl')
print("Model loaded successfully!\n")

# Get user input interactively
fruit_type = input("Enter produce type (Banana, Apple, Tomato, Mango, Grape, Strawberry, Cucumber, Carrot, Potato, Onion): ").strip()
temp = float(input("Enter storage temperature (°C): "))
hum = float(input("Enter storage humidity (%): "))

# Prepare one-hot encoded inputs
all_types = ['Apple', 'Banana', 'Tomato', 'Mango', 'Grape', 'Strawberry', 'Cucumber', 'Carrot', 'Potato', 'Onion']
features = {'Temperature_C': temp, 'Humidity_%': hum}

for t in all_types:
    features[f'Type_{t}'] = 1 if t.lower() == fruit_type.lower() else 0

# Convert to DataFrame
input_df = pd.DataFrame([features])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict shelf life
predicted_life = model.predict(input_df)[0]
print(f"\nPredicted shelf life for {fruit_type} at {temp}°C and {hum}% humidity: {predicted_life:.1f} days")
