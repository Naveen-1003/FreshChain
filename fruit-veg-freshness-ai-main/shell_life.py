import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load your already existing dataset
df = pd.read_csv('fruit_veg_shelf_life.csv')

# Encode the fruit/vegetable type for modeling
df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True)

# Split into features (X) and target (y)
X = df_encoded.drop('Shelf_Life_Days', axis=1)
y = df_encoded['Shelf_Life_Days']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'R² Score: {r2:.3f}')

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Features:\n", importances.head(10))

# Save trained model
joblib.dump(model, 'shelf_life_model.pkl')
print("\nModel saved as shelf_life_model.pkl")
