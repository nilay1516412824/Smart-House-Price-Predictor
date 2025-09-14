# smart_house_price_predictor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load Dataset
data = pd.read_csv(r'C:\nilay code\Projects\shp\large_house_prices.csv')
print("Dataset preview:")
print(data.head())

# 2. Features & Target
X = data.drop("Price", axis=1)
y = data["Price"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_score = -1

# 5. Train & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {"MSE": mse, "R2 Score": r2}
    
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

# 6. Compare Performance
print("\nModel Performance:")
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.2f}, R2={metrics['R2 Score']:.2f}")

print(f"\n Best Model: {best_name} (R²={best_score:.2f})")

# 7. Save Best Model
joblib.dump(best_model, "best_house_price_model.pkl")

# 8. Example Prediction
sample_house = np.array([[2000, 3, 8, 10]])  # Size=2000, Rooms=3, Location=8, Age=10
predicted_price = best_model.predict(sample_house)[0]
print(f"\nPredicted Price for sample house: ${predicted_price:,.2f}")


