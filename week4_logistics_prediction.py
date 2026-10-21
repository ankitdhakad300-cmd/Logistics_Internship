import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


os.makedirs("week4_output", exist_ok=True)

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "Distance_km": np.random.randint(20, 1001, n),
    "Shipment_Weight_kg": np.random.randint(5, 1001, n),
    "Transport_Mode": np.random.choice(
        ["Road", "Rail", "Air"], n, p=[0.6, 0.25, 0.15]
    ),
    "Traffic_Level": np.random.choice(
        ["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25]
    ),
    "Weather": np.random.choice(
        ["Clear", "Rainy", "Stormy"], n, p=[0.6, 0.3, 0.1]
    )})

# 3. Generate delivery time using realistic patterns
mode_hours_per_km = df["Transport_Mode"].map({
    "Road": 0.065,
    "Rail": 0.045,
    "Air": 0.015})

traffic_delay = df["Traffic_Level"].map({
    "Low": 1,
    "Medium": 5,
    "High": 12})

weather_delay = df["Weather"].map({
    "Clear": 0,
    "Rainy": 4,
    "Stormy": 10})

noise = np.random.normal(0, 4, n)

df["Delivery_Time_hours"] = (
    5
    + df["Distance_km"] * mode_hours_per_km
    + df["Shipment_Weight_kg"] * 0.004
    + traffic_delay
    + weather_delay
    + noise
).clip(lower=1).round(2)

df.to_csv("week4_output/logistics_prediction_dataset.csv", index=False)

print(df.head())

print(df.shape)
print(df.isnull().sum())
print(df.describe().round(2))

X = df.drop(columns=["Delivery_Time_hours"])
y = df["Delivery_Time_hours"]

numeric_features = ["Distance_km", "Shipment_Weight_kg"]
categorical_features = ["Transport_Mode", "Traffic_Level", "Weather"]

preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())])

linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))])

parameters = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, None]}

grid_search = GridSearchCV(
    rf_pipeline,
    parameters,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
rf_predictions = best_rf.predict(X_test)

print(grid_search.best_params_)
print("Best cross-validation MAE:",round(-grid_search.best_score_, 2), "hours")

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE_hours": round(mae, 2),
        "RMSE_hours": round(rmse, 2),
        "R2_score": round(r2, 3)
    }

results = pd.DataFrame([
    evaluate_model("Linear Regression", y_test, linear_predictions),
    evaluate_model("Random Forest", y_test, rf_predictions)])

print(results.to_string(index=False))

results.to_csv("week4_output/model_comparison.csv", index=False)

if mean_absolute_error(y_test, rf_predictions) < mean_absolute_error(
    y_test, linear_predictions
):
    chosen_name = "Random Forest"
    chosen_model = best_rf
    predictions = rf_predictions
else:
    chosen_name = "Linear Regression"
    chosen_model = linear_model
    predictions = linear_predictions

print("\nSelected model based on test MAE:", chosen_name)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions, alpha=0.7)

limits = [
    min(y_test.min(), predictions.min()),
    max(y_test.max(), predictions.max())
]
plt.plot(limits, limits, linestyle="--")

plt.xlabel("Actual Delivery Time (hours)")
plt.ylabel("Predicted Delivery Time (hours)")
plt.title("Actual vs Predicted Delivery Times")
plt.tight_layout()
plt.savefig("week4_output/1_actual_vs_predicted.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.bar(results["Model"], results["MAE_hours"])
plt.ylabel("Mean Absolute Error (hours)")
plt.title("Model Comparison: MAE (Lower Is Better)")
plt.tight_layout()
plt.savefig("week4_output/2_model_mae_comparison.png")
plt.show()

errors = y_test.to_numpy() - predictions

plt.figure(figsize=(7, 5))
plt.hist(errors, bins=20, edgecolor="black")
plt.axvline(0, linestyle="--")
plt.xlabel("Actual - Predicted (hours)")
plt.ylabel("Number of Shipments")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("week4_output/3_prediction_errors.png")
plt.show()

rf_for_importance = best_rf.named_steps["model"]
feature_names = best_rf.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pd.Series(
    rf_for_importance.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print(importance.round(3))

importance.to_csv("week4_output/feature_importance.csv")

plt.figure(figsize=(9, 6))
importance.sort_values().plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("Factors Influencing Delivery Time Predictions")
plt.tight_layout()
plt.savefig("week4_output/4_feature_importance.png")
plt.show()

planning_target = 10

shipment_results = X_test.copy()
shipment_results["Actual_Delivery_Time_hours"] = y_test
shipment_results["Predicted_Delivery_Time_hours"] = np.round(
    predictions, 2
)
shipment_results["Potential_Delay"] = np.where(
    predictions > planning_target,
    "Review Schedule",
    "Within Target"
)

shipment_results = shipment_results.sort_values(
    "Predicted_Delivery_Time_hours", ascending=False
)

shipment_results.to_csv(
    "week4_output/shipment_predictions.csv", index=False
)

print(
    shipment_results[
        shipment_results["Potential_Delay"] == "Review Schedule"
    ].head(10).to_string(index=False)
)

print("\nAll results and charts saved in the week4_output folder.")
print("Week 4 predictive modeling completed successfully!")
