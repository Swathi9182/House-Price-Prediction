import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================================
# 2. CHECK DATA
# ==========================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==========================================================
# 3. FEATURES AND TARGET
# ==========================================================

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# ==========================================================
# 4. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================================
# 5. LINEAR REGRESSION
# ==========================================================

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_pred)


# ==========================================================
# 6. DECISION TREE
# ==========================================================

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

tree_mae = mean_absolute_error(y_test, tree_pred)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_rmse = np.sqrt(tree_mse)
tree_r2 = r2_score(y_test, tree_pred)


# ==========================================================
# 7. RANDOM FOREST
# ==========================================================

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)

forest_mae = mean_absolute_error(y_test, forest_pred)
forest_mse = mean_squared_error(y_test, forest_pred)
forest_rmse = np.sqrt(forest_mse)
forest_r2 = r2_score(y_test, forest_pred)


# ==========================================================
# 8. FEATURE SCALING FOR SVR
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================================
# 9. SUPPORT VECTOR REGRESSION
# ==========================================================

svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)

svr_pred = svr_model.predict(X_test_scaled)

svr_mae = mean_absolute_error(y_test, svr_pred)
svr_mse = mean_squared_error(y_test, svr_pred)
svr_rmse = np.sqrt(svr_mse)
svr_r2 = r2_score(y_test, svr_pred)


# ==========================================================
# 10. MODEL COMPARISON
# ==========================================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "SVR"
    ],
    "MAE": [
        linear_mae,
        tree_mae,
        forest_mae,
        svr_mae
    ],
    "RMSE": [
        linear_rmse,
        tree_rmse,
        forest_rmse,
        svr_rmse
    ],
    "R2": [
        linear_r2,
        tree_r2,
        forest_r2,
        svr_r2
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))


# ==========================================================
# 11. SAVE RANDOM FOREST MODEL
# ==========================================================

joblib.dump(forest_model, "house_price_model.pkl")

print("\nRandom Forest model saved as house_price_model.pkl")


# ==========================================================
# 12. ACTUAL VS PREDICTED VISUALIZATION
# ==========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    forest_pred,
    alpha=0.5
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")

plt.show()

# ==========================================================
# 13. FEATURE IMPORTANCE
# ==========================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)


# Plot feature importance
plt.figure(figsize=(8, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("feature_importance.png")

plt.show()
# ==========================================================
# 14. COMPLETION MESSAGE
# ==========================================================

print("\n======================================")
print("HOUSE PRICE PREDICTION PROJECT READY!")
print("======================================")