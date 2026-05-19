import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the test features and ground truth targets
df_test = pd.read_csv('test.csv')
df_labels = pd.read_csv('housing.csv')  # Contains target 'SalePrice' for test set

# 2. Select the exact same features used during training
features = [
    'GrLivArea', 
    'BedroomAbvGr', 
    'TotalBsmtSF', 
    'FullBath', 
    'OverallQual', 
    'GarageCars', 
    'YearBuilt'
]

X_test = df_test[features]
y_test = df_labels['SalePrice']

# 3. Load the pre-trained pipeline asset
pipeline = joblib.load('housing_pipeline.pkl')

# 4. Generate predictions
predictions = pipeline.predict(X_test)

# 5. Compute evaluation metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=== Project 2: Housing Regression Evaluation ===")
print(f"Model R² Variance Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")

# 6. Show sample comparisons (Actual vs Predicted)
eval_df = pd.DataFrame({
    'Id': df_test['Id'],
    'Actual Price': y_test,
    'Predicted Price': predictions,
    'Absolute Error': abs(y_test - predictions)
}).round(2)

print("\nFirst 10 Rows of Validation Performance Matrix:")
print(eval_df.head(10).to_string(index=False))

# 7. Export a formal text summary report
with open('evaluation_report.txt', 'w') as f:
    f.write("=== HOUSING REGRESSION MODEL PERFORMANCE ===\n")
    f.write(f"R-squared ($R^2$) Score: {r2:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): ${mae:,.2f}\n")
print("\nPerformance summary report successfully exported to 'evaluation_report.txt'.")