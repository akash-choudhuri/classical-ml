import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# 1. Load the Ames housing training dataset
df_train = pd.read_csv('train.csv')

# 2. Select high-impact numerical features
features = [
    'GrLivArea',     # Above grade ground living area (SqFt)
    'BedroomAbvGr',  # Bedrooms above grade
    'TotalBsmtSF',   # Total basement square footage
    'FullBath',      # Full bathrooms
    'OverallQual',   # Overall material & finish quality (1-10)
    'GarageCars',    # Garage capacity in car count
    'YearBuilt'      # Original construction year
]

X_train = df_train[features]
y_train = df_train['SalePrice']

# 3. Create a production pipeline with imputation and regression
# This learns training medians and packages them together with the model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', LinearRegression())
])

# 4. Train the model
print("Training Linear Regression model on Ames Housing dataset...")
pipeline.fit(X_train, y_train)

# 5. Save the complete pipeline asset
joblib.dump(pipeline, 'housing_pipeline.pkl')
print("Model pipeline successfully saved to 'housing_pipeline.pkl'.")