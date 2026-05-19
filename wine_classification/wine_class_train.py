import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load the comprehensive wine quality dataset
df = pd.read_csv('wine_quality.csv')

# 2. Extract features and target column
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
    'pH', 'sulphates', 'alcohol', 'quality'
]
X = df[features]
y = df['type']

# Encode text targets ('red'/'white') into binary values (0/1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 3. Stratified Train/Test Split (ensures proportional class representation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Define an integrated data transformation and classification pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Clean missing numeric values
    ('scaler', StandardScaler()),                  # Normalize feature variances
    ('classifier', LogisticRegression(max_iter=1000)) # Train the linear classifier
])

# 5. Fit the model pipeline
print("Training Logistic Regression pipeline on Wine Quality dataset...")
pipeline.fit(X_train, y_train)

# 6. Export pipeline components and test sets for validation
joblib.dump(pipeline, 'wine_pipeline.pkl')
joblib.dump(encoder, 'label_encoder.pkl')

X_test.to_csv('wine_test_features.csv', index=False)
pd.Series(y_test).to_csv('wine_test_labels.csv', index=False, header=['type'])

# 7. Print initial evaluation
predictions = pipeline.predict(X_test)
print("\nTraining Complete! Classification Report:")
print(classification_report(y_test, predictions, target_names=encoder.classes_))
