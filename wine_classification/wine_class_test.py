import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load pre-trained model pipeline and label encoder
pipeline = joblib.load('wine_pipeline.pkl')
encoder = joblib.load('label_encoder.pkl')

# 2. Load the evaluation test sets
X_test = pd.read_csv('wine_test_features.csv')
y_test = pd.read_csv('wine_test_labels.csv')['type']

# 3. Predict classifications
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("=== Project 1: Testing Saved Wine Pipeline ===")
print(f"Loaded Model Test Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

# 4. Simulating live production classification
print("\nPredicting classifications for a new batch of incoming wines...")
new_wines = pd.DataFrame([
    [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 5],   # Expected Red profile
    [6.6, 0.16, 0.40, 1.5, 0.044, 48.0, 143.0, 0.9912, 3.54, 0.52, 12.4, 7]  # Expected White profile
], columns=X_test.columns)

pred_codes = pipeline.predict(new_wines)
pred_labels = encoder.inverse_transform(pred_codes)

for i, wine_type in enumerate(pred_labels):
    print(f"Wine Sample {i+1} Characteristics -> Predicted Class: {wine_type.upper()}")
