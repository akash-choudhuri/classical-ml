import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load saved artifacts
model = joblib.load('wine_model.pkl')
encoder = joblib.load('label_encoder.pkl')
X_test = pd.read_csv('unseen_test_features.csv')
y_test = pd.read_csv('unseen_test_labels.csv')['wine_type']

# 2. Evaluate model
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("=== Project 1: Testing Saved Model ===")
print(f"Loaded Model Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:")
print(cm)

# 3. Predict on a single raw manual input
print("\nPredicting a new random wine signature...")
# Custom sample: high alcohol, low malic acid, high magnesium
new_wine = pd.DataFrame([[14.1, 1.5, 2.3, 115, 2.7]], columns=X_test.columns)
pred_code = model.predict(new_wine)
pred_label = encoder.inverse_transform(pred_code)
print(f"Predicted Class: {pred_label[0].upper()}")