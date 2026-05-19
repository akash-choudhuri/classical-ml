import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Load and preprocessing
df = pd.read_csv('wine.csv')
X = df[['alcohol', 'malic_acid', 'ash', 'magnesium', 'total_phenols']]
y = df['wine_type']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 3. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Save model, encoder, and test partition for test.py
joblib.dump(model, 'wine_model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')
X_test.to_csv('unseen_test_features.csv', index=False)
pd.Series(y_test).to_csv('unseen_test_labels.csv', index=False, header=['wine_type'])

# 5. Write an evaluation report artifact
predictions = model.predict(X_test)
report = classification_report(y_test, predictions, target_names=encoder.classes_, zero_division=0)

with open('evaluation_report.txt', 'w') as f:
    f.write("=== WINE CLASSIFICATION MODEL REPORT ===\n")
    f.write(report)

print("Training complete. Model and evaluation report saved.")