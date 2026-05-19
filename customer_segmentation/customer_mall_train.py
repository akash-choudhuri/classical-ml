import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the custom mall customers dataset
df = pd.read_csv('mall_customers.csv')

# 2. Select features for clustering matching the exact dataset schema
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# 3. Feature Scaling (critical for distance calculations in K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train K-Means (5 clusters is the analytical benchmark for this specific dataset)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 5. Export trained artifacts to files
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Clustering pipeline assets successfully created:")
print("- 'kmeans_model.pkl' (Trained Clustering Model)")
print("- 'scaler.pkl' (Fitted Feature Scaler)")