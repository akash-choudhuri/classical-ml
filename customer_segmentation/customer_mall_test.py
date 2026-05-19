import pandas as pd
import joblib

# 1. Load pre-trained pipeline assets
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

print("=== Project 3: Segmenting New Customer Data ===")

# 2. Simulated real-time customer data profiles mapping to original schema
incoming_customers = pd.DataFrame([
    [15, 80],   # Profile Archetype: Low income, high spending score
    [55, 50],   # Profile Archetype: Moderate income, moderate spending score
    [95, 15],   # Profile Archetype: High income, low spending score
    [100, 85],  # Profile Archetype: High income, high spending score
    [20, 10]    # Profile Archetype: Low income, low spending score
], columns=['Annual Income (k$)', 'Spending Score (1-100)'])

# 3. Apply the learned scaling parameters (crucial to prevent data distortion)
incoming_scaled = scaler.transform(incoming_customers)

# 4. Predict cluster assignments
assigned_clusters = kmeans.predict(incoming_scaled)
incoming_customers['Assigned_Cluster'] = assigned_clusters

print("\nCategorized Customer Target Profiles:")
print(incoming_customers.to_string(index=False))