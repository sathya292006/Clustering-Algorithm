import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------
# Title
# -------------------------------
st.title("🏦 Bank Customer Clustering using K-Means")

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("bank.csv", sep=';')

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Convert numeric columns
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='ignore')

# Select numeric data
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# -------------------------------
# Feature Selection
# -------------------------------
features = st.multiselect(
    "🎯 Select Features",
    numeric_data.columns,
    default=list(numeric_data.columns[:2])
)

# -------------------------------
# Validation
# -------------------------------
if len(features) < 2:
    st.warning("⚠️ Please select at least 2 features")

else:
    X = numeric_data[features]

    # -------------------------------
    # Select K
    # -------------------------------
    k = st.slider("📌 Number of Clusters (k)", 2, 10, 3)

    # -------------------------------
    # Apply KMeans
    # -------------------------------
    model = KMeans(n_clusters=k, random_state=42)
    data["cluster"] = model.fit_predict(X)

    # -------------------------------
    # Show Cluster Graph ONLY
    # -------------------------------
    st.subheader("📈 Cluster Visualization")

    fig, ax = plt.subplots()

    for i in range(k):
        cluster = data[data["cluster"] == i]
        ax.scatter(
            cluster[features[0]],
            cluster[features[1]],
            label=f"Cluster {i}"
        )

    # Centroids
    centroids = model.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        marker='X',
        label='Centroids'
    )

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.legend()

    st.pyplot(fig)