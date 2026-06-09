# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load the dataset
df = pd.read_csv('mall_customers.csv')  # change to your filename
print("First 5 rows:")
print(df.head())

# 3. Explore data
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# 4. Select features
X = df[['Annual_Income', 'Spending_Score']]

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show(block=False)  # This won't pause the code
plt.pause(1)  # Show for 1 second
plt.close()   # Close automatically

# 7. Apply K-Means
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 8. Visualise clusters
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score',
                hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title(f'Customer Segments (k={k_optimal})')
plt.show(block=False)
plt.pause(2)
plt.close()

# 9. Cluster profiles
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual_Income': 'mean',
    'Spending_Score': 'mean'
}).round(2)
print("\n" + "="*50)
print("CLUSTER PROFILES (Average values):")
print("="*50)
print(cluster_summary)
print("\n" + "="*50)

# 10. Bar charts
fig, axes = plt.subplots(1, 3, figsize=(15,4))
cluster_summary['Age'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Average Age per Cluster')
cluster_summary['Annual_Income'].plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Average Annual Income per Cluster')
cluster_summary['Spending_Score'].plot(kind='bar', ax=axes[2], color='salmon')
axes[2].set_title('Average Spending Score per Cluster')
plt.tight_layout()
plt.show(block=False)
plt.pause(3)
plt.close()

# 11. Insights
print("\n" + "="*50)
print("INSIGHTS - Customer Segments Analysis")
print("="*50)

# Generate insights based on actual cluster values
for cluster in range(k_optimal):
    cluster_data = cluster_summary.loc[cluster]
    income = cluster_data['Annual_Income']
    spend = cluster_data['Spending_Score']
    age = cluster_data['Age']
    
    if income > 60000 and spend > 50:
        type_desc = "💎 PREMIUM CUSTOMERS - High income, high spending"
    elif income > 60000 and spend <= 50:
        type_desc = "💰 SAVERS - High income but careful spending"
    elif income <= 40000 and spend > 50:
        type_desc = "🛍️ ASPIRATIONAL - Limited budget but enthusiastic spenders"
    elif income <= 40000 and spend <= 50:
        type_desc = "📌 BUDGET CONSCIOUS - Low income, careful spending"
    else:
        type_desc = "📊 AVERAGE CUSTOMERS - Moderate income and spending"
    
    print(f"\n🔹 Cluster {cluster}: {type_desc}")
    print(f"   • Average Age: {age:.0f} years")
    print(f"   • Average Income: ₹{income:,.0f}")
    print(f"   • Average Spending Score: {spend:.0f}/100")

print("\n" + "="*50)

print("="*50)