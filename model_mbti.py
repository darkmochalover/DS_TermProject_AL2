import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('drive/MyDrive/preprocessed_mbti_data.csv')

# Define color palette for MBTI types
mbti_color_palette = {
    'INTJ': 'red',
    'INTP': 'blue',
    'ENTJ': 'green',
    'ENTP': 'purple',
    'INFJ': 'orange',
    'INFP': 'brown',
    'ENFJ': 'pink',
    'ENFP': 'gray',
    'ISTJ': 'olive',
    'ISFJ': 'cyan',
    'ESTJ': 'magenta',
    'ESFJ': 'lime',
    'ISTP': 'gold',
    'ISFP': 'teal',
    'ESTP': 'navy',
    'ESFP': 'silver'
}

# Classification Modeling (Random Forest)
features = ['danceability', 'valence', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'major_count', 'minor_count']
target = 'mbti'

X = data[features]
y = data[target]

# Convert MBTI labels to numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Random Forest Model with feature importance selection
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a dataframe with feature names and importance scores
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the dataframe by importance score in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Select the top 5 features
top_features = feature_importance_df['Feature'].head(5).tolist()

X_top_features = X[top_features]

X_train, X_test, y_train, y_test = train_test_split(X_top_features, y_encoded, test_size=0.2, random_state=42)

# Random Forest Model with top 5 features
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy with Top 5 Features:", rf_accuracy)

# Clustering Modeling (K-Means)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Map MBTI types to colors
target_colors = [mbti_color_palette[mbti] for mbti in data[target]]

# Regression Modeling (Linear Regression)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
reg_predictions = reg_model.predict(X_test)
reg_mse = mean_squared_error(y_test, reg_predictions)
print("Mean Squared Error (Regression):", reg_mse)

# Visualization of Classification Results
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=target_colors)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Classification Results')
plt.show()

# Visualization of Clustering Results
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustering Results')
plt.show()
