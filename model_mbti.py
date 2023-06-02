import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# Load
# Classification Modeling (Random Forest)
features = ['danceability', 'valence', 'energy', 'loudness', 'acousticness']
target = 'mbti'

X = data[features]
y = data[target]

# Convert MBTI labels to numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)

# Clustering Modeling (K-Means)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Map MBTI types to colors
target_colors = [mbti_color_palette[mbti] for mbti in data[target]]

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
