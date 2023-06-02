import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed data
df = pd.read_csv("C:/Users/user/Desktop/preprocessed_mbti_data.csv")

# Define the features and target variable
features = ['danceability', 'valence', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'major_count', 'minor_count']
target = 'mbti'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Create the decision tree classifier
model = DecisionTreeClassifier()

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the model with the best parameters
model = DecisionTreeClassifier(**best_params)
model.fit(X_train, y_train)

# Evaluate on the training set
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluate on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the results
print("모델이 훈련된 데이터에서 얼마나 잘 수행되는지 평가")
print('Training Accuracy:', train_accuracy)
print("새로 들어오는 데이터에 대해 얼마나 잘 예측하는지 평가")
print('Test Accuracy:', test_accuracy)
print("Best Parameters:", best_params)
print("Best Score:", best_score)
