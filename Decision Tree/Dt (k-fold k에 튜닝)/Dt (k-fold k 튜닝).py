import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed data
df = pd.read_csv("C:/Users/user/Desktop/preprocessed_mbti_data.csv")

# Define the features and target variable
features = ['danceability', 'valence', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'major_count', 'minor_count']
target = 'mbti'

# Split the data into features and target
X = df[features]
y = df[target]

# Encode the target variable
y_encoded = pd.factorize(y)[0]

# Define the different values of k for k-fold cross-validation
k_values = [3, 5, 7, 10]

# Perform grid search for k-fold cross-validation
best_params = {}
best_avg_test_accuracy = 0.0

for k in k_values:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    model = DecisionTreeClassifier()

    # Define the parameter grid for grid search
    param_grid = {'max_depth': [None, 3, 5, 7], 'min_samples_split': [2, 5, 10]}

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=skf)
    grid_search.fit(X, y_encoded)

    # Get the best parameters and best average test accuracy
    if grid_search.best_score_ > best_avg_test_accuracy:
        best_params = grid_search.best_params_
        best_avg_test_accuracy = grid_search.best_score_

    print(f"Best params (k={k}): {grid_search.best_params_}")
    print(f"Best average test accuracy (k={k}): {grid_search.best_score_}\n")

print(f"Best Hyperparameters: {best_params}")
print(f"Best average test accuracy: {best_avg_test_accuracy}")
