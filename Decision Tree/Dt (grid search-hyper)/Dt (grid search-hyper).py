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

# Define the parameter grid for k
param_grid = {
    'k': [3, 5, 7, 10]
}

# Create the StratifiedKFold object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=skf)

# Perform grid search to find the optimal k
grid_search.fit(X, y_encoded)

# Get the best estimator and best hyperparameters from grid search
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Perform k-fold cross-validation with the best k
fold_train_accuracies = []
fold_test_accuracies = []
for fold, (train_index, test_index) in enumerate(skf.split(X, y_encoded), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Fit the best model on the training data
    best_model.fit(X_train, y_train)

    # Predict on the training data
    train_predictions = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    # Predict on the test data
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Print the train accuracy, test accuracy, and best hyperparameters for each fold
    print(f"Fold {fold} Train Accuracy: {train_accuracy}")
    print(f"Fold {fold} Test Accuracy: {test_accuracy}")
    print(f"Best Hyperparameters: {best_params}")

    # Append the train accuracy and test accuracy to the fold lists
    fold_train_accuracies.append(train_accuracy)
    fold_test_accuracies.append(test_accuracy)

# Calculate and print the average train and test accuracies across folds
average_train_accuracy = sum(fold_train_accuracies) / len(fold_train_accuracies)
average_test_accuracy = sum(fold_test_accuracies) / len(fold_test_accuracies)

print("\nAverage Training Accuracy:", average_train_accuracy)
print("Average Test Accuracy:", average_test_accuracy)
