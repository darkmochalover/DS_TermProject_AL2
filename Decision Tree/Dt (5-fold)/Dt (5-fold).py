import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed data
df = pd.read_csv("C:/Users/user/Desktop/preprocessed_mbti_data.csv")

# Define the features and target variable
features = ['danceability', 'valence', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'major_count', 'minor_count']
target = 'mbti'

# Validate column names
invalid_columns = [col for col in features + [target] if col not in df.columns]
if invalid_columns:
    raise ValueError(f"The following columns are not present in the DataFrame: {', '.join(invalid_columns)}")

X = df[features]
y = df[target]


# Define the number of folds
k = 5

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_train_accuracies = []
fold_test_accuracies = []

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Initialize and fit the model
    model = RandomForestClassifier()  
    model.fit(X_train, y_train)

    # Predict on the train set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    fold_train_accuracies.append(train_accuracy)

    # Predict on the test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    fold_test_accuracies.append(test_accuracy)

    print(f"Fold {fold} Train Accuracy: {train_accuracy}")
    print(f"Fold {fold} Test Accuracy: {test_accuracy}")

# Calculate and print the average train and test accuracies across folds
average_train_accuracy = sum(fold_train_accuracies) / k
average_test_accuracy = sum(fold_test_accuracies) / k

print("\nAverage Training Accuracy:", average_train_accuracy)
print("Average Test Accuracy:", average_test_accuracy)
