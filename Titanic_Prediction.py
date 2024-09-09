import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_csv("train.csv")

# Feature engineering: create a new feature 'FamilySize'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1


# Select features and preprocess them
features = ["Pclass", "Sex", "FamilySize", "Fare"]
X = pd.get_dummies(train_data[features], drop_first=True)
y = train_data["Survived"]

# Create a pipeline with preprocessing and XGBoost model
xgb_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scale', StandardScaler()),  # Scale features
    ('model', xgb.XGBClassifier(objective='binary:logistic', random_state=1))  # XGBoost model
])

# Define a grid of hyperparameters to search for XGBoost
xgb_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.8, 1],
    'model__colsample_bytree': [0.8, 1],
}

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Set up the grid search for XGBoost
xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='accuracy')

# Perform the grid search on the training set
xgb_grid_search.fit(X_train, y_train)

# Get the best XGBoost model
best_xgb_model = xgb_grid_search.best_estimator_

# Predict on the validation set
y_pred = best_xgb_model.predict(X_val)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Create a table showing the correct values versus predicted values
prediction_comparison = pd.DataFrame({'True Values': y_val, 'Predicted Values': y_pred})

# Print the best parameters and the best score for XGBoost
print(f"Best XGBoost parameters: {xgb_grid_search.best_params_}")
print(f"Best XGBoost CV score: {xgb_grid_search.best_score_}")

# Display the first few entries of the prediction comparison table
print(prediction_comparison.head())

# Calculate and plot the correlation matrix
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Independent Features')
plt.show()

test_data = pd.read_csv("test.csv")
# Preprocess the test data
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
X_test = pd.get_dummies(test_data[features], drop_first=True)
test_predictions = best_xgb_model.predict(X_test)

# Create the output DataFrame with 'PassengerId' and 'Survived' columns
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
print(output.head())

# Save the predictions to a CSV file (optional)
output.to_csv('titanic_predictions.csv', index=False)

gender_submission = pd.read_csv("gender_submission.csv")

# Ensure the PassengerId matches between your predictions and the gender_submission
# This is important to ensure that you are comparing the predictions with the correct labels
assert all(output['PassengerId'] == gender_submission['PassengerId']), "PassengerId does not match between predictions and true labels."

# Calculate the accuracy of your predictions
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(gender_submission['Survived'], output['Survived'])
print(f"Accuracy on the test set: {accuracy}")