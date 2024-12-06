import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mysql.connector
import joblib

# Connect to the MySQL Database
def get_db_connection():
    connection = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='cdss',
        database='CDSS Diabetes'
    )
    return connection

# Fetch data from the database
connection = get_db_connection()
query = "SELECT * FROM DatasetofDiabetes"
df = pd.read_sql(query, connection)
connection.close()

# Display first few rows of the dataset to verify
print("First 5 rows of the dataset:")
print(df.head())

# Data Preprocessing
label_encoder = LabelEncoder()

# Encode 'Gender' column (e.g., 'M' -> 1, 'F' -> 0)
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Map 'CLASS' column values: 'N' -> 0 (non-diabetic), 'P' -> 1 (pre-diabetic), 'Y' -> 2 (diabetic)
class_mapping = {'N': 0, 'P': 1, 'Y': 2}
df['CLASS'] = df['CLASS'].map(class_mapping)

# Drop missing values
df.dropna(inplace=True)

# Ensure not needed columns are not included in training of models
X = df.drop(columns=['ID', 'CLASS', 'No_Pation']) 
y = df['CLASS']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Cross-validation function
def cross_validation_evaluation(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.4f}")
    print(f"Standard Deviation: {np.std(scores):.4f}")

# Logistic Regression Model
def logistic_regression_model(X_train, X_test, y_train, y_test):
    print("\nRunning Logistic Regression...")
    lr = LogisticRegression(solver='liblinear', max_iter=1000)
    param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10]}
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Logistic Regression Results")
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=1))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cross_validation_evaluation(best_model, X_scaled, y)
    return best_model

# Decision Tree Model
def decision_tree_model(X_train, X_test, y_train, y_test):
    print("\nRunning Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    print("Decision Tree Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=1))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cross_validation_evaluation(dt, X_scaled, y)
    return dt

# Random Forest Model (For Prediction)
def random_forest_model(X_train, X_test, y_train, y_test):
    print("\nRunning Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Random Forest Results")
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=1))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cross_validation_evaluation(best_model, X_scaled, y)
    return best_model

# Train Models
logistic_model = logistic_regression_model(X_train, X_test, y_train, y_test)
decision_tree_model = decision_tree_model(X_train, X_test, y_train, y_test)
random_forest_model = random_forest_model(X_train, X_test, y_train, y_test)

# Save models
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(decision_tree_model, 'decision_tree_model.pkl')
joblib.dump(random_forest_model, 'random_forest_model.pkl')

# Save Preprocessing Tools
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models and preprocessing tools saved successfully.")

# For the Webpage: Use Random Forest as the predictor
def predict_random_forest(features):
    """
    Predict diabetes using the Random Forest model.

    Parameters:
        features (array-like): A 2D array of input features (same shape as the training data).

    Returns:
        int: Predicted class (0: Non-diabetic, 1: Pre-diabetic, 2: Diabetic).
    """
    # Load the Random Forest model
    rf_model = joblib.load('random_forest_model.pkl')

    # Make the prediction
    prediction = rf_model.predict(features)
    return prediction[0]  # Return the prediction for the given input features
