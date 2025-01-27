from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

def split_data(data, target_col):
    """
    Split the dataset into training and testing sets.
    """
    print("\nSplitting data into training and testing sets...")
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def tune_hyperparameters(X_train, y_train):
    """
    Tune the hyperparameters of the Random Forest model using Grid Search.
    """
    print("\nTuning hyperparameters using Grid Search...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    """
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def cross_validate_model(model, X, y):
    """
    Perform cross-validation on the model.
    """
    print("\nPerforming cross-validation...")
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

def plot_feature_importance(model, feature_names):
    """
    Plot the importance of features in the trained Random Forest model.
    """
    importance = model.feature_importances_
    sorted_indices = importance.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance[sorted_indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in sorted_indices], rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, labels):
    """
    Plot a confusion matrix for the model predictions.
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}.")
