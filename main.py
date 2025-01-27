# Main execution
from train_model import (
    split_data,
    tune_hyperparameters,
    evaluate_model,
    cross_validate_model,
    plot_feature_importance,
    plot_confusion_matrix,
    save_model,
)
from data_preprocessing import load_dataset, clean_data, preprocess_data
from exploratory_analysis import plot_distributions, plot_correlations, explore_data

# File path to the dataset
DATASET_PATH = "data/chronic_disease.csv"

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset(DATASET_PATH)

    if dataset is not None:
        # Explore the dataset
        explore_data(dataset)

        # Clean the dataset
        dataset = clean_data(dataset)

        # Preprocess the dataset
        dataset = preprocess_data(dataset)

        print("\nPerforming Exploratory Data Analysis...")
        plot_distributions(dataset)
        plot_correlations(dataset)

        # Define target column
        target_column = "classification"

        if target_column in dataset.columns:
            # Split the data
            X_train, X_test, y_train, y_test = split_data(dataset, target_column)

            # Tune hyperparameters and train the model
            model = tune_hyperparameters(X_train, y_train)

            # Evaluate the optimized model
            evaluate_model(model, X_test, y_test)

            # Perform cross-validation
            X = dataset.drop(columns=[target_column])
            y = dataset[target_column]
            cross_validate_model(model, X, y)

            # Plot feature importance
            plot_feature_importance(model, X.columns)

            # Plot confusion matrix
            y_pred = model.predict(X_test)
            plot_confusion_matrix(y_test, y_pred, labels=model.classes_)

            # Save the trained model
            save_model(model, "trained_random_forest.pkl")
        else:
            print(f"Error: Target column '{target_column}' not found.")
