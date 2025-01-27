import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data):
    """
    Display basic information about the dataset.
    :param data: DataFrame containing the dataset.
    """
    print("\nPreview of the dataset:")
    print(data.head())

    print("\nStatistical summary:")
    print(data.describe())

    print("\nDataset information:")
    print(data.info())

def plot_distributions(data):
    """
    Plot the distributions of numeric features.
    :param data: DataFrame containing the dataset.
    """
    print("\nPlotting feature distributions...")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

def plot_correlations(data):
    """
    Plot a heatmap to show correlations between features.
    :param data: DataFrame containing the dataset.
    """
    print("\nPlotting correlation heatmap...")
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
