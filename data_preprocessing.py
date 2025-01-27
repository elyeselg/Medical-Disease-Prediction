import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path):
    """
    Load the dataset from the given file path.
    :param path: Path to the dataset CSV file.
    :return: DataFrame containing the dataset.
    """
    try:
        data = pd.read_csv(path)
        print("Dataset loaded successfully!")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {path}. Please check the path.")
        return None

def clean_data(data):
    """
    Clean the dataset by handling missing values and correcting data types.
    :param data: DataFrame containing the dataset.
    :return: Cleaned DataFrame.
    """
    print("\nChecking for missing values...")
    print(data.isnull().sum())

    # Drop rows with missing values
    data = data.dropna()
    print("\nMissing values handled (rows with NaNs dropped).")

    # Convert categorical columns to numeric codes
    for col in data.columns:
        if data[col].dtype == 'object':
            print(f" - Converting '{col}' to categorical...")
            data[col] = data[col].astype('category').cat.codes

    print("\nData cleaning complete!")
    return data

def preprocess_data(data):
    """
    Normalize numeric features for consistent ranges.
    :param data: DataFrame containing the cleaned dataset.
    :return: Preprocessed DataFrame (features scaled).
    """
    print("\nNormalizing numeric features...")
    scaler = MinMaxScaler()

    # Scale numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    print("Normalization complete!")
    return data
