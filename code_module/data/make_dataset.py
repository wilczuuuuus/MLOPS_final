import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(input_filepath):
    """
    Load the crop recommendation dataset from the specified file path.
    
    Args:
        input_filepath: Path to the raw data file
        
    Returns:
        A pandas DataFrame containing the dataset
    """
    logger.info(f'Loading data from {input_filepath}')
    try:
        data = pd.read_csv(input_filepath)
        logger.info(f'Loaded data with shape {data.shape}')
        return data
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise

def preprocess_data(data, output_dir, test_size=0.2, random_state=42):
    """
    Preprocess the data by:
    1. Separating features and target
    2. Splitting into train and test sets
    3. Scaling numerical features
    4. Saving processed data and artifacts
    
    Args:
        data: Pandas DataFrame with the raw dataset
        output_dir: Directory to save processed data and artifacts
        test_size: Size of the test split (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data splits
    """
    logger.info('Preprocessing data')
    
    # Check if data is not empty
    if data.empty:
        logger.error('The dataset is empty')
        raise ValueError('The dataset is empty')
    
    # Separate features and target
    X = data.drop('label', axis=1)  # Assuming the target is called 'label', adjust if needed
    y = data['label']
    
    logger.info(f'Feature columns: {X.columns.tolist()}')
    logger.info(f'Target variable: label with {y.nunique()} unique values')

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the scaler
    scaler_filepath = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_filepath)
    logger.info(f'Saved scaler to {scaler_filepath}')
    
    # Save train and test data
    train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_data['label'] = y_train.values
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    
    test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_data['label'] = y_test.values
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    logger.info(f'Saved processed data to {output_dir}')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def main(input_filepath='data/raw/crop_recommendation.csv', 
         output_filepath='data/processed'):
    """
    Main function to run the data processing pipeline
    
    Args:
        input_filepath: Path to the raw data file
        output_filepath: Directory to save processed data
    """
    logger.info('Starting data processing pipeline')
    
    # Load data
    data = load_data(input_filepath)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data, output_filepath)
    
    logger.info('Data processing completed successfully')
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    main() 