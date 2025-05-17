import os
import pandas as pd
import joblib
import logging
from code_module.models.model import CropRecommendationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_scaler(scaler_path):
    """
    Load the scaler from disk

    Args:
        scaler_path: Path to the saved scaler

    Returns:
        Loaded scaler
    """
    logger.info(f"Loading scaler from {scaler_path}")

    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    scaler = joblib.load(scaler_path)
    logger.info("Scaler loaded successfully")

    return scaler


def predict_crop(input_data, model_path, scaler_path):
    """
    Make crop recommendations for input data

    Args:
        input_data: Pandas DataFrame or path to CSV with soil/climate data
        model_path: Path to the trained model file
        scaler_path: Path to the saved scaler

    Returns:
        DataFrame with input data and predicted crop
    """
    logger.info("Starting prediction process")

    # Load input data
    if isinstance(input_data, str):
        logger.info(f"Loading input data from {input_data}")
        input_df = pd.read_csv(input_data)
    else:
        input_df = input_data.copy()

    logger.info(f"Input data shape: {input_df.shape}")

    # Load the model
    model = CropRecommendationModel.load(model_path)

    # Load the scaler
    scaler = load_scaler(scaler_path)

    # Scale the input features
    input_scaled = scaler.transform(input_df)

    # Make predictions
    logger.info("Making predictions")
    predictions = model.predict(input_scaled)

    # Add predictions to the input data
    input_df["predicted_crop"] = predictions

    logger.info("Prediction process completed")

    return input_df


def predict_single_sample(
    N, P, K, temperature, humidity, ph, rainfall, model_path, scaler_path
):
    """
    Make a crop recommendation for a single soil/climate sample

    Args:
        N: Nitrogen content in soil
        P: Phosphorus content in soil
        K: Potassium content in soil
        temperature: Temperature in degrees Celsius
        humidity: Relative humidity in percent
        ph: pH value of the soil
        rainfall: Rainfall in mm
        model_path: Path to the trained model file
        scaler_path: Path to the saved scaler

    Returns:
        Predicted crop for the given conditions
    """
    # Create a single sample dataframe
    sample = pd.DataFrame(
        {
            "N": [N],
            "P": [P],
            "K": [K],
            "temperature": [temperature],
            "humidity": [humidity],
            "ph": [ph],
            "rainfall": [rainfall],
        }
    )

    # Make prediction
    result = predict_crop(sample, model_path, scaler_path)

    return result["predicted_crop"].iloc[0]


def batch_predict(input_file, output_file, model_path, scaler_path):
    """
    Make crop recommendations for a batch of samples

    Args:
        input_file: Path to CSV file with input data
        output_file: Path to save predictions
        model_path: Path to the trained model file
        scaler_path: Path to the saved scaler
    """
    logger.info(f"Starting batch prediction from {input_file}")

    # Load input data
    input_data = pd.read_csv(input_file)

    # Make predictions
    predictions = predict_crop(input_data, model_path, scaler_path)

    # Save predictions
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    predictions.to_csv(output_file, index=False)

    logger.info(f"Batch predictions saved to {output_file}")

    return predictions


def main(input_file=None, output_file=None):
    """
    Main function to run the prediction pipeline

    Args:
        input_file: Path to input CSV file (optional)
        output_file: Path to save output predictions (optional)
    """
    logger.info("Starting prediction pipeline")

    # Define paths
    model_path = "models/crop_model_latest.joblib"
    scaler_path = "data/processed/scaler.joblib"

    if input_file and output_file:
        # Batch prediction
        batch_predict(input_file, output_file, model_path, scaler_path)
    else:
        # Example of a single prediction (replace with actual values)
        crop = predict_single_sample(
            N=90,
            P=42,
            K=43,
            temperature=21,
            humidity=82,
            ph=6.5,
            rainfall=200,
            model_path=model_path,
            scaler_path=scaler_path,
        )
        logger.info(f"Predicted crop: {crop}")

    logger.info("Prediction pipeline completed successfully")


if __name__ == "__main__":
    # You can modify this to accept command line arguments
    import sys

    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)
    else:
        main()
