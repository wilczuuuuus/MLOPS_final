import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import logging
from datetime import datetime
from code_module.data.make_dataset import load_data, preprocess_data
from code_module.models.model import CropRecommendationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_feature_importance(model, feature_names, output_dir):
    """
    Plot feature importance

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        output_dir: Directory to save the plot
    """
    # Get feature importances
    importances = model.get_feature_importance()

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance")
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path)
    logger.info(f"Feature importance plot saved to {plot_path}")

    return feature_importance_df


def plot_confusion_matrix(confusion_matrix, class_names, output_dir):
    """
    Plot confusion matrix

    Args:
        confusion_matrix: Confusion matrix to plot
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    logger.info(f"Confusion matrix plot saved to {plot_path}")


def train_and_evaluate(
    data_path="data/raw/crop_recommendation.csv",
    processed_data_dir="data/processed",
    model_dir="models",
    reports_dir="reports/figures",
    n_estimators=100,
    max_depth=None,
    test_size=0.2,
    random_state=42,
):
    """
    Train and evaluate the crop recommendation model

    Args:
        data_path: Path to the raw data file
        processed_data_dir: Directory for processed data
        model_dir: Directory to save the model
        reports_dir: Directory to save reports and figures
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility

    Returns:
        Evaluation metrics
    """
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth if max_depth else "None")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        data = load_data(data_path)
        X_train, X_test, y_train, y_test = preprocess_data(
            data, processed_data_dir, test_size=test_size, random_state=random_state
        )

        # Get feature names and class names
        feature_names = list(data.drop("label", axis=1).columns)
        class_names = sorted(data["label"].unique())

        # Initialize and train the model
        logger.info("Initializing and training the model")
        model = CropRecommendationModel(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        model.train(X_train, y_train)

        # Evaluate the model
        logger.info("Evaluating the model")
        metrics = model.evaluate(X_test, y_test)

        # Log metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        for class_name in metrics["classification_report"]:
            if class_name in ["accuracy", "macro avg", "weighted avg"]:
                continue
            precision_metrics = metrics["classification_report"]
            precision = precision_metrics[class_name]["precision"]
            recall = metrics["classification_report"][class_name]["recall"]
            f1 = metrics["classification_report"][class_name]["f1-score"]
            mlflow.log_metric(f"{class_name}_precision", precision)
            mlflow.log_metric(f"{class_name}_recall", recall)
            mlflow.log_metric(f"{class_name}_f1", f1)

        # Plot feature importance and log artifact
        plot_feature_importance(model, feature_names, reports_dir)
        artifact_path = os.path.join(reports_dir, "feature_importance.png")
        mlflow.log_artifact(artifact_path)

        # Plot confusion matrix
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), class_names, reports_dir
        )
        mlflow.log_artifact(os.path.join(reports_dir, "confusion_matrix.png"))

        # Save metrics to JSON file
        metrics_path = os.path.join(reports_dir, "metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metrics_path)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"crop_model_{timestamp}.joblib")
        model.save(model_path)
        mlflow.log_artifact(model_path)

        # Also save as "latest" model for easier access
        latest_path = os.path.join(model_dir, "crop_model_latest.joblib")
        model.save(latest_path)

        accuracy = metrics["accuracy"]
        logger.info(f"Training completed with accuracy: {accuracy:.4f}")

        return metrics


def main():
    """
    Main function to run the training pipeline
    """
    logger.info("Starting model training pipeline")

    # Set MLflow tracking URI from environment variable or use default if not set
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info(f"Using MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Ensure bucket name is set before any MLflow operations
    bucket_name = os.environ.get("ARTIFACT_BUCKET", "")
    if not bucket_name or bucket_name.strip() == "":
        bucket_name = "mlops-final-task"
        os.environ["ARTIFACT_BUCKET"] = bucket_name
    
    # Set MLflow artifact URI with exactly two slashes after s3:
    artifact_root = f"s3://{bucket_name}/models"
    # Set both environment variables to ensure consistency
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_root
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://s3.amazonaws.com"
    
    logger.info(f"Setting artifact root to: {artifact_root}")
    logger.info(f"Setting S3 endpoint URL to: http://s3.amazonaws.com")
    
    mlflow.set_experiment("crop-recommendation")
    
    # Log AWS configuration for debugging
    logger.info("AWS environment variables:")
    logger.info(f"AWS_ACCESS_KEY_ID set: {'Yes' if os.environ.get('AWS_ACCESS_KEY_ID') else 'No'}")
    logger.info(f"AWS_SECRET_ACCESS_KEY set: {'Yes' if os.environ.get('AWS_SECRET_ACCESS_KEY') else 'No'}")
    logger.info(f"AWS_DEFAULT_REGION set: {'Yes' if os.environ.get('AWS_DEFAULT_REGION') else 'No'}")
    logger.info(f"ARTIFACT_BUCKET set: {bucket_name}")
    
    # Test MLflow artifact URI immediately to verify format
    try:
        with mlflow.start_run() as run:
            logger.info(f"Test run ID: {run.info.run_id}")
            artifact_uri = mlflow.get_artifact_uri()
            logger.info(f"Artifact URI: {artifact_uri}")
            
            # Verify S3 URI format
            if artifact_uri.startswith("s3://"):
                logger.info(f"S3 URI format looks correct with double slashes")
                from urllib.parse import urlparse
                parsed_uri = urlparse(artifact_uri)
                logger.info(f"S3 path components: scheme={parsed_uri.scheme}, netloc={parsed_uri.netloc}, path={parsed_uri.path}")
            else:
                logger.warning(f"S3 URI format might be incorrect: {artifact_uri}")
            
            # Log a test artifact to verify S3 access
            test_path = "test_artifact.txt"
            with open(test_path, "w") as f:
                f.write("Test artifact to verify S3 access")
            
            try:
                mlflow.log_artifact(test_path)
                logger.info("Successfully logged test artifact")
            except Exception as e:
                logger.error(f"Failed to log test artifact: {e}")
            
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error checking MLflow artifact URI: {e}")
    
    # Set up AWS S3 endpoint and perform training
    try:
        # Ensure correct S3 endpoint configuration for boto3
        os.environ["AWS_S3_ENDPOINT"] = "s3.amazonaws.com"
        
        metrics = train_and_evaluate()
        logger.info("Model training pipeline completed successfully")
        return metrics
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
