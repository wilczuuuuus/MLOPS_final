import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CropRecommendationModel:
    """
    A model for crop recommendation based on soil and climate data
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the crop recommendation model
        
        Args:
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of the trees (default: None)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """
        Train the model on the provided data
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self: The trained model
        """
        logger.info('Training the model')
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info('Model training completed')
        return self
        
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted crop labels
        """
        if not self.is_trained:
            logger.error('Model is not trained yet')
            raise ValueError('The model must be trained before making predictions')
            
        logger.info('Making predictions')
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            logger.error('Model is not trained yet')
            raise ValueError('The model must be trained before evaluation')
            
        logger.info('Evaluating the model')
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f'Model accuracy: {accuracy:.4f}')
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return metrics
    
    def save(self, model_filepath):
        """
        Save the trained model to disk
        
        Args:
            model_filepath: Path where the model will be saved
        """
        if not self.is_trained:
            logger.error('Model is not trained yet')
            raise ValueError('The model must be trained before saving')
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
        
        logger.info(f'Saving model to {model_filepath}')
        joblib.dump(self.model, model_filepath)
        logger.info('Model saved successfully')
        
    @classmethod
    def load(cls, model_filepath):
        """
        Load a trained model from disk
        
        Args:
            model_filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        logger.info(f'Loading model from {model_filepath}')
        
        # Check if model file exists
        if not os.path.exists(model_filepath):
            logger.error(f'Model file not found: {model_filepath}')
            raise FileNotFoundError(f'Model file not found: {model_filepath}')
        
        # Create a new instance
        instance = cls()
        
        # Load the model
        instance.model = joblib.load(model_filepath)
        instance.is_trained = True
        
        logger.info('Model loaded successfully')
        return instance
    
    def get_feature_importance(self):
        """
        Get the feature importance from the trained model
        
        Returns:
            DataFrame with feature names and their importance scores
        """
        if not self.is_trained:
            logger.error('Model is not trained yet')
            raise ValueError('The model must be trained to get feature importances')
            
        # Get feature importances
        importances = self.model.feature_importances_
        
        # We need feature names for this to work properly
        # Assuming feature_names would be passed somehow
        # For now, return just the importances
        return importances 