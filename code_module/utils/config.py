import os
import yaml
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration values
    """
    logger.info(f'Loading configuration from {config_path}')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info('Configuration loaded successfully')
        return config
    except Exception as e:
        logger.error(f'Error loading configuration: {e}')
        # Return a default configuration if the file doesn't exist
        return {
            'data': {
                'raw_path': 'data/raw/crop_recommendation.csv',
                'processed_dir': 'data/processed',
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            },
            'training': {
                'model_dir': 'models',
                'reports_dir': 'reports/figures'
            },
            'mlflow': {
                'tracking_uri': 'http://localhost:5000',
                'experiment_name': 'crop-recommendation'
            }
        }

def get_env_var(var_name, default=None):
    """
    Get an environment variable or return a default value
    
    Args:
        var_name: Name of the environment variable
        default: Default value if the environment variable is not set
        
    Returns:
        Value of the environment variable or the default value
    """
    value = os.environ.get(var_name, default)
    
    if value is None:
        logger.warning(f'Environment variable {var_name} not set and no default provided')
        
    return value 