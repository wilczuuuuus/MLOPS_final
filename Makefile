.PHONY: requirements preprocess train predict lint resolve clean

# Install dependencies
requirements:
	pip install -r requirements.txt

# Run data preprocessing
preprocess:
	python -m code_module.data.make_dataset

# Train the model
train:
	python -m code_module.train.train_model

# Make predictions with the trained model
predict:
	python -m code_module.train.predict_model data/processed/test.csv models/predictions.csv

# Lint code with flake8 and black
lint:
	flake8 code_module/
	black --check code_module/

# Resolve any issues (combining data preprocessing and model evaluation)
resolve:
	python -m code_module.data.make_dataset
	python -m code_module.train.train_model

# Clean generated files
clean:
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf reports/figures/*

# Run the full pipeline
pipeline:
	python run_pipeline.py 