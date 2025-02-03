# Data

This folder stores the datasets used for training, testing, and validating the AI models.

## Structure:
- `raw/`: Contains raw data exported from Home Assistant or other sources.
- `processed/`: Preprocessed data ready for use in machine learning pipelines.
- `example_data.csv`: An example dataset to test the system.

**Usage**:
1. Place your raw data in the `raw/` folder.
2. Preprocess the data using the scripts in the `scripts/` directory and save the results in the `processed/` folder.
3. Use the processed data for model training in `notebooks/` or `scripts/`.
