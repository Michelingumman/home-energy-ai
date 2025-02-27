# Price Prediction Models

This directory contains the models and evaluation results for the price prediction system.

## Directory Structure

```
models/
  ├── production/  - Production model (trained on all data)
  │   ├── saved/   - Model files and scalers
  │   └── logs/    - TensorBoard logs
  └── evaluation/  - Evaluation model (with test split)
      ├── saved/   - Model files and scalers
      ├── logs/    - TensorBoard logs
      ├── test_data/ - Test data for evaluation
      └── results/ - Evaluation results and visualizations
          └── comprehensive_evaluation/ - Complete model analysis
```

## Model Types

### Production Model

The production model is trained on all available data without holding out a test set. This maximizes the information available to the model for making real-time predictions. This model is used by the `predictions.py` module for forecasting prices for the current day or future periods.

Files:
- `price_model_production.keras` - The trained model
- `price_scaler.save` - Price scaler for normalizing price inputs/outputs
- `grid_scaler.save` - Grid data scaler for normalizing grid features

### Evaluation Model

The evaluation model is trained on a subset of the data, with separate validation and test sets held out. This allows for rigorous evaluation of model performance on unseen data. This model is used by the `evaluate.py` module to assess prediction quality.

Files:
- `price_model_evaluation.keras` - The trained model
- `price_scaler.save` - Price scaler for normalizing price inputs/outputs
- `grid_scaler.save` - Grid data scaler for normalizing grid features

## Training Models

To train the models, use the `train.py` script with the appropriate mode:

```bash
# Train the production model (uses all available data)
python train.py production

# Train the evaluation model (creates train/val/test split)
python train.py evaluation
```

## Evaluating Model Performance

The `evaluate.py` script provides a comprehensive, automatic analysis of model performance. Simply run:

```bash
python evaluate.py
```

This will automatically:

1. Load the evaluation model and test data
2. Generate predictions across the entire test period
3. Create three focused visualizations for different aspects of performance:
   
   **Time Period Analysis** (`time_period_analysis.png`):
   - Full test period overview with daily averages
   - Yearly comparisons (boxplots and statistical analysis)
   - Monthly comparisons (seasonal patterns and trends)
   - Weekly detailed analysis (with weekend highlights)
   
   **Day Analysis** (`day_comparison.png`):
   - Representative day comparisons (low, high, and volatile price days)
   - Detailed hourly patterns for each day type
   
   **Error Analysis** (`error_analysis.png`):
   - Error distribution and statistics
   - Error patterns by hour of day
   - Residual analysis
   - Actual vs. predicted scatter plots

4. Calculate comprehensive metrics:
   - Basic metrics (MAE, RMSE, MAPE, R²)
   - Advanced metrics (bias, standard error, maximum error, correlation)
   - Time-based analysis (performance by year, month, hour of day)
   - Dataset comparisons (test vs. validation)

## Evaluation Results

Evaluation results are stored in the `evaluation/results/comprehensive_evaluation/` directory and include:

- `time_period_analysis.png` - Multi-panel visualization of performance across different time periods
- `day_comparison.png` - Detailed analysis of representative days
- `error_analysis.png` - Comprehensive error analysis and model diagnostics
- `metrics_summary.md` - Markdown file with all calculated metrics
- `metrics_summary.png` - Visual representation of key metrics
- Various CSV files with detailed metrics for different time periods

The reorganized visualizations provide clearer and more focused presentations of model performance across different aspects, making it easier to assess the model's strengths and weaknesses in specific contexts. Each figure now has more space for details and better readability, making them more suitable for inclusion in reports or presentations. 