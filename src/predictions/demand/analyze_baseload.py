#!/usr/bin/env python
"""
Baseload Analysis Script

This script analyzes the baseload calculation and model performance issues
by examining the key metrics and data distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import pickle
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set nicer plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (14, 7)

# Constants
DATA_PATH = 'data/processed/villamichelin/VillamichelinEnergyData.csv'
AGENT_MODEL_PATH = 'src/predictions/demand/models/baseload/villamichelin_baseload_model.pkl'
HEAT_PUMP_DATA_PATH = 'data/processed/villamichelin/Thermia/HeatPumpPower.csv'
WEATHER_DATA_PATH = 'data/processed/weather_data.csv'
PLOTS_DIR = 'src/predictions/demand/plots/baseload_analysis/'

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

def load_data():
    """Load the raw energy data"""
    logging.info(f"Loading energy data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')
    df.index = df.index.tz_localize('Europe/Stockholm', ambiguous=True, nonexistent='shift_forward')
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def load_model():
    """Load the trained baseload model"""
    logging.info(f"Loading model from {AGENT_MODEL_PATH}")
    try:
        with open(AGENT_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def analyze_baseload_data(df):
    """Analyze the baseload data and its relationship to consumption/production"""
    logging.info("Analyzing baseload data...")
    
    # Calculate baseload
    if 'production' in df.columns:
        df['baseload'] = df['consumption'] + df['production']
    else:
        logging.warning("Production data not found. Using consumption as baseload.")
        df['baseload'] = df['consumption']
    
    # Basic statistics
    logging.info("\nBasic Statistics:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Create summary statistics
    stats = pd.DataFrame({
        'mean': [df['consumption'].mean(), df['production'].mean(), df['baseload'].mean()],
        'std': [df['consumption'].std(), df['production'].std(), df['baseload'].std()],
        'min': [df['consumption'].min(), df['production'].min(), df['baseload'].min()],
        'max': [df['consumption'].max(), df['production'].max(), df['baseload'].max()],
        'zeros': [(df['consumption'] == 0).sum(), (df['production'] == 0).sum(), (df['baseload'] == 0).sum()],
        'near_zeros': [(df['consumption'] < 0.1).sum(), (df['production'] < 0.1).sum(), (df['baseload'] < 0.1).sum()]
    }, index=['consumption', 'production', 'baseload'])
    
    logging.info("\nSummary Statistics:")
    logging.info(stats)
    
    # Calculate zero and near-zero percentages
    logging.info("\nZero and Near-Zero Analysis:")
    for col in ['consumption', 'production', 'baseload']:
        zeros = (df[col] == 0).sum()
        near_zeros = ((df[col] > 0) & (df[col] < 0.1)).sum()
        logging.info(f"{col.capitalize()} zeros: {zeros} ({zeros/len(df)*100:.2f}%)")
        logging.info(f"{col.capitalize()} near-zeros (<0.1): {near_zeros} ({near_zeros/len(df)*100:.2f}%)")
    
    # Plot distributions
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    sns.histplot(df['consumption'], kde=True, bins=50)
    plt.title('Consumption Distribution')
    plt.xlabel('Consumption (kW)')
    
    plt.subplot(132)
    sns.histplot(df['production'], kde=True, bins=50)
    plt.title('Production Distribution')
    plt.xlabel('Production (kW)')
    
    plt.subplot(133)
    sns.histplot(df['baseload'], kde=True, bins=50)
    plt.title('Baseload Distribution')
    plt.xlabel('Baseload (kW)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'distribution_comparison.png'))
    plt.close()
    
    # Plot time series for a random week to see patterns
    # Find a week with non-zero production
    weeks = pd.Series(df.index.isocalendar().week.unique())
    for week in weeks.sample(frac=1):  # shuffle and try different weeks
        week_data = df[df.index.isocalendar().week == week]
        if week_data['production'].max() > 0.5:  # Found a week with some production
            break
    
    week_data = week_data.sort_index()
    start_date = week_data.index[0]
    end_date = week_data.index[-1]
    
    plt.figure(figsize=(16, 8))
    plt.plot(week_data.index, week_data['consumption'], label='Consumption')
    plt.plot(week_data.index, week_data['production'], label='Production')
    plt.plot(week_data.index, week_data['baseload'], label='Baseload', linestyle='--')
    plt.title(f'Energy Data: {start_date.date()} to {end_date.date()}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'time_series_comparison.png'))
    plt.close()
    
    # Plot baseload vs consumption with identity line
    plt.figure(figsize=(10, 10))
    plt.scatter(df['consumption'], df['baseload'], alpha=0.5)
    max_val = max(df['consumption'].max(), df['baseload'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    plt.title('Baseload vs Consumption')
    plt.xlabel('Consumption (kW)')
    plt.ylabel('Baseload (kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'baseload_vs_consumption.png'))
    plt.close()
    
    return df

def analyze_persistence_models(df):
    """Analyze the performance of different persistence models"""
    logging.info("Analyzing persistence model performance...")
    
    # Create persistence models
    df['persistence_1h'] = df['baseload'].shift(1)
    df['persistence_24h'] = df['baseload'].shift(24)
    df['persistence_168h'] = df['baseload'].shift(168)  # 1 week
    
    # Create sliding window averages
    df['avg_24h'] = df['baseload'].rolling(24).mean().shift(1)
    df['avg_168h'] = df['baseload'].rolling(168).mean().shift(1)
    
    # Calculate absolute errors
    for col in ['persistence_1h', 'persistence_24h', 'persistence_168h', 'avg_24h', 'avg_168h']:
        df[f'{col}_error'] = np.abs(df[col] - df['baseload'])
    
    # Drop NaN values for analysis
    df_valid = df.dropna(subset=[c for c in df.columns if '_error' in c])
    
    # Calculate mean errors
    errors = {
        'Persistence (1h)': df_valid['persistence_1h_error'].mean(),
        'Persistence (24h)': df_valid['persistence_24h_error'].mean(),
        'Persistence (168h)': df_valid['persistence_168h_error'].mean(),
        'Moving Avg (24h)': df_valid['avg_24h_error'].mean(),
        'Moving Avg (168h)': df_valid['avg_168h_error'].mean()
    }
    
    logging.info("\nMean Absolute Errors:")
    for model, error in sorted(errors.items(), key=lambda x: x[1]):
        logging.info(f"{model}: {error:.4f}")
    
    # Plot errors
    plt.figure(figsize=(12, 6))
    plt.bar(errors.keys(), errors.values())
    plt.title('Mean Absolute Error by Model')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'persistence_model_comparison.png'))
    plt.close()
    
    # Analyze error patterns over time
    # Sample a month of data
    recent_months = sorted(list(set([f"{ts.year}-{ts.month:02d}" for ts in df.index])))[-3:]
    latest_month = recent_months[-1]
    year, month = map(int, latest_month.split('-'))
    
    # Filter for the latest month
    month_data = df[(df.index.year == year) & (df.index.month == month)]
    
    plt.figure(figsize=(16, 10))
    
    plt.subplot(211)
    plt.plot(month_data.index, month_data['baseload'], label='Actual Baseload')
    plt.plot(month_data.index, month_data['persistence_24h'], label='24h Persistence', linestyle='--')
    plt.title(f'Baseload vs. 24h Persistence: {year}-{month:02d}')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.subplot(212)
    plt.plot(month_data.index, month_data['persistence_24h_error'], label='24h Persistence Error')
    plt.title('Absolute Error Over Time')
    plt.ylabel('Absolute Error')
    plt.axhline(y=month_data['persistence_24h_error'].mean(), color='r', linestyle='--', 
                label=f'Mean Error: {month_data["persistence_24h_error"].mean():.4f}')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'persistence_errors_{year}_{month:02d}.png'))
    plt.close()
    
    return df_valid

def analyze_model_feature_importance(model, df):
    """Analyze feature importance from the trained model"""
    if model is None:
        logging.warning("Model not loaded. Skipping feature importance analysis.")
        return
    
    logging.info("Analyzing model feature importance...")
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = model.feature_names_in_
    
    # Sort by importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Print top features
    logging.info("\nTop 20 Important Features:")
    for i, idx in enumerate(indices[:20]):
        logging.info(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.6f}")
    
    # Plot top 20 features
    plt.figure(figsize=(12, 10))
    top_indices = indices[:20]
    plt.barh(range(len(top_indices)), feature_importance[top_indices])
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'))
    plt.close()
    
    # Analyze feature correlation with target
    common_features = [f for f in feature_names if f in df.columns]
    if common_features:
        corr_df = df[['baseload'] + common_features].corr()['baseload'].abs().sort_values(ascending=False)
        
        logging.info("\nTop 20 Features by Correlation with Baseload:")
        for i, (feature, corr) in enumerate(corr_df.iloc[1:21].items()):
            logging.info(f"{i+1}. {feature}: {corr:.6f}")
        
        # Plot top correlated features
        plt.figure(figsize=(12, 10))
        corr_df.iloc[1:21].plot(kind='barh')
        plt.title('Top 20 Features by Correlation with Baseload')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_correlation.png'))
        plt.close()

def main():
    """Main analysis function"""
    logging.info("=" * 60)
    logging.info("BASELOAD MODEL ANALYSIS")
    logging.info("=" * 60)
    
    # Load data
    df = load_data()
    
    # Analyze baseload
    df = analyze_baseload_data(df)
    
    # Analyze persistence models
    df_valid = analyze_persistence_models(df)
    
    # Load and analyze model
    model = load_model()
    analyze_model_feature_importance(model, df_valid)
    
    logging.info("\nAnalysis complete. Results saved to: " + PLOTS_DIR)
    logging.info("\nSUMMARY OF FINDINGS:")
    logging.info("1. Check the proportion of zero and near-zero values in baseload")
    logging.info("2. Examine the relationship between baseload and consumption")
    logging.info("3. Compare performance of persistence models")
    logging.info("4. Look at feature importance to understand model decisions")
    logging.info("5. Review error patterns over time")

if __name__ == "__main__":
    main() 