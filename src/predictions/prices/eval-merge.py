import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import tensorflow as tf
import xgboost as xgb
import os
from sklearn.preprocessing import StandardScaler

def load_models():
    """Load the trend, peak and valley models"""
    # Load trend model
    trend_model_path = "src/predictions/prices/models/trend_model/best_trend_model.pkl"
    trend_model = xgb.Booster(model_file=trend_model_path)
    trend_features = [] # Load your feature list here
    
    # Load peak model
    peak_model_path = "src/predictions/prices/models/peak_model/best_peak_model.keras"
    peak_model = tf.keras.models.load_model(peak_model_path, compile=False)
    peak_features = [] # Load your feature list here
    peak_scaler = StandardScaler()
    # Load your scaler here
    
    # Load valley model
    valley_model_path = "src/predictions/prices/models/valley_model/best_valley_model.keras"
    valley_model = tf.keras.models.load_model(valley_model_path, compile=False)
    valley_features = [] # Load your feature list here
    valley_scaler = StandardScaler()
    # Load your scaler here
    
    return {
        "trend": {"model": trend_model, "features": trend_features},
        "peak": {"model": peak_model, "features": peak_features, "scaler": peak_scaler},
        "valley": {"model": valley_model, "features": valley_features, "scaler": valley_scaler}
    }

def load_data(start_date=None, end_date=None):
    """Load price data for the specified period"""
    # Implement your data loading logic here
    # Should return a DataFrame with price data and necessary features
    pass

def process_data_for_models(df, models):
    """Prepare data for each model by adding required features"""
    # Add trend features
    # Implement trend feature generation
    
    # Add peak features
    # Implement peak feature generation
    
    # Add valley features
    # Implement valley feature generation
    
    return df

def generate_predictions(df, models, peak_threshold=0.8, valley_threshold=0.65):
    """Generate predictions from all models"""
    # Generate trend predictions
    trend_features = [col for col in df.columns if col in models["trend"]["features"]]
    dmatrix = xgb.DMatrix(df[trend_features])
    trend_predictions = models["trend"]["model"].predict(dmatrix)
    
    # Generate peak predictions and probabilities
    peak_features = [col for col in df.columns if col in models["peak"]["features"]]
    peak_sequences = create_sequences(df[peak_features], window_size=168)
    peak_sequences_scaled = models["peak"]["scaler"].transform(peak_sequences.reshape(-1, peak_sequences.shape[-1]))
    peak_sequences_scaled = peak_sequences_scaled.reshape(peak_sequences.shape)
    peak_probabilities = models["peak"]["model"].predict(peak_sequences_scaled)
    peak_predictions = (peak_probabilities >= peak_threshold).astype(int)
    
    # Generate valley predictions and probabilities
    valley_features = [col for col in df.columns if col in models["valley"]["features"]]
    valley_sequences = create_sequences(df[valley_features], window_size=168)
    valley_sequences_scaled = models["valley"]["scaler"].transform(valley_sequences.reshape(-1, valley_sequences.shape[-1]))
    valley_sequences_scaled = valley_sequences_scaled.reshape(valley_sequences.shape)
    valley_probabilities = models["valley"]["model"].predict(valley_sequences_scaled)
    valley_predictions = (valley_probabilities >= valley_threshold).astype(int)
    
    return {
        "trend": trend_predictions,
        "peak_prob": peak_probabilities,
        "peak": peak_predictions,
        "valley_prob": valley_probabilities,
        "valley": valley_predictions
    }

def create_sequences(data, window_size=168):
    """Create sequences for time series model input"""
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data.iloc[i:i+window_size].values)
    return np.array(sequences)

def merge_predictions(df, predictions, peak_influence=30, valley_influence=-15):
    """Merge trend, peak and valley predictions into combined forecast"""
    # Start with trend predictions
    df['trend_prediction'] = predictions['trend']
    df['merged_prediction'] = df['trend_prediction'].copy()
    
    # Resolve conflicts - where both peak and valley are predicted
    conflicts = (predictions['peak'] == 1) & (predictions['valley'] == 1)
    if conflicts.sum() > 0:
        # Choose peak or valley based on which has higher probability
        for i in np.where(conflicts)[0]:
            if predictions['peak_prob'][i] > predictions['valley_prob'][i]:
                predictions['valley'][i] = 0
            else:
                predictions['peak'][i] = 0
    
    # Apply peak influence
    peak_indices = np.where(predictions['peak'] == 1)[0]
    for idx in peak_indices:
        # Apply peak influence with smoothing around peaks
        influence_range = 12  # hours around peak to apply smoothing
        for i in range(-influence_range, influence_range + 1):
            if 0 <= idx + i < len(df):
                # Apply decreasing influence with distance from peak
                weight = 1 - abs(i) / (influence_range + 1)
                df.loc[df.index[idx + i], 'merged_prediction'] += peak_influence * weight
    
    # Apply valley influence
    valley_indices = np.where(predictions['valley'] == 1)[0]
    for idx in valley_indices:
        # Apply valley influence with smoothing around valleys
        influence_range = 12  # hours around valley to apply smoothing
        for i in range(-influence_range, influence_range + 1):
            if 0 <= idx + i < len(df):
                # Apply decreasing influence with distance from valley
                weight = 1 - abs(i) / (influence_range + 1)
                df.loc[df.index[idx + i], 'merged_prediction'] += valley_influence * weight
    
    return df

def plot_results(df, predictions, output_dir="plots", date_range=None):
    """Create plots similar to the provided image"""
    if date_range:
        df = df[(df.index >= date_range[0]) & (df.index <= date_range[1])]
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [1, 1, 0.6]})
    
    # Plot 1: Actual Price and Base Trend Model
    axs[0].plot(df.index, df['price'], 'b-', label='Actual Price')
    axs[0].plot(df.index, df['trend_prediction'], 'g-', label='Trend Prediction')
    axs[0].set_title('Actual Price and Base Trend Model: ' + 
                   date_range[0].strftime('%Y-%m-%d') + ' to ' + 
                   date_range[1].strftime('%Y-%m-%d'))
    axs[0].set_ylabel('Price (EUR/MWh)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot 2: Combined Model: Trend with Peak & Valley Volatility
    axs[1].plot(df.index, df['price'], 'gray', label='Actual Price')
    axs[1].plot(df.index, df['trend_prediction'], 'g-', label='Trend Prediction')
    axs[1].plot(df.index, df['merged_prediction'], 'm-', label='Merged Prediction')
    
    # Add peak markers
    peak_indices = np.where(predictions['peak'] == 1)[0]
    if len(peak_indices) > 0:
        peak_times = df.index[peak_indices]
        peak_values = df['merged_prediction'].iloc[peak_indices]
        axs[1].scatter(peak_times, peak_values, color='red', marker='^', 
                      s=80, label='Predicted Peaks')
    
    # Add valley markers
    valley_indices = np.where(predictions['valley'] == 1)[0]
    if len(valley_indices) > 0:
        valley_times = df.index[valley_indices]
        valley_values = df['merged_prediction'].iloc[valley_indices]
        axs[1].scatter(valley_times, valley_values, color='blue', marker='v', 
                      s=80, label='Predicted Valleys')
    
    axs[1].set_title('Combined Model: Trend with Peak & Valley Volatility')
    axs[1].set_ylabel('Price (EUR/MWh)')
    axs[1].grid(True)
    axs[1].legend()
    
    # Plot 3: Peak and Valley Probabilities
    axs[2].plot(df.index, predictions['peak_prob'], 'r-', label='Peak Probability')
    axs[2].plot(df.index, predictions['valley_prob'], 'b-', label='Valley Probability')
    
    # Add horizontal line at peak threshold
    axs[2].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Peak Threshold (0.8)')
    
    # Add horizontal line at valley threshold
    axs[2].axhline(y=0.65, color='b', linestyle='--', alpha=0.5, label='Valley Threshold (0.65)')
    
    # Actual peaks/valleys
    axs[2].scatter(df[df['is_peak'] == 1].index, [0.95] * len(df[df['is_peak'] == 1]),
                 color='r', marker='x', label='Actual Peaks')
    axs[2].scatter(df[df['is_valley'] == 1].index, [0.05] * len(df[df['is_valley'] == 1]),
                 color='b', marker='x', label='Actual Valleys')
    
    axs[2].set_title('Peak and Valley Probabilities')
    axs[2].set_ylabel('Probability')
    axs[2].set_ylim(0, 1)
    axs[2].grid(True)
    axs[2].legend()
    
    # Format x-axis dates for all plots
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/merged_model_{date_range[0].strftime('%Y%m%d')}-{date_range[1].strftime('%Y%m%d')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

def main():
    # Set date range for evaluation
    start_date = datetime(2023, 9, 21)
    end_date = datetime(2023, 9, 28)
    
    # Load models
    models = load_models()
    
    # Load data
    df = load_data(start_date, end_date)
    
    # Process data for models
    df = process_data_for_models(df, models)
    
    # Generate predictions
    predictions = generate_predictions(df, models, peak_threshold=0.8, valley_threshold=0.65)
    
    # Merge predictions
    df = merge_predictions(df, predictions)
    
    # Plot results
    plot_results(df, predictions, date_range=(start_date, end_date))

if __name__ == "__main__":
    main()