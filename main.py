import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import load_model
import yfinance as finance_data

# Constants for simulation and model parameters
SEASON_AMP = 5  # Amplitude of seasonal variation
NOISE_LEVEL = 1.0  # Standard deviation of noise
DRIFT_RATE = 0.1  # Rate of drift in the data
VOLATILITY = 1.0  # Volatility for sudden jumps
WINDOW_SIZE = 10  # Size of the sliding window for LSTM
NUM_SAMPLES = 3000  # Number of samples for training
AVG_WINDOW = 60  # Window size for averaging predictions
PLOT_FREQ = 10  # Frequency of plot updates
THRESHOLD_FACTOR = 3.0  # Factor for setting dynamic threshold
TRAIN_TEST_SPLIT = 0.8  # Ratio for splitting training and validation data
EPOCHS = 200  # Number of epochs for model training
BATCH_SIZE = 64  # Batch size for training
TICKER = 'NVDA'  # Stock ticker symbol
PERIOD = '10y'  # Period for fetching historical data
INTERVAL = '1d'  # Interval for fetching historical data
MODEL_PATH = "anomaly_model.keras"  # Path to save the trained model
ANOMALY_IMG_PATH = "/home/sumit/Desktop/data/anomalies_detected.png"  # Path to save the anomaly plot

# Simulates a data stream with seasonal trends, drift, and noise
def simulate_data_stream(avg=100):
    """Generate a data stream with seasonality, drift, and noise."""
    time_step = 0
    curr_value = avg
    while True:
        seasonal_pattern = SEASON_AMP * np.sin(2 * np.pi * time_step / 50)
        drift_effect = DRIFT_RATE * time_step
        noise = np.random.normal(0, NOISE_LEVEL)
        sudden_jump = np.random.normal(0, VOLATILITY) if np.random.rand() < 0.1 else 0
        curr_value += noise + seasonal_pattern + drift_effect + sudden_jump
        yield curr_value
        time_step += 1

# Loads historical stock price data from Yahoo Finance
def get_historical_data(stock=TICKER, period=PERIOD, interval=INTERVAL):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        data = finance_data.download(stock, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data fetched for the given parameters.")
        return data['Close'].values
    except Exception as e:
        print(f"Error fetching data: {e}")
        return np.array([])

# Creates an LSTM autoencoder for anomaly detection
def build_lstm_autoencoder(input_shape):
    """Builds and compiles the LSTM autoencoder model."""
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.compile(optimizer='adam', loss='mse')  # Mean Squared Error loss
    return model

# Prepares data for LSTM training in a sliding window format
def prepare_training_data(data_source, num_samples=NUM_SAMPLES, window_size=WINDOW_SIZE):
    """Prepares the training data in sliding window format."""
    samples = []
    temp_window = []
    for i, value in enumerate(data_source):
        if value is not None:
            temp_window.append([value])
            if len(temp_window) == window_size:
                samples.append(temp_window)
                temp_window = []
            if len(samples) == num_samples:
                break
    return np.array(samples)

# Calculates a dynamic threshold to identify anomalies
def visualize_data_with_anomalies(data_stream, model, window_size=WINDOW_SIZE, plot_freq=PLOT_FREQ, avg_window=AVG_WINDOW):
    """Visualizes the data stream and highlights detected anomalies."""
    sliding_window = deque(maxlen=window_size)
    error_buffer = deque(maxlen=avg_window)
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.ion()

    times, real_values = [], []
    anomaly_times, anomaly_values = [], []
    detected_anomalies = []
    predictions, smooth_preds, dynamic_thresholds = [], [], []
    ongoing_anomaly = None

    for i, value in enumerate(data_stream):
        if value is not None:
            sliding_window.append([value])

            if len(sliding_window) == window_size:
                reshaped_window = np.array(sliding_window).reshape(1, window_size, 1)
                reconstructed_window = model.predict(reshaped_window)
                reconstruction_error = np.mean(np.abs(reshaped_window - reconstructed_window))
                error_buffer.append(reconstruction_error)

                # Update dynamic threshold based on recent errors
                current_threshold = np.mean(error_buffer) + THRESHOLD_FACTOR * np.std(error_buffer)
                dynamic_thresholds.append(current_threshold)

                times.append(i)
                real_values.append(value)
                predictions.append(reconstructed_window[0, -1, 0])

                # Flag anomalies when error exceeds the threshold
                if reconstruction_error > current_threshold:
                    anomaly_times.append(i)
                    anomaly_values.append(value)
                    if ongoing_anomaly is None:
                        ongoing_anomaly = i
                else:
                    if ongoing_anomaly is not None:
                        detected_anomalies.append((ongoing_anomaly, i))
                        ongoing_anomaly = None

                # Smooth predictions
                if len(predictions) >= avg_window:
                    smooth_preds.append(np.mean(predictions[-avg_window:]))
                else:
                    smooth_preds.append(np.mean(predictions))

                # Update plot every few steps
                if i % plot_freq == 0:
                    ax.clear()
                    ax.set_facecolor('#f2f2f2')

                    # Plot data stream with a thicker line
                    ax.plot(times, real_values, color='#1f77b4', linewidth=2, label='Data Stream')
                    ax.plot(times, smooth_preds, color='#ff7f0e', linewidth=1.5, linestyle='-', label='Smoothed Predictions', alpha=0.8)
                    ax.plot(times, dynamic_thresholds, color='#2ca02c', linestyle='--', linewidth=1.5, label='Dynamic Threshold')

                    # Highlight anomaly intervals
                    for start, end in detected_anomalies:
                        ax.axvspan(start, end, color='salmon', alpha=0.3)

                    # Mark individual anomalies with a different marker style
                    ax.scatter(anomaly_times, anomaly_values, color='darkred', marker='o', s=50, label='Anomalies')

                    # Set labels and title with a new font size
                    ax.set_xlabel('Time Steps', fontsize=12)
                    ax.set_ylabel('Value', fontsize=12)
                    ax.set_title('Data Stream with Anomaly Detection', fontsize=14)
                    ax.legend(fontsize=10)

                    # Add grid for better readability
                    ax.grid(True, linestyle='--', alpha=0.7)

                    plt.pause(0.01)

    # Handle any ongoing anomaly
    if ongoing_anomaly is not None:
        detected_anomalies.append((ongoing_anomaly, len(times) - 1))

    plt.savefig(ANOMALY_IMG_PATH)
    plt.ioff()
    plt.show()

# Trains the LSTM autoencoder
def train_autoencoder(input_shape, train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Trains the LSTM autoencoder and returns the trained model."""
    model = build_lstm_autoencoder(input_shape)
    model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, validation_data=(val_data, val_data), shuffle=True)
    return model

# Main workflow to detect anomalies in the data stream
def detect_anomalies(ticker=TICKER, period=PERIOD, interval=INTERVAL):
    """Main function to detect anomalies in stock price data."""
    # Load stock price data
    stock_data = get_historical_data(stock=ticker, period=period, interval=interval)
    if stock_data.size == 0:
        print("No data available to process.")
        return

    data_stream = (p for p in stock_data)

    # Prepare training data
    input_shape = (WINDOW_SIZE, 1)
    train_data = prepare_training_data(data_stream, num_samples=NUM_SAMPLES, window_size=WINDOW_SIZE)

    # Split data into training and validation sets
    split_idx = int(TRAIN_TEST_SPLIT * len(train_data))
    train_set, val_set = train_data[:split_idx], train_data[split_idx:]

    # Train the model
    lstm_model = train_autoencoder(input_shape, train_set, val_set)

    # Save and reload the model
    lstm_model.save(MODEL_PATH)
    lstm_model = load_model(MODEL_PATH)

    # Reuse data stream for real-time anomaly detection
    stock_data = get_historical_data(stock=ticker, period=period, interval=interval)
    if stock_data.size == 0:
        print("No data available for real-time detection.")
        return

    data_stream = (p for p in stock_data)

    # Detect and visualize anomalies
    visualize_data_with_anomalies(data_stream, lstm_model, window_size=WINDOW_SIZE, avg_window=AVG_WINDOW)

if __name__ == '__main__':
    detect_anomalies()
