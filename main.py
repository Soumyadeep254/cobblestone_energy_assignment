#!/usr/bin/env python3
# anomaly_detection.py

import os
import sys
import logging
from collections import deque

import numpy as np
import matplotlib
# Use 'Agg' backend for environments without a display (e.g., servers)
matplotlib.use('TkAgg')  # Change to 'Agg' if you encounter issues
import matplotlib.pyplot as plt

import yfinance as finance_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ------------------------------- Constants ----------------------------------- #

# Simulation and Model Parameters
SEASON_AMP = 5  # Amplitude of seasonal variation
NOISE_LEVEL = 1.0  # Standard deviation of noise
DRIFT_RATE = 0.1  # Rate of drift in the data
VOLATILITY = 1.0  # Volatility for sudden jumps

# Data and Training Parameters
WINDOW_SIZE = 10  # Size of the sliding window for LSTM
NUM_SAMPLES = 3000  # Number of samples for training
AVG_WINDOW = 60  # Window size for averaging predictions
PLOT_FREQ = 10  # Frequency of plot updates
THRESHOLD_FACTOR = 3.0  # Factor for setting dynamic threshold
TRAIN_TEST_SPLIT_RATIO = 0.8  # Ratio for splitting training and validation data
EPOCHS = 200  # Number of epochs for model training
BATCH_SIZE = 64  # Batch size for training

# Stock Data Parameters
TICKER = 'NVDA'  # Stock ticker symbol
PERIOD = '10y'  # Period for fetching historical data
INTERVAL = '1d'  # Interval for fetching historical data

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(_file_))
MODEL_PATH = os.path.join(SCRIPT_DIR, "anomaly_model.keras")  # Path to save the trained model
ANOMALY_IMG_PATH = os.path.join(SCRIPT_DIR, "anomalies_detected.png")  # Path to save the anomaly plot

# ---------------------------- Data Simulation -------------------------------- #

def simulate_data_stream(avg=100):
    """
    Generate a data stream with seasonality, drift, and noise.
    This function is not used in the current implementation but can be
    utilized for synthetic data generation.
    """
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

# -------------------------- Data Acquisition --------------------------------- #

def get_historical_data(stock=TICKER, period=PERIOD, interval=INTERVAL):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
        stock (str): Stock ticker symbol.
        period (str): Data period (e.g., '10y' for 10 years).
        interval (str): Data interval (e.g., '1d' for daily).

    Returns:
        np.ndarray: Array of closing prices.
    """
    try:
        logging.info(f"Fetching data for {stock} over period '{period}' with interval '{interval}'.")
        data = finance_data.download(stock, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data fetched for the given parameters.")
        logging.info(f"Fetched {len(data)} records.")
        return data['Close'].values
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return np.array([])

# ------------------------- Data Preparation ---------------------------------- #

def prepare_training_data(data_source, num_samples=NUM_SAMPLES, window_size=WINDOW_SIZE):
    """
    Prepares the training data in sliding window format with overlapping windows.

    Parameters:
        data_source (iterable): Iterable data source (e.g., list or generator).
        num_samples (int): Number of samples to prepare.
        window_size (int): Size of each sliding window.

    Returns:
        np.ndarray: Array of shape (num_samples, window_size, 1).
    """
    samples = []
    temp_window = deque(maxlen=window_size)
    for value in data_source:
        if value is not None:
            temp_window.append([value])
            if len(temp_window) == window_size:
                samples.append(list(temp_window))
                if len(samples) == num_samples:
                    break
    if len(samples) < num_samples:
        logging.warning(f"Only {len(samples)} samples collected, expected {num_samples}.")
    return np.array(samples)

# ------------------------ Model Construction --------------------------------- #

def build_lstm_autoencoder(input_shape):
    """
    Builds and compiles the LSTM autoencoder model.

    Parameters:
        input_shape (tuple): Shape of the input data (window_size, features).

    Returns:
        tensorflow.keras.Model: Compiled LSTM autoencoder model.
    """
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=False),
        RepeatVector(input_shape[0]),
        LSTM(64, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mse')  # Mean Squared Error loss
    logging.info("LSTM autoencoder model built and compiled.")
    return model

# -------------------------- Model Training ----------------------------------- #

def train_autoencoder(input_shape, train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Trains the LSTM autoencoder and returns the trained model.

    Parameters:
        input_shape (tuple): Shape of the input data (window_size, features).
        train_data (np.ndarray): Training data.
        val_data (np.ndarray): Validation data.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        tensorflow.keras.Model: Trained LSTM autoencoder model.
    """
    try:
        model = build_lstm_autoencoder(input_shape)
        logging.info("Starting model training...")
        model.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, val_data),
            shuffle=True,
            verbose=1
        )
        logging.info("Model training completed.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None

# ----------------------- Anomaly Detection & Visualization ------------------ #

def visualize_data_with_anomalies(data_stream, model, scaler, window_size=WINDOW_SIZE, plot_freq=PLOT_FREQ, avg_window=AVG_WINDOW):
    """
    Visualizes the data stream and highlights detected anomalies.

    Parameters:
        data_stream (iterable): Iterable data source for detection.
        model (tensorflow.keras.Model): Trained LSTM autoencoder model.
        scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler for inverse transformation.
        window_size (int): Size of the sliding window.
        plot_freq (int): Frequency of plot updates.
        avg_window (int): Window size for averaging predictions.
    """
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
                reconstructed_window = model.predict(reshaped_window, verbose=0)
                reconstruction_error = np.mean(np.abs(reshaped_window - reconstructed_window))
                error_buffer.append(reconstruction_error)

                # Update dynamic threshold based on recent errors
                current_threshold = np.mean(error_buffer) + THRESHOLD_FACTOR * np.std(error_buffer)
                dynamic_thresholds.append(current_threshold)

                times.append(i)
                real_values.append(scaler.inverse_transform([[value]])[0][0])
                predictions.append(scaler.inverse_transform([[reconstructed_window[0, -1, 0]]])[0][0])

                # Flag anomalies when error exceeds the threshold
                if reconstruction_error > current_threshold:
                    anomaly_times.append(i)
                    anomaly_values.append(scaler.inverse_transform([[value]])[0][0])
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

    # Final plot update
    ax.clear()
    ax.set_facecolor('#f2f2f2')

    ax.plot(times, real_values, color='#1f77b4', linewidth=2, label='Data Stream')
    ax.plot(times, smooth_preds, color='#ff7f0e', linewidth=1.5, linestyle='-', label='Smoothed Predictions', alpha=0.8)
    ax.plot(times, dynamic_thresholds, color='#2ca02c', linestyle='--', linewidth=1.5, label='Dynamic Threshold')

    for start, end in detected_anomalies:
        ax.axvspan(start, end, color='salmon', alpha=0.3)

    ax.scatter(anomaly_times, anomaly_values, color='darkred', marker='o', s=50, label='Anomalies')

    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Data Stream with Anomaly Detection', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(ANOMALY_IMG_PATH)
    plt.ioff()
    plt.show()
    logging.info(f"Anomaly plot saved to {ANOMALY_IMG_PATH}.")

# --------------------------- Main Workflow ----------------------------------- #

def detect_anomalies(ticker=TICKER, period=PERIOD, interval=INTERVAL):
    """
    Main function to detect anomalies in stock price data.

    Parameters:
        ticker (str): Stock ticker symbol.
        period (str): Data period (e.g., '10y' for 10 years).
        interval (str): Data interval (e.g., '1d' for daily).
    """
    # Load stock price data
    stock_data = get_historical_data(stock=ticker, period=period, interval=interval)
    if stock_data.size == 0:
        logging.error("No data available to process.")
        return

    # Normalize data
    scaler = MinMaxScaler()
    stock_data_scaled = scaler.fit_transform(stock_data.reshape(-1, 1)).flatten()
    logging.info("Data normalization complete.")

    # Split data into training and testing sets
    train_data, test_data = train_test_split(
        stock_data_scaled,
        train_size=TRAIN_TEST_SPLIT_RATIO,
        shuffle=False
    )
    logging.info(f"Data split into {len(train_data)} training samples and {len(test_data)} testing samples.")

    # Prepare training data
    train_samples = prepare_training_data(iter(train_data), num_samples=NUM_SAMPLES, window_size=WINDOW_SIZE)
    if len(train_samples) == 0:
        logging.error("Insufficient training data.")
        return
    train_set = np.array(train_samples)
    logging.info(f"Prepared {len(train_set)} training samples.")

    # Prepare validation data
    val_samples = prepare_training_data(iter(train_data), num_samples=int(NUM_SAMPLES * 0.1), window_size=WINDOW_SIZE)
    val_set = np.array(val_samples)
    logging.info(f"Prepared {len(val_set)} validation samples.")

    # Define input shape
    input_shape = (WINDOW_SIZE, 1)

    # Train the model
    lstm_model = train_autoencoder(input_shape, train_set, val_set)
    if lstm_model is None:
        logging.error("Model training failed.")
        return

    # Save the model
    lstm_model.save(MODEL_PATH)
    logging.info(f"Trained model saved to {MODEL_PATH}.")

    # Reload the model (optional, can be skipped)
    # lstm_model = load_model(MODEL_PATH)

    # Prepare test data stream
    test_data_stream = iter(test_data)
    logging.info("Starting anomaly detection on test data.")

    # Detect and visualize anomalies
    visualize_data_with_anomalies(
        data_stream=test_data_stream,
        model=lstm_model,
        scaler=scaler,
        window_size=WINDOW_SIZE,
        avg_window=AVG_WINDOW,
        plot_freq=PLOT_FREQ
    )

# ------------------------------- Entry Point ---------------------------------- #

if _name_ == '_main_':
    try:
        detect_anomalies()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        logging.info("Anomaly detection script has completed.")
