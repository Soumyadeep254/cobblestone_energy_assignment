# Efficient Data Stream Anomaly Detection

## Introduction

This repository contains a Python-based solution for detecting anomalies in a continuous data stream. It simulates a real-time data stream with seasonal variations, random noise, and drift, and uses an LSTM Autoencoder to detect anomalies. The project is specifically designed to handle real-time financial data, but can be adapted to other use cases requiring anomaly detection.

## Features

- **Data Stream Simulation**: Generates a continuous data stream with seasonality, drift, and noise.
- **LSTM Autoencoder**: Utilizes an LSTM-based autoencoder to detect anomalies based on reconstruction error.
- **Real-Time Anomaly Detection**: Detects and flags anomalies as they occur in the data stream.
- **Dynamic Threshold**: Automatically adjusts the threshold based on recent reconstruction errors to account for concept drift and seasonal variations.
- **Visualization**: Provides a real-time visualization of the data stream and detected anomalies.

## Getting Started

### Prerequisites

You will need the following Python packages to run the project:
- `numpy`
- `matplotlib`
- `tensorflow`
- `yfinance`

You can install the required packages by running:
```bash
pip install -r requirements.txt
