# Real-Time Anomaly Detection in Data Streams

## Project Description

This project focuses on developing a Python script capable of detecting anomalies in a continuous data stream. The data stream simulates real-time sequences of floating-point numbers, which could represent metrics such as financial transactions or system performance metrics. The primary objective is to identify unusual patterns such as spikes, deviations from the norm, and seasonal drifts, all in real-time.
## Project Demo Link:
https://drive.google.com/file/d/1Oq5LhGarHTHxtsWLOHpXNhPifbm5ClPe/view?usp=drive_link

The key features of the project include:

- **Algorithm Selection**: A suitable machine learning algorithm (LSTM Autoencoder) was selected and implemented for real-time anomaly detection. This algorithm is capable of adapting to concept drift and handling seasonal variations in the data.
- **Data Stream Simulation**: A function was created to simulate a data stream, incorporating seasonal trends, drift, and random noise.
- **Anomaly Detection**: The system flags anomalies in real time using a dynamic thresholding mechanism based on reconstruction error from the autoencoder.
- **Optimization**: The script is designed for efficiency, ensuring fast detection in continuous data streams.
- **Visualization**: A real-time visualization tool was developed using `matplotlib` to display both the data stream and any detected anomalies.

## Objectives

1. **Algorithm Selection**:
   - Implement an LSTM Autoencoder for detecting anomalies in time-series data.
   - Adapt to changes in the data stream over time, handling seasonal and noisy elements.

2. **Data Stream Simulation**:
   - Design a function to generate a data stream with seasonal variations, drift, and random noise.
   
3. **Anomaly Detection**:
   - Detect anomalies in real time using a sliding window approach and dynamic threshold based on recent errors.

4. **Optimization**:
   - Optimize for speed and resource efficiency to ensure real-time anomaly detection.

5. **Visualization**:
   - Develop a real-time visualization tool using `matplotlib` to display the data stream, predicted values, dynamic threshold, and flagged anomalies.

## Requirements

- **Python Version**: Python 3.x
- **Dependencies**: All necessary libraries are listed in `requirements.txt`. The main dependencies are `numpy`, `matplotlib`, `tensorflow`, and `yfinance`.
- **Documentation**: Code is thoroughly documented, with comments to explain key sections of the implementation.
- **Algorithm Explanation**: The chosen algorithm (LSTM Autoencoder) is designed to detect patterns in time-series data by reconstructing the input and identifying anomalies where reconstruction error is abnormally high.
- **Error Handling**: The code includes robust error handling and data validation to ensure proper functioning even with missing or invalid data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Soumyadeep254/cobblestone_energy_assignment
   pip install -r requirements.txt
   python main.py
   Visualization
The real-time data stream is plotted using matplotlib. Detected anomalies are highlighted and dynamically updated on the plot. The visualization includes:

Data Stream: The main time-series data.
Predictions: Smoothed predictions from the LSTM autoencoder.
Dynamic Threshold: A line showing the dynamically adjusted threshold for detecting anomalies.
Anomalies: Marked with red dots when the reconstruction error exceeds the threshold.

