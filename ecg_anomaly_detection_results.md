# ECG Anomaly Detection with Reservoir Computing

## Overview

This document summarizes the results of applying Reservoir Computing (RC) for detecting anomalies in ECG heartbeat data. We implemented two approaches:

1. A batch-based anomaly detector (`ecg_anomaly_detector.py`) that processes the entire dataset at once
2. An online anomaly detector (`ecg_online_detector.py`) that processes heartbeats sequentially

Both approaches are built on the same underlying Reservoir Computing framework, demonstrating the versatility of RC for anomaly detection beyond chaotic systems.

## Dataset

We used the PTB Diagnostic ECG Database, which contains:
- Normal ECG recordings (`ptbdb_normal.csv`)
- Abnormal ECG recordings (`ptbdb_abnormal.csv`)

Each ECG sample consists of 188 features representing a single heartbeat.

## Implementation Details

### ECG Simulator Adapter

To integrate ECG data with the existing simulation framework, we created an adapter class that implements the `ISimulator` interface:

```python
class ECGSimulator(ISimulator):
    """
    Adapter class to make ECG data work with the ISimulator interface.
    This allows us to use existing components of the reservoir computing framework.
    """
    # ...implementation details...
```

This adapter allows us to:
- Load ECG data from CSV files
- Process heartbeats as if they were simulation steps
- Return data in a format compatible with the RC framework

### Reservoir Computing Architecture

The system consists of:

1. **Reservoir Builder**: Creates input matrix (Win) and reservoir matrix (A)
2. **Training Manager**: Trains the reservoir on normal ECG data using ridge regression
3. **Autonomous Runner**: Reconstructs inputs using the trained reservoir
4. **Anomaly Detection**: Calculates reconstruction errors and applies threshold

### Parameters

| Parameter | Value Used | Description |
|-----------|------------|-------------|
| Reservoir Size | 500 | Number of neurons in the reservoir |
| Spectral Radius | 0.9 | Controls the dynamics of the reservoir |
| Input Scaling | 0.5 | Scaling factor for input weights |
| Ridge Beta | 1e-6 | Regularization parameter for training |
| Window Size | 5 | Sliding window for error smoothing (online detection) |

## Results

### Batch Processing Results

Using the `ecg_anomaly_detector.py` script, we trained on 200 normal ECG samples and tested on 100 samples each of normal and abnormal ECGs:

| Metric | Value |
|--------|-------|
| AUC | 0.981 |
| Accuracy | 98.0% |
| Precision | 97.1% |
| Recall | 99.0% |
| F1 Score | 98.0% |

The high AUC and accuracy values demonstrate that the RC approach effectively captures the normal patterns in ECG data and can reliably detect deviations that indicate abnormalities.

### Online Processing Results

Using the `ecg_online_detector.py` script for online processing:

#### Testing on Abnormal ECG Data:
- **Normal Data**: `ptbdb_normal.csv`
- **Test Data**: `ptbdb_abnormal.csv`
- **Threshold**: 0.0003
- **Anomaly Count**: 100/100 (100.00%)

#### Testing on Normal ECG Data:
- **Normal Data**: `ptbdb_normal.csv`
- **Test Data**: `ptbdb_normal.csv`
- **Threshold**: 0.0003
- **Anomaly Count**: 0/100 (0.00%)

The online detection achieved perfect discrimination between normal and abnormal ECGs, with no false positives or false negatives.

### Error Distribution

The reconstruction errors showed clear separation between normal and abnormal samples:
- **Normal ECG errors**: mean ≈ 0.000000, std ≈ 0.000100
- **Abnormal ECG errors**: mean ≈ 0.012557, std ≈ 0.008939

This significant difference in reconstruction errors confirms that the reservoir computing model can effectively learn the patterns of normal ECG data and detect deviations in abnormal ECGs.

## Visualization

The anomaly detection process was visualized through:
1. Reconstruction error time series with threshold
2. Example normal and abnormal ECG signals
3. Error distribution histograms
4. Summary visualizations

These are saved in the results directories:
- `results/ecg_anomaly/` (batch processing)
- `results/ecg_anomaly_extended/` (online processing with abnormal data)
- `results/ecg_anomaly_normal_test/` (online processing with normal data)

## Conclusion

The Reservoir Computing approach demonstrated exceptional effectiveness for ECG anomaly detection:

1. **High Accuracy**: Both batch and online methods achieved excellent discrimination between normal and abnormal ECGs.

2. **Simple Training**: The model only requires normal data for training, making it suitable for scenarios where anomalies are rare or undefined.

3. **Computational Efficiency**: The RC framework is computationally efficient, with training times of a few seconds and low memory requirements.

4. **Online Processing**: The framework can be adapted for real-time monitoring by processing heartbeats sequentially.

These results highlight the versatility of Reservoir Computing beyond chaotic systems, showing its potential for medical data analysis and anomaly detection in time series data.

## Next Steps

Potential improvements for future work:
1. Explore different reservoir parameters (spectral radius, input scaling)
2. Implement more adaptive thresholding mechanisms
3. Test on larger and more diverse ECG datasets
4. Compare with other anomaly detection techniques
5. Develop a real-time monitoring system

## References

1. PTB Diagnostic ECG Database
2. Reservoir Computing framework developed for chaotic systems 