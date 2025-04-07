# Reservoir Computing for Chaotic Systems and Anomaly Detection

<div align="center">
  <img src="results/ecg_anomaly_extended/summary.png" alt="Reservoir Computing Results" width="800"/>
</div>

## 📋 Overview

This project implements a state-of-the-art **Reservoir Computing (RC)** framework for modeling complex dynamical systems and detecting anomalies in time series data. The system demonstrates exceptional performance in:

1. **Predicting chaotic attractors** (like the Lorenz system)
2. **Calculating Lyapunov exponents** to quantify chaos
3. **Detecting anomalies** in ECG heartbeat patterns

Reservoir Computing offers a powerful approach to modeling nonlinear dynamical systems with significantly lower computational requirements than deep learning alternatives while maintaining impressive accuracy.

## ✨ Key Features

- **Modular architecture**: Decoupled components enable flexible experimentation and reuse
- **High configurability**: YAML-based configuration system for easy hyperparameter tuning
- **Reproducibility**: Consistent random seed handling and configuration management
- **Multiple applications**: Chaotic system modeling and anomaly detection in a single framework
- **Real-time processing**: Online anomaly detection capabilities for streaming data
- **Comprehensive visualization**: Detailed plots and metrics for analysis

## 🧪 Applications

### 1. Chaotic System Prediction

<div align="center">
  <p><i>Accurately predicts the behavior of chaotic systems like the Lorenz attractor</i></p>
</div>

The system can:
- Generate future trajectories of chaotic systems
- Replicate long-term statistical properties (climate)
- Calculate Lyapunov exponents to quantify chaos

### 2. ECG Anomaly Detection

<div align="center">
  <p><i>Detects abnormal heartbeat patterns with extremely high accuracy</i></p>
</div>

Two approaches implemented:
- **Batch processing**: 98% accuracy, 0.981 AUC
- **Online detection**: Perfect separation of normal and abnormal heartbeats

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/reservoir-computing.git
cd reservoir-computing

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data logs results
```

## 📁 Project Structure

```
├── data/                 # Data storage
│   └── ECG Heartbeat Categorization Dataset/  # ECG data
├── logs/                 # Log files 
├── results/              # Results and visualizations
├── src/                  # Source code
│   ├── ml/               # Machine learning components
│   │   ├── reservoir_builder.py        # Constructs reservoir matrices
│   │   ├── training_manager.py         # Handles model training
│   │   ├── autonomous_runner.py        # Generates autonomous predictions
│   │   ├── evaluation_manager.py       # Evaluates performance & Lyapunov exponents
│   │   ├── model_persistence.py        # Saves/loads model components
│   │   ├── main_ml.py                  # Main script for chaotic systems
│   │   ├── ecg_anomaly_detector.py     # Batch ECG anomaly detection
│   │   └── ecg_online_detector.py      # Online ECG anomaly detection
│   └── simulators/       # System simulators
│       ├── base.py                     # Base simulator interface
│       ├── lorenz.py                   # Lorenz system simulator
│       └── ecg_simulator.py            # ECG data simulator adapter
└── ecg_anomaly_detection_results.md    # Detailed ECG analysis results
```

## 🚀 Usage Examples

### Chaotic System Prediction

```bash
# Run the reservoir computing model on the Lorenz system
python src/ml/main_ml.py --config src/ml/config_example.yaml

# Run with debug logging and skip Lyapunov calculation
python src/ml/main_ml.py --config src/ml/config_example.yaml --debug_level=DEBUG --skip_lyapunov
```

### ECG Anomaly Detection

```bash
# Batch anomaly detection
python src/ml/ecg_anomaly_detector.py --train_samples 200 --test_samples 100 --reservoir_size 500

# Online anomaly detection
python src/ml/ecg_online_detector.py --train_samples 200 --test_samples 100 --window_size 5
```

## 📊 Results

### Chaotic Systems

The system achieves remarkable prediction accuracy for the Lorenz system:

- **Short-term prediction**: Accurate for ~8-10 Lyapunov times
- **Long-term statistical properties**: Successfully reproduces the attractor's climate
- **Lyapunov spectrum**: Correctly identifies the positive exponent (λ₁ ≈ 0.01) that confirms chaos

### ECG Anomaly Detection

<div align="center">
  <table>
    <tr>
      <td><b>Batch Processing</b></td>
      <td><b>Online Detection</b></td>
    </tr>
    <tr>
      <td>
        <ul>
          <li>AUC: 0.981</li>
          <li>Accuracy: 98.0%</li>
          <li>Precision: 97.1%</li>
          <li>Recall: 99.0%</li>
          <li>F1 Score: 98.0%</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>100% detection of abnormal ECGs</li>
          <li>0% false positives on normal ECGs</li>
          <li>Clear separation in reconstruction errors</li>
          <li>Real-time processing capability</li>
        </ul>
      </td>
    </tr>
  </table>
</div>

The error distribution shows exceptional separation:
- **Normal ECG errors**: mean ≈ 0.000000, std ≈ 0.000100
- **Abnormal ECG errors**: mean ≈ 0.012557, std ≈ 0.008939

## 🧠 Technical Implementation

### Reservoir Computing Architecture

The system implements the Echo State Network variant of Reservoir Computing with:

1. **Random, sparse reservoir**: Fixed recurrent connections that create complex dynamics
2. **Input layer**: Projects input signals into the high-dimensional reservoir space
3. **Output layer**: Linear readout trained with ridge regression
4. **Hyperparameters**: Spectral radius, input scaling, reservoir size, sparsity

### ECG Simulator Adapter

A custom adapter follows the simulator interface pattern to integrate ECG data with the RC framework:

```python
class ECGSimulator(ISimulator):
    """
    Adapter class to make ECG data work with the ISimulator interface.
    This allows us to use existing components of the reservoir computing framework.
    """
    # Methods: initialize, run_transient, run_record, get_state_dimension
```

## 🔍 Visualizations

The project generates comprehensive visualizations:

1. **Error Distribution**: Histogram showing separation between normal and abnormal ECGs
2. **ROC Curve**: Performance evaluation across different thresholds
3. **Example Signals**: Visual comparison of normal vs abnormal heartbeats
4. **Error Time Series**: Reconstruction errors with threshold indication
5. **Summary Plot**: Combined visualization of key results

## 🔮 Future Work

- Parameter optimization for improved performance
- Integration with real-time monitoring systems
- Extended applications to other time series domains
- Comparison with deep learning approaches
- Hardware implementation for embedded systems

## 📚 References

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
2. Pathak, J., Lu, Z., Hunt, B. R., Girvan, M., & Ott, E. (2017). Using machine learning to replicate chaotic attractors and calculate Lyapunov exponents from data. Chaos, 27(12), 121102. https://doi.org/10.1063/1.5010300
3. PTB Diagnostic ECG Database

---

<div align="center">
  <p><i>Developed with a focus on modularity, configurability, and practical applications in nonlinear dynamics and anomaly detection.</i></p>
</div> 