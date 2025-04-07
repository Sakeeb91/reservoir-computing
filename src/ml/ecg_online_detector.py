#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.reservoir_builder import ReservoirBuilder
from ml.training_manager import TrainingManager
from ml.autonomous_runner import AutonomousRunner
from simulators.ecg_simulator import create_ecg_simulator


def setup_logging(log_dir='logs', level=logging.INFO):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'ecg_online_detection.log')
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('ecg_online_detection')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Online ECG Anomaly Detection with Reservoir Computing')
    parser.add_argument('--normal_data', type=str, default='data/ECG Heartbeat Categorization Dataset/ptbdb_normal.csv',
                        help='Path to normal ECG data')
    parser.add_argument('--test_data', type=str, default='data/ECG Heartbeat Categorization Dataset/ptbdb_abnormal.csv',
                        help='Path to test ECG data (can be normal or abnormal)')
    parser.add_argument('--results_dir', type=str, default='results/ecg_anomaly_extended',
                        help='Directory to save results')
    parser.add_argument('--train_samples', type=int, default=200,
                        help='Number of normal samples to use for training')
    parser.add_argument('--test_samples', type=int, default=100,
                        help='Number of samples to use for testing')
    parser.add_argument('--reservoir_size', type=int, default=500,
                        help='Size of the reservoir (number of neurons)')
    parser.add_argument('--spectral_radius', type=float, default=0.9,
                        help='Spectral radius of the reservoir')
    parser.add_argument('--input_scaling', type=float, default=0.5,
                        help='Scaling of input weights')
    parser.add_argument('--ridge_beta', type=float, default=1e-6,
                        help='Regularization parameter for ridge regression')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Anomaly detection threshold (None to auto-calculate)')
    parser.add_argument('--window_size', type=int, default=10,
                        help='Window size for online detection')
    parser.add_argument('--debug_level', choices=['DEBUG', 'INFO', 'WARNING'], default='INFO',
                        help='Logging level')
    
    return parser.parse_args()


def train_model(normal_data_path, train_samples, reservoir_params, logger):
    """
    Train a reservoir model on normal ECG data.
    
    Args:
        normal_data_path: Path to normal ECG data file
        train_samples: Number of samples to use for training
        reservoir_params: Dictionary of reservoir parameters
        logger: Logger instance
        
    Returns:
        Trained model and scaler
    """
    logger.info(f"Creating simulator for normal data: {normal_data_path}")
    normal_sim = create_ecg_simulator(
        data_path=normal_data_path,
        is_normal=True,
        max_samples=train_samples
    )
    
    # Get data from simulator
    logger.info(f"Getting training data from simulator")
    _, trajectory = normal_sim.run_record(
        duration=train_samples,
        dt_integration=1.0,
        dt_sampling=1.0
    )
    
    # Scale the data
    logger.info("Scaling training data")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(trajectory)
    
    # Initialize reservoir
    logger.info(f"Initializing reservoir (size={reservoir_params['size']}, radius={reservoir_params['spectral_radius']})")
    reservoir_builder = ReservoirBuilder(
        Dr=reservoir_params['size'],
        D=normal_sim.get_state_dimension(),
        rho=reservoir_params['spectral_radius'],
        degree=3,  # Sparse connectivity
        sigma_input=reservoir_params['input_scaling']
    )
    
    # Build matrices
    logger.info("Building reservoir matrices")
    Win = reservoir_builder.build_input_matrix()
    A = reservoir_builder.build_reservoir_matrix()
    
    # Prepare training data (using each sample as input and target)
    U = scaled_data.copy()  # Input
    Yd = scaled_data.copy()  # Target (for auto-encoding / reconstruction)
    
    # Train the reservoir
    logger.info(f"Training the reservoir (ridge_beta={reservoir_params['ridge_beta']})")
    training_manager = TrainingManager(
        Win=Win,
        A=A,
        washout_duration=0,  # No washout for ECG samples
        ridge_beta=reservoir_params['ridge_beta']
    )
    
    # Train and get output weights
    Wout = training_manager.train(U, Yd)
    
    logger.info(f"Training complete, Wout shape: {Wout.shape}")
    
    return {
        'Win': Win,
        'A': A,
        'Wout': Wout,
        'scaler': scaler
    }


def calculate_threshold(model, normal_data_path, num_samples, window_size, logger):
    """
    Calculate a threshold for anomaly detection based on normal data.
    
    Args:
        model: Trained reservoir model
        normal_data_path: Path to normal ECG data file
        num_samples: Number of samples to use
        window_size: Window size for smoothing
        logger: Logger instance
        
    Returns:
        Threshold for anomaly detection
    """
    logger.info(f"Calculating threshold using {num_samples} normal samples")
    
    # Create simulator for threshold calculation
    normal_sim = create_ecg_simulator(
        data_path=normal_data_path,
        is_normal=True,
        max_samples=num_samples
    )
    
    # Get data from simulator
    _, trajectory = normal_sim.run_record(
        duration=num_samples,
        dt_integration=1.0,
        dt_sampling=1.0
    )
    
    # Scale the data
    scaled_data = model['scaler'].transform(trajectory)
    
    # Initialize reservoir state with zeros
    r = np.zeros(model['Win'].shape[0])
    errors = []
    
    # Process each sample to calculate reconstruction errors
    logger.info("Processing normal samples to calculate threshold...")
    for i, sample in enumerate(scaled_data):
        # Update reservoir state
        A_r = model['A'].dot(r)
        Win_sample = model['Win'].dot(sample)
        r = np.tanh(A_r + Win_sample)
        
        # Generate prediction/reconstruction
        output = np.dot(model['Wout'], r)
        
        # Calculate error
        mse = np.mean((output - sample) ** 2)
        errors.append(mse)
    
    # Convert to numpy array
    errors = np.array(errors)
    
    # Remove the first few samples (transient)
    if len(errors) > 10:
        errors = errors[10:]
    
    # Calculate threshold (mean + 3*std)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Make sure we don't have zero standard deviation
    if std_error < 1e-10:
        std_error = 1e-4  # Small default value
        logger.warning(f"Very small standard deviation detected, using default value: {std_error}")
    
    # Calculate threshold
    threshold = mean_error + 3 * std_error
    logger.info(f"Calculated threshold: {threshold:.6f} (mean: {mean_error:.6f}, std: {std_error:.6f})")
    
    return threshold


def online_detection(model, data, window_size, logger, verbose=True):
    """
    Perform online anomaly detection on sequential data.
    
    Args:
        model: Trained reservoir model
        data: Scaled data to analyze
        window_size: Window size for smoothing
        logger: Logger instance
        verbose: Whether to log progress
        
    Returns:
        Array of reconstruction errors for each sample
    """
    if verbose:
        logger.info(f"Starting online detection on {len(data)} samples (window_size={window_size})")
    
    # Initialize the reservoir state with zeros
    r = np.zeros(model['Win'].shape[0])
    
    # Initialize errors array
    errors = np.zeros(len(data))
    smoothed_errors = np.zeros(len(data))
    
    # Process each sample
    for i, sample in enumerate(data):
        # Update reservoir state
        A_r = model['A'].dot(r)
        Win_sample = model['Win'].dot(sample)
        r = np.tanh(A_r + Win_sample)
        
        # Generate prediction/reconstruction
        output = np.dot(model['Wout'], r)
        
        # Calculate error
        mse = np.mean((output - sample) ** 2)
        errors[i] = mse
        
        # Apply smoothing with a sliding window
        start_idx = max(0, i - window_size + 1)
        smoothed_errors[i] = np.mean(errors[start_idx:i+1])
        
        if verbose and (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(data)} samples")
    
    return smoothed_errors


def detect_anomalies(model, test_data_path, num_samples, threshold, window_size, logger):
    """
    Detect anomalies in test data.
    
    Args:
        model: Trained reservoir model
        test_data_path: Path to test ECG data file
        num_samples: Number of samples to analyze
        threshold: Threshold for anomaly detection
        window_size: Window size for smoothing
        logger: Logger instance
        
    Returns:
        Dictionary with detection results
    """
    logger.info(f"Detecting anomalies in {test_data_path}")
    
    # Create simulator for test data
    test_sim = create_ecg_simulator(
        data_path=test_data_path,
        is_normal=False,  # We don't know if it's normal or abnormal
        max_samples=num_samples
    )
    
    # Get data from simulator
    _, trajectory = test_sim.run_record(
        duration=num_samples,
        dt_integration=1.0,
        dt_sampling=1.0
    )
    
    # Scale the data
    scaled_data = model['scaler'].transform(trajectory)
    
    # Run online detection
    errors = online_detection(model, scaled_data, window_size, logger)
    
    # Apply threshold to identify anomalies
    anomalies = errors > threshold
    
    # Count anomalies
    anomaly_count = np.sum(anomalies)
    anomaly_percentage = 100 * anomaly_count / len(anomalies)
    
    logger.info(f"Detected {anomaly_count} anomalies out of {len(anomalies)} samples ({anomaly_percentage:.2f}%)")
    
    return {
        'data': trajectory,
        'scaled_data': scaled_data,
        'errors': errors,
        'anomalies': anomalies,
        'anomaly_count': anomaly_count,
        'anomaly_percentage': anomaly_percentage
    }


def plot_results(results, threshold, window_size, results_dir):
    """
    Create and save plots of detection results.
    
    Args:
        results: Dictionary with detection results
        threshold: Threshold for anomaly detection
        window_size: Window size used for smoothing
        results_dir: Directory to save plots
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract data
    data = results['data']
    errors = results['errors']
    anomalies = results['anomalies']
    
    # 1. Plot reconstruction errors with threshold
    plt.figure(figsize=(14, 6))
    plt.plot(errors, label='Reconstruction Error', color='blue', linewidth=1.5)
    plt.axhline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.6f})')
    
    # Highlight anomalies
    for i in range(len(errors)):
        if anomalies[i]:
            plt.axvspan(i-0.5, i+0.5, color='red', alpha=0.3)
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Smoothed Reconstruction Error', fontsize=12)
    plt.title(f'Online Anomaly Detection (Window Size: {window_size})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'online_detection.png'), dpi=300)
    plt.close()
    
    # 2. Plot example ECG signals
    # Choose a few normal and abnormal examples for visualization
    normal_indices = []
    abnormal_indices = []
    
    for i in range(len(anomalies)):
        if anomalies[i] and len(abnormal_indices) < 3:
            abnormal_indices.append(i)
        elif not anomalies[i] and len(normal_indices) < 3:
            normal_indices.append(i)
            
        if len(normal_indices) == 3 and len(abnormal_indices) == 3:
            break
    
    # If we couldn't find 3 of each, fill with random indices
    while len(normal_indices) < 3 and np.sum(~anomalies) > 0:
        idx = np.random.choice(np.where(~anomalies)[0])
        if idx not in normal_indices:
            normal_indices.append(idx)
            
    while len(abnormal_indices) < 3 and np.sum(anomalies) > 0:
        idx = np.random.choice(np.where(anomalies)[0])
        if idx not in abnormal_indices:
            abnormal_indices.append(idx)
    
    # Plot the examples
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.suptitle('Example ECG Signals', fontsize=16)
    
    # Plot normal examples
    for i, idx in enumerate(normal_indices):
        if i < len(normal_indices):
            ax = axes[0, i]
            ax.plot(data[idx], color='blue', linewidth=1.5)
            ax.set_title(f'Normal Sample #{idx} (Error: {errors[idx]:.6f})')
            ax.grid(True)
            if i == 0:
                ax.set_ylabel('Amplitude', fontsize=12)
    
    # Plot abnormal examples
    for i, idx in enumerate(abnormal_indices):
        if i < len(abnormal_indices):
            ax = axes[1, i]
            ax.plot(data[idx], color='red', linewidth=1.5)
            ax.set_title(f'Abnormal Sample #{idx} (Error: {errors[idx]:.6f})')
            ax.grid(True)
            ax.set_xlabel('Feature Index', fontsize=12)
            if i == 0:
                ax.set_ylabel('Amplitude', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(results_dir, 'example_signals.png'), dpi=300)
    plt.close()
    
    # 3. Plot error distribution with clear threshold indication
    plt.figure(figsize=(12, 6))
    
    # Create histogram with KDE
    hist_bins = min(30, len(np.unique(errors)))
    plt.hist(errors, bins=hist_bins, alpha=0.6, density=True, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    
    # Add vertical line for threshold
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold:.6f})')
    
    # Add text annotation
    plt.text(threshold*1.1, plt.ylim()[1]*0.9, f'Threshold = {threshold:.6f}',
             color='red', fontsize=12, verticalalignment='top')
    
    # Add percentage of anomalies
    anomaly_percent = 100 * np.sum(anomalies) / len(anomalies)
    plt.text(plt.xlim()[1]*0.7, plt.ylim()[1]*0.8, 
             f'Anomalies: {np.sum(anomalies)}/{len(anomalies)} ({anomaly_percent:.1f}%)',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Frequency (Density)', fontsize=12)
    plt.title('Distribution of Reconstruction Errors', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'error_distribution.png'), dpi=300)
    plt.close()
    
    # 4. Create a summary figure for presentation
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Error time series
    plt.subplot(2, 1, 1)
    plt.plot(errors, label='Reconstruction Error', color='blue', linewidth=1.5)
    plt.axhline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.6f})')
    for i in range(len(errors)):
        if anomalies[i]:
            plt.axvspan(i-0.5, i+0.5, color='red', alpha=0.3)
    plt.title('Online Anomaly Detection with Reservoir Computing', fontsize=16)
    plt.ylabel('Reconstruction Error', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Example normal and abnormal signals
    plt.subplot(2, 2, 3)
    if len(normal_indices) > 0:
        plt.plot(data[normal_indices[0]], color='blue', linewidth=1.5)
        plt.title(f'Normal ECG Sample (Error: {errors[normal_indices[0]]:.6f})')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Amplitude', fontsize=12)
    
    plt.subplot(2, 2, 4)
    if len(abnormal_indices) > 0:
        plt.plot(data[abnormal_indices[0]], color='red', linewidth=1.5)
        plt.title(f'Abnormal ECG Sample (Error: {errors[abnormal_indices[0]]:.6f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'summary.png'), dpi=300)
    plt.close()


def main():
    """Main function to orchestrate the online anomaly detection workflow."""
    start_time = time.time()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.debug_level)
    logger = setup_logging(level=log_level)
    logger.info("Starting Online ECG Anomaly Detection with Reservoir Computing")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    logger.info(f"Results will be saved to {args.results_dir}")
    
    # Prepare reservoir parameters
    reservoir_params = {
        'size': args.reservoir_size,
        'spectral_radius': args.spectral_radius,
        'input_scaling': args.input_scaling,
        'ridge_beta': args.ridge_beta
    }
    
    # Train the model
    model = train_model(
        normal_data_path=args.normal_data,
        train_samples=args.train_samples,
        reservoir_params=reservoir_params,
        logger=logger
    )
    
    # Calculate or use provided threshold
    threshold = args.threshold
    if threshold is None:
        threshold = calculate_threshold(
            model=model,
            normal_data_path=args.normal_data,
            num_samples=args.train_samples,
            window_size=args.window_size,
            logger=logger
        )
    
    # Detect anomalies
    results = detect_anomalies(
        model=model,
        test_data_path=args.test_data,
        num_samples=args.test_samples,
        threshold=threshold,
        window_size=args.window_size,
        logger=logger
    )
    
    # Plot and save results
    plot_results(
        results=results,
        threshold=threshold,
        window_size=args.window_size,
        results_dir=args.results_dir
    )
    
    # Save results to file
    np.savez(
        os.path.join(args.results_dir, 'detection_results.npz'),
        errors=results['errors'],
        anomalies=results['anomalies'],
        threshold=threshold
    )
    
    # Save configuration and results summary to CSV
    summary_file = os.path.join(args.results_dir, 'summary.csv')
    with open(summary_file, 'w') as f:
        f.write("Parameter,Value\n")
        f.write(f"Normal Data,{args.normal_data}\n")
        f.write(f"Test Data,{args.test_data}\n")
        f.write(f"Reservoir Size,{args.reservoir_size}\n")
        f.write(f"Spectral Radius,{args.spectral_radius}\n")
        f.write(f"Input Scaling,{args.input_scaling}\n")
        f.write(f"Ridge Beta,{args.ridge_beta}\n")
        f.write(f"Window Size,{args.window_size}\n")
        f.write(f"Threshold,{threshold}\n")
        f.write(f"Anomaly Count,{results['anomaly_count']}\n")
        f.write(f"Total Samples,{len(results['anomalies'])}\n")
        f.write(f"Anomaly Percentage,{results['anomaly_percentage']:.2f}%\n")
    
    end_time = time.time()
    logger.info(f"Online ECG Anomaly Detection completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 