#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import logging
import time
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.reservoir_builder import ReservoirBuilder
from ml.training_manager import TrainingManager
from ml.autonomous_runner import AutonomousRunner


def setup_logging(log_dir='logs', level=logging.INFO):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'ecg_anomaly_detection.log')
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('ecg_anomaly_detection')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='ECG Anomaly Detection with Reservoir Computing')
    parser.add_argument('--normal_data', type=str, default='data/ECG Heartbeat Categorization Dataset/ptbdb_normal.csv',
                        help='Path to normal ECG data')
    parser.add_argument('--abnormal_data', type=str, default='data/ECG Heartbeat Categorization Dataset/ptbdb_abnormal.csv',
                        help='Path to abnormal ECG data')
    parser.add_argument('--results_dir', type=str, default='results/ecg_anomaly',
                        help='Directory to save results')
    parser.add_argument('--train_samples', type=int, default=200,
                        help='Number of normal samples to use for training')
    parser.add_argument('--test_samples', type=int, default=100,
                        help='Number of samples to use for testing (both normal and abnormal)')
    parser.add_argument('--reservoir_size', type=int, default=500,
                        help='Size of the reservoir (number of neurons)')
    parser.add_argument('--spectral_radius', type=float, default=0.9,
                        help='Spectral radius of the reservoir')
    parser.add_argument('--input_scaling', type=float, default=0.5,
                        help='Scaling of input weights')
    parser.add_argument('--ridge_beta', type=float, default=1e-6,
                        help='Regularization parameter for ridge regression')
    parser.add_argument('--debug_level', choices=['DEBUG', 'INFO', 'WARNING'], default='INFO',
                        help='Logging level')
    
    return parser.parse_args()


def load_ecg_data(file_path, num_samples=None):
    """
    Load ECG data from CSV file.
    
    Args:
        file_path: Path to CSV file
        num_samples: Number of samples to load (None to load all)
        
    Returns:
        Array of heartbeat samples [samples, features]
    """
    data = np.loadtxt(file_path, delimiter=',', max_rows=num_samples)
    
    # The last column in each row could be a label, if so, remove it
    if data.shape[1] > 188 and np.all(data[:, -1] == 1.0):
        data = data[:, :-1]
        
    return data


def prepare_data(normal_data, abnormal_data, train_size, test_size):
    """
    Prepare data for training and testing.
    
    Args:
        normal_data: Array of normal heartbeats
        abnormal_data: Array of abnormal heartbeats
        train_size: Number of normal samples for training
        test_size: Number of samples for testing from each class
        
    Returns:
        Dictionary with train and test data and labels
    """
    # Ensure we have enough data
    train_size = min(train_size, normal_data.shape[0] - test_size)
    test_size = min(test_size, min(normal_data.shape[0] - train_size, abnormal_data.shape[0]))
    
    # Scale the data to [0, 1]
    scaler = MinMaxScaler()
    all_data = np.vstack([normal_data, abnormal_data])
    scaler.fit(all_data)
    
    normal_data_scaled = scaler.transform(normal_data)
    abnormal_data_scaled = scaler.transform(abnormal_data)
    
    # Split the data
    train_data = normal_data_scaled[:train_size]
    test_normal_data = normal_data_scaled[train_size:train_size + test_size]
    test_abnormal_data = abnormal_data_scaled[:test_size]
    
    # Create test data and labels
    test_data = np.vstack([test_normal_data, test_abnormal_data])
    test_labels = np.hstack([np.zeros(test_size), np.ones(test_size)])
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'test_labels': test_labels,
        'test_normal': test_normal_data,
        'test_abnormal': test_abnormal_data
    }


def train_reservoir(train_data, reservoir_size, spectral_radius, input_scaling, ridge_beta, logger):
    """
    Train the reservoir on normal data.
    
    Args:
        train_data: Training data array [samples, features]
        reservoir_size: Size of the reservoir
        spectral_radius: Spectral radius of the reservoir
        input_scaling: Scaling of input weights
        ridge_beta: Regularization parameter for ridge regression
        logger: Logger instance
        
    Returns:
        Trained reservoir components (Win, A, Wout)
    """
    # Data dimension
    data_dim = train_data.shape[1]
    
    # Initialize the reservoir builder
    logger.info(f"Initializing reservoir (size={reservoir_size}, radius={spectral_radius})")
    reservoir_builder = ReservoirBuilder(
        Dr=reservoir_size,
        D=data_dim,
        rho=spectral_radius,
        degree=3,  # Sparse connectivity
        sigma_input=input_scaling
    )
    
    # Build the input and reservoir matrices
    logger.info("Building input matrix Win...")
    Win = reservoir_builder.build_input_matrix()
    
    logger.info("Building reservoir matrix A...")
    A = reservoir_builder.build_reservoir_matrix()
    
    # Prepare training data (using each sample as input and target)
    U = train_data.copy()  # Input
    Yd = train_data.copy()  # Target (for auto-encoding / reconstruction)
    
    # Train the reservoir
    logger.info("Training the reservoir...")
    training_manager = TrainingManager(
        Win=Win,
        A=A,
        washout_duration=0,  # No washout for ECG samples
        ridge_beta=ridge_beta
    )
    
    # Train and get output weights
    Wout = training_manager.train(U, Yd)
    final_r = training_manager.r
    
    logger.info(f"Training complete, Wout shape: {Wout.shape}")
    
    return {
        'Win': Win,
        'A': A,
        'Wout': Wout,
        'final_r': final_r
    }


def calculate_reconstruction_errors(test_data, model, logger):
    """
    Calculate reconstruction errors for test data.
    
    Args:
        test_data: Test data array [samples, features]
        model: Trained reservoir model (Win, A, Wout)
        logger: Logger instance
        
    Returns:
        Array of reconstruction errors for each sample
    """
    # Initialize the autonomous runner
    autonomous_runner = AutonomousRunner(
        Win=model['Win'],
        A=model['A'],
        Wout=model['Wout']
    )
    
    # Calculate reconstruction error for each sample
    errors = []
    
    logger.info(f"Calculating reconstruction errors for {len(test_data)} samples...")
    for i, sample in enumerate(test_data):
        # Reshape to 2D for the autonomous runner
        sample_2d = sample.reshape(1, -1)
        
        # Run the sample through the reservoir (just once, no autonomous prediction)
        # We'll initialize with zeros since each sample is independent
        r = np.zeros(model['Win'].shape[0])
        
        # Drive the reservoir with the input
        # Convert sparse matrix to dense for the calculation if needed
        A_r = model['A'].dot(r)  # Sparse matrix multiplication with vector
        Win_sample = model['Win'].dot(sample)  # Sparse matrix multiplication with vector
        r = np.tanh(A_r + Win_sample)  # Now adding two vectors, not sparse + vector
        
        # Generate output from reservoir state
        output = np.dot(model['Wout'], r)
        
        # Calculate error (MSE)
        mse = np.mean((output - sample) ** 2)
        errors.append(mse)
        
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(test_data)} samples")
    
    return np.array(errors)


def find_optimal_threshold(errors_normal, errors_abnormal):
    """
    Find the optimal threshold for anomaly detection.
    
    Args:
        errors_normal: Reconstruction errors for normal samples
        errors_abnormal: Reconstruction errors for abnormal samples
        
    Returns:
        Optimal threshold and corresponding metrics
    """
    # Create labels (0 for normal, 1 for abnormal)
    y_true = np.hstack([np.zeros(len(errors_normal)), np.ones(len(errors_abnormal))])
    
    # Combine errors
    errors_all = np.hstack([errors_normal, errors_abnormal])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, errors_all)
    roc_auc = auc(fpr, tpr)
    
    # Find the optimal threshold (maximize TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'optimal_idx': optimal_idx
    }


def evaluate_results(errors_normal, errors_abnormal, threshold):
    """
    Evaluate detection results based on the threshold.
    
    Args:
        errors_normal: Reconstruction errors for normal samples
        errors_abnormal: Reconstruction errors for abnormal samples
        threshold: Threshold for anomaly detection
        
    Returns:
        Dictionary of metrics
    """
    # Apply threshold
    predictions_normal = (errors_normal > threshold).astype(int)
    predictions_abnormal = (errors_abnormal > threshold).astype(int)
    
    # Calculate metrics
    true_negative = np.sum(predictions_normal == 0)
    false_positive = np.sum(predictions_normal == 1)
    true_positive = np.sum(predictions_abnormal == 1)
    false_negative = np.sum(predictions_abnormal == 0)
    
    # Compute metrics
    accuracy = (true_positive + true_negative) / (len(errors_normal) + len(errors_abnormal))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positive': true_positive,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative
    }


def plot_results(errors_normal, errors_abnormal, threshold, roc_data, results_dir):
    """
    Create and save plots of results.
    
    Args:
        errors_normal: Reconstruction errors for normal samples
        errors_abnormal: Reconstruction errors for abnormal samples
        threshold: Threshold for anomaly detection
        roc_data: ROC curve data
        results_dir: Directory to save plots
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot histogram of errors
    plt.figure(figsize=(10, 6))
    plt.hist(errors_normal, bins=30, alpha=0.5, label='Normal')
    plt.hist(errors_abnormal, bins=30, alpha=0.5, label='Abnormal')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'error_distribution.png'), dpi=300)
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(roc_data['fpr'], roc_data['tpr'], 'b-', label=f'ROC (AUC = {roc_data["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(roc_data['fpr'][roc_data['optimal_idx']], roc_data['tpr'][roc_data['optimal_idx']], 'ro', 
             label=f'Optimal Threshold ({threshold:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # Plot example reconstructions
    plt.figure(figsize=(12, 8))
    
    # Select examples with different reconstruction errors
    normal_indices = np.argsort(errors_normal)
    abnormal_indices = np.argsort(errors_abnormal)
    
    examples = [
        ('Normal (Low Error)', normal_indices[0], errors_normal[normal_indices[0]]),
        ('Normal (High Error)', normal_indices[-1], errors_normal[normal_indices[-1]]),
        ('Abnormal (Low Error)', abnormal_indices[0], errors_abnormal[abnormal_indices[0]]),
        ('Abnormal (High Error)', abnormal_indices[-1], errors_abnormal[abnormal_indices[-1]])
    ]
    
    # TODO: Add code to plot example reconstructions if you save them during evaluation
    # This would require saving the actual reconstructed signals


def main():
    """Main function to orchestrate the anomaly detection workflow."""
    start_time = time.time()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.debug_level)
    logger = setup_logging(level=log_level)
    logger.info("Starting ECG Anomaly Detection with Reservoir Computing")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    logger.info(f"Results will be saved to {args.results_dir}")
    
    # Load data
    logger.info(f"Loading normal ECG data from {args.normal_data}")
    normal_data = load_ecg_data(args.normal_data, args.train_samples + args.test_samples)
    logger.info(f"Loaded {len(normal_data)} normal samples")
    
    logger.info(f"Loading abnormal ECG data from {args.abnormal_data}")
    abnormal_data = load_ecg_data(args.abnormal_data, args.test_samples)
    logger.info(f"Loaded {len(abnormal_data)} abnormal samples")
    
    # Prepare data for training and testing
    logger.info("Preparing data for training and testing...")
    data = prepare_data(
        normal_data, 
        abnormal_data, 
        args.train_samples, 
        args.test_samples
    )
    
    logger.info(f"Train data shape: {data['train_data'].shape}")
    logger.info(f"Test data shape: {data['test_data'].shape}")
    
    # Train the reservoir
    model = train_reservoir(
        data['train_data'],
        args.reservoir_size,
        args.spectral_radius,
        args.input_scaling,
        args.ridge_beta,
        logger
    )
    
    # Calculate reconstruction errors
    errors_normal = calculate_reconstruction_errors(data['test_normal'], model, logger)
    errors_abnormal = calculate_reconstruction_errors(data['test_abnormal'], model, logger)
    
    logger.info(f"Normal errors: mean={np.mean(errors_normal):.6f}, std={np.std(errors_normal):.6f}")
    logger.info(f"Abnormal errors: mean={np.mean(errors_abnormal):.6f}, std={np.std(errors_abnormal):.6f}")
    
    # Find optimal threshold
    roc_data = find_optimal_threshold(errors_normal, errors_abnormal)
    threshold = roc_data['threshold']
    logger.info(f"Optimal threshold: {threshold:.6f}, AUC: {roc_data['auc']:.4f}")
    
    # Evaluate results
    metrics = evaluate_results(errors_normal, errors_abnormal, threshold)
    logger.info(f"Results: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # Create and save plots
    plot_results(errors_normal, errors_abnormal, threshold, roc_data, args.results_dir)
    
    # Save metrics to CSV
    metrics_file = os.path.join(args.results_dir, 'metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Reservoir Size,{args.reservoir_size}\n")
        f.write(f"Spectral Radius,{args.spectral_radius}\n")
        f.write(f"Input Scaling,{args.input_scaling}\n")
        f.write(f"Ridge Beta,{args.ridge_beta}\n")
        f.write(f"Threshold,{threshold}\n")
        f.write(f"AUC,{roc_data['auc']}\n")
        f.write(f"Accuracy,{metrics['accuracy']}\n")
        f.write(f"Precision,{metrics['precision']}\n")
        f.write(f"Recall,{metrics['recall']}\n")
        f.write(f"F1,{metrics['f1']}\n")
        f.write(f"True Positive,{metrics['true_positive']}\n")
        f.write(f"True Negative,{metrics['true_negative']}\n")
        f.write(f"False Positive,{metrics['false_positive']}\n")
        f.write(f"False Negative,{metrics['false_negative']}\n")
    
    # Save errors to file for further analysis
    np.savez(os.path.join(args.results_dir, 'errors.npz'),
             normal=errors_normal,
             abnormal=errors_abnormal)
    
    end_time = time.time()
    logger.info(f"ECG Anomaly Detection completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 