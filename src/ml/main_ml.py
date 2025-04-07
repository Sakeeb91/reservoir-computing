#!/usr/bin/env python
import argparse
import sys
import os
import numpy as np
import logging
import time  # Added for timing

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config_manager import ConfigurationManager
from ml.data_loader import DataLoader
from ml.reservoir_builder import ReservoirBuilder
from ml.training_manager import TrainingManager
from ml.autonomous_runner import AutonomousRunner
from ml.evaluation_manager import EvaluationManager
from ml.model_persistence import ModelPersistence


def setup_logging(log_dir='logs', level=logging.INFO):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'reservoir_computing.log')
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('reservoir_computing')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Reservoir Computing for Chaotic Systems')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    # Allow for direct parameter specification as alternative to config file
    parser.add_argument('--training_data_path', help='Path to training data')
    parser.add_argument('--results_dir', help='Directory to save results')
    
    # Debug flags
    parser.add_argument('--debug_level', choices=['DEBUG', 'INFO', 'WARNING'], default='INFO', 
                        help='Logging level')
    parser.add_argument('--skip_lyapunov', action='store_true', help='Skip Lyapunov exponent calculation')
    
    return parser.parse_args()


def main():
    """Main function to orchestrate the ML workflow."""
    start_time = time.time()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.debug_level)
    logger = setup_logging(level=log_level)
    logger.info("Starting Reservoir Computing system")
    
    # Initialize ConfigurationManager
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config_manager = ConfigurationManager(config_path=args.config)
    else:
        logger.info("Loading configuration from command-line arguments")
        config_manager = ConfigurationManager(args=sys.argv[1:])
    
    config = config_manager.get_config()
    logger.info("Configuration loaded successfully")
    
    # Extract configuration values for easier access
    data_settings = config['data_settings']
    reservoir_params = config['reservoir_hyperparams']
    training_settings = config['training_settings']
    eval_settings = config['evaluation_settings']
    output_settings = config['output_settings']
    
    # Create results directory
    results_dir = output_settings['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to {results_dir}")
    
    # Initialize DataLoader and load data
    logger.info(f"Loading data from {data_settings['training_data_path']}")
    data_loader = DataLoader(data_settings['training_data_path'])
    data, time_points = data_loader.load_data(training_settings['train_length'])
    logger.info(f"Loaded {len(data)} time steps of data, dimension {data.shape[1]}")
    
    # Prepare training data (input/target sequences)
    logger.info("Preparing training data (input/target sequences)...")
    U, Yd = data_loader.prepare_training_data()
    D = data_loader.get_data_dimension()
    logger.info(f"Prepared training data: {len(U)} input-target pairs")
    
    # Initialize ReservoirBuilder and build matrices
    logger.info("Starting reservoir matrix construction...")
    reservoir_builder = ReservoirBuilder(
        Dr=reservoir_params['Dr'],
        D=D,
        rho=reservoir_params['rho'],
        degree=reservoir_params['degree'],
        sigma_input=reservoir_params['sigma_input']
    )
    logger.debug(f"Reservoir size Dr={reservoir_params['Dr']}, spectral radius rho={reservoir_params['rho']}")
    
    logger.info("Building input matrix Win...")
    Win = reservoir_builder.build_input_matrix()
    logger.info(f"Input matrix Win shape: {Win.shape}")
    
    logger.info("Building reservoir matrix A (this may take a moment)...")
    A = reservoir_builder.build_reservoir_matrix()
    logger.info(f"Reservoir matrix A shape: {A.shape}")
    
    logger.info(f"Built input matrix Win ({Win.shape}) and reservoir matrix A ({A.shape})")
    
    # Initialize TrainingManager and train the model
    logger.info("Training the reservoir")
    logger.info("Initializing training manager...")
    training_manager = TrainingManager(
        Win=Win,
        A=A,
        washout_duration=data_settings['washout_duration'],
        ridge_beta=reservoir_params['ridge_beta'],
        use_nonlinear_output=reservoir_params.get('use_nonlinear_output', False),
        nonlinear_fraction=reservoir_params.get('nonlinear_fraction', 0.5)
    )
    
    logger.info(f"Starting reservoir driving and state collection (washout_duration={data_settings['washout_duration']})...")
    training_start = time.time()
    Wout = training_manager.train(U, Yd)
    training_end = time.time()
    logger.info(f"Trained output matrix Wout ({Wout.shape}) in {training_end - training_start:.2f} seconds")
    
    # Save the model if requested
    if output_settings['save_model']:
        model_file = os.path.join(results_dir, 'reservoir_model.h5')
        logger.info(f"Saving model to {model_file}")
        ModelPersistence.save_model(
            Win=Win,
            A=A,
            Wout=Wout,
            file_path=model_file,
            metadata={
                'reservoir_size': reservoir_params['Dr'],
                'spectral_radius': reservoir_params['rho'],
                'data_dimension': D,
                'training_length': len(U),
                'nonlinear_output': reservoir_params.get('use_nonlinear_output', False)
            }
        )
    
    # Get the final reservoir state from training
    final_r = training_manager.r
    
    # Initialize AutonomousRunner for prediction/generation
    logger.info("Initializing autonomous runner")
    autonomous_runner = AutonomousRunner(
        Win=Win,
        A=A,
        Wout=Wout,
        use_nonlinear_output=reservoir_params.get('use_nonlinear_output', False),
        nonlinear_indices=training_manager.nonlinear_indices
    )
    
    # Run short-term prediction
    short_term_length = eval_settings['prediction_length_short']
    logger.info(f"Running short-term prediction for {short_term_length} steps")
    short_pred_start = time.time()
    V_short, R_short = autonomous_runner.run_autonomous(final_r, short_term_length)
    short_pred_end = time.time()
    logger.info(f"Completed short-term prediction in {short_pred_end - short_pred_start:.2f} seconds")
    
    # Run long-term generation
    long_term_length = eval_settings['generation_length_long']
    logger.info(f"Running long-term generation for {long_term_length} steps")
    long_gen_start = time.time()
    V_long, R_long = autonomous_runner.run_autonomous(final_r, long_term_length)
    long_gen_end = time.time()
    logger.info(f"Completed long-term generation in {long_gen_end - long_gen_start:.2f} seconds")
    
    # Initialize EvaluationManager and perform evaluations
    logger.info("Performing evaluations")
    evaluator = EvaluationManager(
        original_data=data,
        generated_data=V_long,
        generated_states=R_long,
        Win=Win,
        A=A,
        Wout=Wout
    )
    
    # Evaluate short-term prediction accuracy
    logger.info("Evaluating short-term prediction accuracy")
    short_term_start = time.time()
    short_term_metrics = evaluator.evaluate_short_term(
        length=short_term_length,
        plot=output_settings['save_plots'],
        save_dir=results_dir
    )
    short_term_end = time.time()
    logger.info(f"Short-term prediction MSE: {short_term_metrics['mse_total']}")
    logger.info(f"Completed short-term evaluation in {short_term_end - short_term_start:.2f} seconds")
    
    # Evaluate climate statistics
    logger.info("Evaluating climate statistics")
    climate_start = time.time()
    climate_metrics = evaluator.evaluate_climate(
        plot=output_settings['save_plots'],
        save_dir=results_dir
    )
    climate_end = time.time()
    logger.info(f"Completed climate evaluation in {climate_end - climate_start:.2f} seconds")
    
    # Skip Lyapunov calculation if requested
    if args.skip_lyapunov:
        logger.info("Skipping Lyapunov exponent calculation as requested")
        lyapunov_exponents = None
    else:
        # Evaluate Lyapunov exponents
        lyapunov_params = config['lyapunov_params']
        logger.info("Calculating Lyapunov exponents")
        logger.info(f"Parameters: num_exponents={lyapunov_params['num_exponents']}, "
                   f"transient_steps={lyapunov_params['transient_steps_lyap']}, "
                   f"orthonormalization_interval={lyapunov_params['orthonormalization_interval']}")
        
        logger.info("Starting Jacobian calculation and QR decompositions...")
        lyap_start = time.time()
        try:
            lyapunov_exponents = evaluator.evaluate_lyapunov(
                num_exponents=lyapunov_params['num_exponents'],
                transient_steps=lyapunov_params['transient_steps_lyap'],
                orthonormalization_interval=lyapunov_params['orthonormalization_interval'],
                plot=output_settings['save_plots'],
                save_dir=results_dir
            )
            lyap_end = time.time()
            logger.info(f"Calculated {len(lyapunov_exponents)} Lyapunov exponents in {lyap_end - lyap_start:.2f} seconds")
            logger.info(f"First 3 Lyapunov exponents: {lyapunov_exponents[:3]}")
        except Exception as e:
            logger.error(f"Error during Lyapunov calculation: {str(e)}")
            lyapunov_exponents = None
    
    # Save metrics if requested
    if output_settings['save_metrics'] and (short_term_metrics or climate_metrics or lyapunov_exponents is not None):
        metrics_file = os.path.join(results_dir, 'metrics.npz')
        logger.info(f"Saving metrics to {metrics_file}")
        
        # Combine all metrics
        all_metrics = {
            'short_term': short_term_metrics,
            'climate': {
                'mean_diff': climate_metrics['mean_diff'],
                'std_diff': climate_metrics['std_diff'],
                'hist_distances': climate_metrics['hist_distances'],
                'spectra_distances': climate_metrics['spectra_distances']
            }
        }
        
        if lyapunov_exponents is not None:
            all_metrics['lyapunov_exponents'] = lyapunov_exponents
        
        # Save metrics to file
        np.savez(metrics_file, **all_metrics)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Reservoir Computing system completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    main() 