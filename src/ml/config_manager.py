import argparse
import yaml
import os


class ConfigurationManager:
    """
    Configuration manager for the Machine Learning system (Reservoir Computing).
    Responsible for loading, validating, and providing ML-specific configuration.
    """

    def __init__(self, config_path=None, args=None):
        """
        Initialize with either a path to a config file or command-line arguments.
        
        Args:
            config_path: Path to YAML configuration file
            args: Command-line arguments
        """
        self.config = None
        
        if config_path:
            self._load_from_file(config_path)
        elif args:
            self._load_from_args(args)
        else:
            raise ValueError("Either config_path or args must be provided")
        
        self._validate_config()
        
    def _load_from_file(self, config_path):
        """Load configuration from a YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def _load_from_args(self, args):
        """Load configuration from command-line arguments."""
        parser = argparse.ArgumentParser(description="Reservoir Computing configuration")
        
        # Data settings
        parser.add_argument('--training_data_path', required=True, help='Path to training data')
        parser.add_argument('--validation_data_path', help='Path to validation data')
        parser.add_argument('--washout_duration', type=int, default=1000, help='Washout duration')
        
        # Reservoir hyperparameters
        parser.add_argument('--reservoir_size', type=int, default=1000, help='Reservoir size (Dr)')
        parser.add_argument('--spectral_radius', type=float, default=1.2, help='Spectral radius (rho)')
        parser.add_argument('--degree', type=float, default=3.0, help='Average degree for sparse matrix A')
        parser.add_argument('--sigma_input', type=float, default=0.1, help='Scaling factor for Win')
        parser.add_argument('--ridge_beta', type=float, default=1e-6, help='Tikhonov regularization parameter')
        parser.add_argument('--use_nonlinear_output', action='store_true', help='Use nonlinear output')
        parser.add_argument('--nonlinear_fraction', type=float, default=0.5, help='Fraction of nodes for nonlinearity')
        
        # Training settings
        parser.add_argument('--train_length', type=int, default=10000, help='Number of training time steps')
        
        # Evaluation settings
        parser.add_argument('--prediction_length_short', type=int, default=1000, help='Short-term prediction steps')
        parser.add_argument('--generation_length_long', type=int, default=50000, help='Long-term generation steps')
        
        # Lyapunov parameters
        parser.add_argument('--num_exponents', type=int, default=10, help='Number of Lyapunov exponents to calculate')
        parser.add_argument('--transient_steps_lyap', type=int, default=1000, help='Transient steps for Lyapunov')
        parser.add_argument('--orthonormalization_interval', type=int, default=10, help='Steps between QR decompositions')
        
        # Output settings
        parser.add_argument('--results_dir', default='results/ml', help='Directory to save results')
        parser.add_argument('--save_model', action='store_true', help='Save model matrices')
        parser.add_argument('--save_plots', action='store_true', help='Save plots')
        parser.add_argument('--save_metrics', action='store_true', help='Save metrics')
        
        parsed_args = parser.parse_args(args)
        
        # Convert to dictionary
        self.config = {
            'data_settings': {
                'training_data_path': parsed_args.training_data_path,
                'validation_data_path': parsed_args.validation_data_path,
                'washout_duration': parsed_args.washout_duration
            },
            'reservoir_hyperparams': {
                'Dr': parsed_args.reservoir_size,
                'rho': parsed_args.spectral_radius,
                'degree': parsed_args.degree,
                'sigma_input': parsed_args.sigma_input,
                'ridge_beta': parsed_args.ridge_beta,
                'use_nonlinear_output': parsed_args.use_nonlinear_output,
                'nonlinear_fraction': parsed_args.nonlinear_fraction
            },
            'training_settings': {
                'train_length': parsed_args.train_length
            },
            'evaluation_settings': {
                'prediction_length_short': parsed_args.prediction_length_short,
                'generation_length_long': parsed_args.generation_length_long
            },
            'lyapunov_params': {
                'num_exponents': parsed_args.num_exponents,
                'transient_steps_lyap': parsed_args.transient_steps_lyap,
                'orthonormalization_interval': parsed_args.orthonormalization_interval
            },
            'output_settings': {
                'results_dir': parsed_args.results_dir,
                'save_model': parsed_args.save_model,
                'save_plots': parsed_args.save_plots,
                'save_metrics': parsed_args.save_metrics
            }
        }
    
    def _validate_config(self):
        """Validate the configuration."""
        # Required fields
        required_sections = [
            'data_settings', 
            'reservoir_hyperparams', 
            'training_settings',
            'evaluation_settings',
            'lyapunov_params',
            'output_settings'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data_settings
        if 'training_data_path' not in self.config['data_settings']:
            raise ValueError("training_data_path is required in data_settings")
        
        # Validate reservoir_hyperparams
        reservoir_params = self.config['reservoir_hyperparams']
        required_reservoir_params = ['Dr', 'rho', 'degree', 'sigma_input', 'ridge_beta']
        for param in required_reservoir_params:
            if param not in reservoir_params:
                raise ValueError(f"Missing required reservoir parameter: {param}")
        
        # Validate types and ranges
        if not isinstance(reservoir_params['Dr'], int) or reservoir_params['Dr'] <= 0:
            raise ValueError("Reservoir size Dr must be a positive integer")
        
        if not isinstance(reservoir_params['rho'], (int, float)) or reservoir_params['rho'] <= 0:
            raise ValueError("Spectral radius rho must be a positive number")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_settings']['results_dir'], exist_ok=True)
    
    def get_config(self):
        """Get the validated configuration."""
        return self.config 