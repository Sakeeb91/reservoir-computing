import os
import yaml
import json
import argparse
from typing import Dict, Any, Optional


class ConfigurationManager:
    """
    Responsible for loading, validating, and providing configuration parameters
    for chaotic dynamical system simulations.
    """
    
    def __init__(self):
        """Initialize the ConfigurationManager."""
        self.config = {}
        
    def load_from_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file (YAML or JSON).
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            The validated configuration dictionary
            
        Raises:
            FileNotFoundError: If the config file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format from extension
        _, ext = os.path.splitext(config_path)
        ext = ext.lower()
        
        # Load configuration based on file format
        if ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif ext == '.json':
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        # Validate the loaded configuration
        self._validate_config()
        
        return self.config
    
    def load_from_args(self) -> Dict[str, Any]:
        """
        Load configuration from command-line arguments.
        
        Returns:
            The validated configuration dictionary
        """
        parser = argparse.ArgumentParser(description='Chaotic Dynamical Systems Data Generator')
        
        # System type and parameters
        parser.add_argument('--system_type', type=str, choices=['lorenz', 'kuramoto_sivashinsky', 'ks'],
                           help='Type of chaotic system to simulate')
        
        # Lorenz parameters
        parser.add_argument('--sigma', type=float, help='Sigma parameter for Lorenz system')
        parser.add_argument('--rho', type=float, help='Rho parameter for Lorenz system')
        parser.add_argument('--beta', type=float, help='Beta parameter for Lorenz system')
        
        # KS parameters
        parser.add_argument('--L', type=float, help='Domain size for KS equation')
        parser.add_argument('--Q', type=int, help='Number of grid points for KS equation')
        
        # Simulation settings
        parser.add_argument('--initial_conditions', type=str, choices=['random', 'fixed'],
                           help='Method for generating initial conditions')
        parser.add_argument('--initial_state', type=str, help='Comma-separated initial state values (for fixed)')
        parser.add_argument('--transient_time', type=float, help='Duration of transient simulation')
        parser.add_argument('--record_time', type=float, help='Duration of recorded simulation')
        parser.add_argument('--dt_integration', type=float, help='Time step for numerical integration')
        parser.add_argument('--dt_sampling', type=float, help='Time step for sampling output data')
        
        # Noise settings
        parser.add_argument('--add_noise', action='store_true', help='Add noise to output data')
        parser.add_argument('--noise_type', type=str, choices=['uniform', 'gaussian'],
                           help='Type of noise to add')
        parser.add_argument('--noise_level', type=float, help='Noise level relative to signal RMS')
        
        # Output settings
        parser.add_argument('--output_path', type=str, help='Directory to save output data')
        parser.add_argument('--output_filename', type=str, help='Base name for output files')
        parser.add_argument('--output_format', type=str, choices=['npy', 'csv', 'hdf5'],
                           help='Format for output data files')
        parser.add_argument('--save_metadata', action='store_true', help='Save metadata alongside data')
        
        # Configuration file
        parser.add_argument('--config_file', type=str, help='Path to configuration file')
        
        args = parser.parse_args()
        
        # If config file is provided, load it first
        if args.config_file:
            self.load_from_file(args.config_file)
        
        # Override config with command-line arguments
        config_dict = {}
        
        # Process system parameters
        if args.system_type:
            config_dict['system_type'] = args.system_type
            
            # Create system_params dict based on system type
            system_params = {}
            if args.system_type == 'lorenz':
                if args.sigma:
                    system_params['sigma'] = args.sigma
                if args.rho:
                    system_params['rho'] = args.rho
                if args.beta:
                    system_params['beta'] = args.beta
            elif args.system_type in ['kuramoto_sivashinsky', 'ks']:
                if args.L:
                    system_params['L'] = args.L
                if args.Q:
                    system_params['Q'] = args.Q
                    
            if system_params:
                config_dict['system_params'] = system_params
        
        # Process simulation settings
        simulation_settings = {}
        if args.initial_conditions:
            simulation_settings['initial_conditions'] = args.initial_conditions
        if args.initial_state:
            # Parse comma-separated values
            initial_state = [float(val.strip()) for val in args.initial_state.split(',')]
            simulation_settings['initial_state'] = initial_state
        if args.transient_time:
            simulation_settings['transient_time'] = args.transient_time
        if args.record_time:
            simulation_settings['record_time'] = args.record_time
        if args.dt_integration:
            simulation_settings['dt_integration'] = args.dt_integration
        if args.dt_sampling:
            simulation_settings['dt_sampling'] = args.dt_sampling
            
        if simulation_settings:
            config_dict['simulation_settings'] = simulation_settings
        
        # Process noise settings
        if args.add_noise:
            noise_settings = {'add_noise': True}
            if args.noise_type:
                noise_settings['noise_type'] = args.noise_type
            if args.noise_level:
                noise_settings['noise_level'] = args.noise_level
            config_dict['noise_settings'] = noise_settings
        
        # Process output settings
        output_settings = {}
        if args.output_path:
            output_settings['output_path'] = args.output_path
        if args.output_filename:
            output_settings['output_filename'] = args.output_filename
        if args.output_format:
            output_settings['output_format'] = args.output_format
        if args.save_metadata:
            output_settings['save_metadata'] = True
            
        if output_settings:
            config_dict['output_settings'] = output_settings
        
        # Update configuration with command-line arguments
        self._update_nested_dict(self.config, config_dict)
        
        # Validate the resulting configuration
        self._validate_config()
        
        return self.config
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with another dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Check for required top-level keys
        required_keys = ['system_type', 'system_params', 'simulation_settings', 'output_settings']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate system type
        if self.config['system_type'] not in ['lorenz', 'kuramoto_sivashinsky', 'ks']:
            raise ValueError(f"Invalid system_type: {self.config['system_type']}")
        
        # Validate system parameters based on system type
        system_params = self.config['system_params']
        if self.config['system_type'] == 'lorenz':
            required_params = ['sigma', 'rho', 'beta']
            for param in required_params:
                if param not in system_params:
                    raise ValueError(f"Missing required Lorenz parameter: {param}")
        elif self.config['system_type'] in ['kuramoto_sivashinsky', 'ks']:
            required_params = ['L', 'Q']
            for param in required_params:
                if param not in system_params:
                    raise ValueError(f"Missing required Kuramoto-Sivashinsky parameter: {param}")
            # Q must be an integer
            if not isinstance(system_params['Q'], int):
                raise ValueError("Parameter Q must be an integer")
        
        # Validate simulation settings
        sim_settings = self.config['simulation_settings']
        required_sim_settings = ['transient_time', 'record_time', 'dt_integration', 'dt_sampling']
        for setting in required_sim_settings:
            if setting not in sim_settings:
                raise ValueError(f"Missing required simulation setting: {setting}")
        
        # Initial conditions should be 'random' or 'fixed'
        if 'initial_conditions' in sim_settings:
            if sim_settings['initial_conditions'] not in ['random', 'fixed']:
                raise ValueError(f"Invalid initial_conditions value: {sim_settings['initial_conditions']}")
            
            # If initial conditions are fixed, initial_state should be provided
            if sim_settings['initial_conditions'] == 'fixed' and 'initial_state' not in sim_settings:
                raise ValueError("Fixed initial conditions require an initial_state")
        
        # Validate noise settings if provided
        if 'noise_settings' in self.config:
            noise_settings = self.config['noise_settings']
            if 'add_noise' in noise_settings and noise_settings['add_noise']:
                required_noise_settings = ['noise_type', 'noise_level']
                for setting in required_noise_settings:
                    if setting not in noise_settings:
                        raise ValueError(f"Missing required noise setting: {setting}")
                
                if noise_settings['noise_type'] not in ['uniform', 'gaussian']:
                    raise ValueError(f"Invalid noise_type: {noise_settings['noise_type']}")
                
                if not isinstance(noise_settings['noise_level'], (int, float)) or noise_settings['noise_level'] < 0:
                    raise ValueError(f"Invalid noise_level: {noise_settings['noise_level']}")
        
        # Validate output settings
        output_settings = self.config['output_settings']
        required_output_settings = ['output_path', 'output_filename', 'output_format']
        for setting in required_output_settings:
            if setting not in output_settings:
                raise ValueError(f"Missing required output setting: {setting}")
        
        if output_settings['output_format'] not in ['npy', 'csv', 'hdf5']:
            raise ValueError(f"Invalid output_format: {output_settings['output_format']}")
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            The configuration dictionary
        """
        return self.config
        
    def get_system_type(self) -> str:
        """Get the system type from the configuration."""
        return self.config['system_type']
    
    def get_system_params(self) -> Dict[str, Any]:
        """Get the system parameters from the configuration."""
        return self.config['system_params']
    
    def get_simulation_settings(self) -> Dict[str, Any]:
        """Get the simulation settings from the configuration."""
        return self.config['simulation_settings']
    
    def get_noise_settings(self) -> Optional[Dict[str, Any]]:
        """Get the noise settings from the configuration, if present."""
        return self.config.get('noise_settings', None)
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get the output settings from the configuration."""
        return self.config['output_settings'] 