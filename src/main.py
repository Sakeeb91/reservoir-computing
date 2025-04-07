#!/usr/bin/env python3
"""
Chaotic Dynamical Systems Data Generator

This script orchestrates the generation of time-series data from chaotic dynamical systems.
"""

import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple

from simulators import SimulatorFactory
from utils import ConfigurationManager, NoiseInjector, DataWriter


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def setup_initial_conditions(sim_settings: Dict[str, Any], state_dim: int) -> Optional[np.ndarray]:
    """
    Set up initial conditions based on simulation settings.
    
    Args:
        sim_settings: Simulation settings dictionary
        state_dim: Dimension of the system state
        
    Returns:
        Initial state or None to use default random initialization
    """
    if 'initial_conditions' not in sim_settings:
        return None
        
    if sim_settings['initial_conditions'] == 'random':
        return None  # Let the simulator generate random initial conditions
    
    elif sim_settings['initial_conditions'] == 'fixed':
        if 'initial_state' in sim_settings:
            initial_state = np.array(sim_settings['initial_state'])
            
            # Validate dimension
            if len(initial_state) != state_dim:
                raise ValueError(f"Initial state dimension ({len(initial_state)}) "
                                f"does not match system state dimension ({state_dim})")
                
            return initial_state
        else:
            raise ValueError("Fixed initial conditions specified but no initial_state provided")
    
    else:
        raise ValueError(f"Invalid initial_conditions value: {sim_settings['initial_conditions']}")


def generate_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Main function to generate chaotic system data based on configuration.
    
    Returns:
        Tuple of (time_points, trajectory, metadata)
    """
    # Load configuration
    config_manager = ConfigurationManager()
    
    # Check if a config file is provided as command-line argument
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        config = config_manager.load_from_file(sys.argv[1])
        logger.info(f"Loaded configuration from file: {sys.argv[1]}")
    else:
        # Otherwise, load from command-line arguments
        config = config_manager.load_from_args()
        logger.info("Loaded configuration from command-line arguments")
    
    # Get configuration components
    system_type = config_manager.get_system_type()
    system_params = config_manager.get_system_params()
    sim_settings = config_manager.get_simulation_settings()
    noise_settings = config_manager.get_noise_settings()
    output_settings = config_manager.get_output_settings()
    
    logger.info(f"Generating data for {system_type} system")
    logger.debug(f"System parameters: {system_params}")
    
    # Create simulator
    simulator_factory = SimulatorFactory()
    simulator = simulator_factory.create_simulator(system_type)
    
    # Get state dimension
    state_dim = simulator.get_state_dimension()
    logger.debug(f"System state dimension: {state_dim}")
    
    # Set up initial conditions
    initial_state = setup_initial_conditions(sim_settings, state_dim)
    
    # Initialize simulator
    state = simulator.initialize(system_params, initial_state)
    logger.debug(f"Initial state: {state}")
    
    # Run transient simulation
    logger.info(f"Running transient simulation for {sim_settings['transient_time']} time units")
    post_transient_state = simulator.run_transient(
        sim_settings['transient_time'],
        sim_settings['dt_integration']
    )
    logger.debug(f"Post-transient state: {post_transient_state}")
    
    # Run recorded simulation
    logger.info(f"Running recorded simulation for {sim_settings['record_time']} time units")
    time_points, trajectory = simulator.run_record(
        sim_settings['record_time'],
        sim_settings['dt_integration'],
        sim_settings['dt_sampling']
    )
    
    # Add system-specific information to metadata
    system_info = {
        'state_dimension': state_dim,
        'num_time_points': len(time_points),
        'time_range': [float(time_points[0]), float(time_points[-1])],
        'system_type': system_type
    }
    
    # Generate metadata
    metadata = DataWriter.merge_metadata(config, system_info)
    
    # Add noise if specified
    if noise_settings and noise_settings.get('add_noise', False):
        logger.info(f"Adding {noise_settings['noise_type']} noise at level {noise_settings['noise_level']}")
        trajectory = NoiseInjector.add_noise(trajectory, noise_settings)
    
    return time_points, trajectory, metadata


def main():
    """Main entry point for the data generator."""
    try:
        # Generate the data
        time_points, trajectory, metadata = generate_data()
        
        # Get output settings
        config_manager = ConfigurationManager()
        output_settings = config_manager.get_output_settings()
        
        # Save the data
        output_file = DataWriter.save_data(time_points, trajectory, output_settings, metadata)
        
        logger.info(f"Data saved to {output_file}")
        logger.info(f"Generated {len(time_points)} time points with state dimension {trajectory.shape[1]}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 