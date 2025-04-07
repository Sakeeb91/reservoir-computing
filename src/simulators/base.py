from abc import ABC, abstractmethod
import numpy as np


class ISimulator(ABC):
    """
    Interface for all dynamical system simulators.
    """
    
    @abstractmethod
    def initialize(self, params, initial_state=None):
        """
        Initialize the simulator with parameters and initial state.
        
        Args:
            params: Dictionary of system-specific parameters
            initial_state: Initial state vector or None to generate automatically
        """
        pass
    
    @abstractmethod
    def run_transient(self, duration, dt_integration):
        """
        Run the simulation for a transient period without recording full trajectory.
        
        Args:
            duration: Duration of the transient simulation
            dt_integration: Time step for numerical integration
            
        Returns:
            The state of the system after the transient period
        """
        pass
    
    @abstractmethod
    def run_record(self, duration, dt_integration, dt_sampling):
        """
        Run the simulation and record the trajectory.
        
        Args:
            duration: Duration of the recorded simulation
            dt_integration: Time step for numerical integration
            dt_sampling: Time step for sampling the trajectory
            
        Returns:
            Tuple of (time_points, trajectory) where:
                time_points: 1D array of time points
                trajectory: 2D array of state vectors (rows are time points, columns are state variables)
        """
        pass
    
    @abstractmethod
    def get_state_dimension(self):
        """
        Returns the dimension of the system's state vector.
        
        Returns:
            Integer representing the state dimension
        """
        pass 