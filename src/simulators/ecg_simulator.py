import numpy as np
import os
from src.simulators.base import ISimulator


class ECGSimulator(ISimulator):
    """
    Adapter class to make ECG data work with the ISimulator interface.
    This allows us to use existing components of the reservoir computing framework.
    """
    
    def __init__(self):
        """Initialize the ECG simulator."""
        self.data = None
        self.data_index = 0
        self.current_state = None
        self.data_dimension = 0
        self.is_initialized = False
        
    def initialize(self, params, initial_state=None):
        """
        Initialize the simulator with ECG data.
        
        Args:
            params: Dictionary with keys:
                'data_path': Path to ECG data file (CSV)
                'is_normal': Boolean indicating if this is normal data
                'max_samples': Maximum number of samples to load
            initial_state: Not used for ECG data
        """
        data_path = params.get('data_path')
        is_normal = params.get('is_normal', True)
        max_samples = params.get('max_samples', 1000)
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"ECG data file not found: {data_path}")
        
        # Load data from CSV
        self.data = np.loadtxt(data_path, delimiter=',', max_rows=max_samples)
        
        # The last column in each row could be a label, if so, remove it
        if self.data.shape[1] > 188 and np.all(self.data[:, -1] == 1.0):
            self.data = self.data[:, :-1]
        
        self.data_dimension = self.data.shape[1]
        self.is_initialized = True
        self.data_index = 0
        
        # Set initial state to the first sample
        self.current_state = self.data[self.data_index].copy()
        
        # Add metadata
        self.is_normal = is_normal
        self.label = "Normal" if is_normal else "Abnormal"
        
        return self.current_state
        
    def run_transient(self, duration, dt_integration):
        """
        Advance through some samples without recording.
        
        Args:
            duration: Number of samples to skip
            dt_integration: Not used for ECG data
            
        Returns:
            Current state after skipping
        """
        if not self.is_initialized:
            raise RuntimeError("ECG simulator not initialized")
        
        # For ECG data, we just increment the index
        steps = int(duration)
        self.data_index = (self.data_index + steps) % len(self.data)
        self.current_state = self.data[self.data_index].copy()
        
        return self.current_state
        
    def run_record(self, duration, dt_integration, dt_sampling):
        """
        Run through a sequence of ECG samples and record them.
        
        Args:
            duration: Number of samples to return
            dt_integration: Not used for ECG data
            dt_sampling: If > 1, will skip samples
            
        Returns:
            Tuple of (time_points, trajectory)
        """
        if not self.is_initialized:
            raise RuntimeError("ECG simulator not initialized")
        
        # For ECG data, determine how many samples to collect
        steps = int(duration)
        sampling_steps = max(1, int(dt_sampling))
        
        # Create arrays for time and trajectory
        time_points = np.arange(steps) * dt_sampling
        trajectory = np.zeros((steps, self.data_dimension))
        
        # Fill the trajectory with samples from data
        for i in range(steps):
            # Get current sample
            trajectory[i] = self.data[self.data_index].copy()
            
            # Increment the index with sampling_steps
            self.data_index = (self.data_index + sampling_steps) % len(self.data)
        
        # Update current state
        self.current_state = self.data[self.data_index].copy()
        
        return time_points, trajectory
        
    def get_state_dimension(self):
        """Returns the dimension of an ECG sample."""
        if not self.is_initialized:
            raise RuntimeError("ECG simulator not initialized")
        
        return self.data_dimension


def create_ecg_simulator(data_path, is_normal=True, max_samples=1000):
    """
    Helper function to create and initialize an ECG simulator.
    
    Args:
        data_path: Path to ECG data file
        is_normal: Whether this is normal ECG data
        max_samples: Maximum samples to load
        
    Returns:
        Initialized ECGSimulator instance
    """
    sim = ECGSimulator()
    sim.initialize({
        'data_path': data_path,
        'is_normal': is_normal,
        'max_samples': max_samples
    })
    return sim 