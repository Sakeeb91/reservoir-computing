import numpy as np
import h5py


class DataLoader:
    """
    Loads time series data from file and prepares input/target sequences for
    training the Reservoir Computing system.
    """
    
    def __init__(self, data_path):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data file from System 1
        """
        self.data_path = data_path
        self.data = None
        self.dimension = None
        
    def load_data(self, train_length=None):
        """
        Load data from file.
        
        Args:
            train_length: Number of time steps to use for training (if None, use all available data)
            
        Returns:
            Tuple of (input_data, time_points)
        """
        # Determine file type and load accordingly
        if self.data_path.endswith('.h5') or self.data_path.endswith('.hdf5'):
            self._load_hdf5(train_length)
        elif self.data_path.endswith('.npy'):
            self._load_numpy(train_length)
        else:
            raise ValueError(f"Unsupported file type: {self.data_path}")
        
        # Get data dimension (number of state variables)
        self.dimension = self.data.shape[1] if len(self.data.shape) > 1 else 1
        
        return self.data, self.time_points
    
    def _load_hdf5(self, train_length):
        """Load data from HDF5 file."""
        with h5py.File(self.data_path, 'r') as f:
            # Check if expected datasets exist
            if 'trajectory' not in f or 'time_points' not in f:
                raise ValueError("HDF5 file must contain 'trajectory' and 'time_points' datasets")
            
            # Load data
            if train_length is None:
                self.data = f['trajectory'][:]
                self.time_points = f['time_points'][:]
            else:
                self.data = f['trajectory'][:train_length]
                self.time_points = f['time_points'][:train_length]
    
    def _load_numpy(self, train_length):
        """Load data from NumPy file."""
        full_data = np.load(self.data_path)
        
        if train_length is None or train_length >= len(full_data):
            self.data = full_data
        else:
            self.data = full_data[:train_length]
        
        # Generate time points if not available
        self.time_points = np.arange(len(self.data))
    
    def prepare_training_data(self):
        """
        Prepare input and target sequences for training.
        Target is the next time step after input.
        
        Returns:
            Tuple of (U, Yd) where:
                U: Input sequence (all time steps except the last one)
                Yd: Target sequence (all time steps except the first one)
        """
        if self.data is None:
            raise ValueError("Data must be loaded before preparing training data")
        
        # Input sequence U is all time steps except the last one
        U = self.data[:-1]
        
        # Target sequence Yd is all time steps except the first one
        Yd = self.data[1:]
        
        return U, Yd
    
    def get_data_dimension(self):
        """
        Get the dimension of the data.
        
        Returns:
            Integer representing the dimension of the state vector
        """
        if self.dimension is None:
            raise ValueError("Data must be loaded before getting dimension")
        
        return self.dimension 