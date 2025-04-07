import numpy as np
import h5py
from scipy import sparse
import os


class ModelPersistence:
    """
    Handles saving and loading of trained model components.
    """
    
    @staticmethod
    def save_model(Win, A, Wout, file_path, metadata=None):
        """
        Save the trained model components to a file.
        
        Args:
            Win: Input matrix
            A: Reservoir matrix
            Wout: Output weights matrix
            file_path: Path to save the model
            metadata: Optional dictionary with additional metadata
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Determine if we're using HDF5 or NumPy format
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            ModelPersistence._save_hdf5(Win, A, Wout, file_path, metadata)
        else:
            # Default to NumPy format
            if not file_path.endswith('.npz'):
                file_path += '.npz'
            ModelPersistence._save_numpy(Win, A, Wout, file_path, metadata)
    
    @staticmethod
    def _save_hdf5(Win, A, Wout, file_path, metadata=None):
        """Save model components to HDF5 file."""
        with h5py.File(file_path, 'w') as f:
            # Save Win matrix
            f.create_dataset('Win', data=Win)
            
            # Save A matrix (handle sparse matrix)
            if sparse.issparse(A):
                # Create a group for the sparse matrix
                A_group = f.create_group('A')
                A_group.attrs['format'] = A.format
                A_group.attrs['shape'] = A.shape
                
                # Store data, indices, and indptr
                A_group.create_dataset('data', data=A.data)
                A_group.create_dataset('indices', data=A.indices)
                A_group.create_dataset('indptr', data=A.indptr)
            else:
                # Save as dense matrix
                f.create_dataset('A', data=A)
            
            # Save Wout matrix
            f.create_dataset('Wout', data=Wout)
            
            # Save metadata if provided
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    # Convert NumPy types to Python types for proper storage
                    if isinstance(value, (np.integer, np.floating, np.bool_)):
                        value = value.item()
                    
                    # Only store simple types directly as attributes
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        meta_group.create_dataset(key, data=np.array(value))
    
    @staticmethod
    def _save_numpy(Win, A, Wout, file_path, metadata=None):
        """Save model components to NumPy .npz file."""
        save_dict = {
            'Win': Win,
            'Wout': Wout
        }
        
        # Handle sparse matrix A
        if sparse.issparse(A):
            save_dict['A_data'] = A.data
            save_dict['A_indices'] = A.indices
            save_dict['A_indptr'] = A.indptr
            save_dict['A_shape'] = np.array(A.shape)
            save_dict['A_format'] = np.array([A.format], dtype='S10')
        else:
            save_dict['A'] = A
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                save_dict[f'meta_{key}'] = value
        
        # Save to file
        np.savez(file_path, **save_dict)
    
    @staticmethod
    def load_model(file_path):
        """
        Load a trained model from file.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Tuple of (Win, A, Wout, metadata)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Determine file format
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            return ModelPersistence._load_hdf5(file_path)
        else:
            return ModelPersistence._load_numpy(file_path)
    
    @staticmethod
    def _load_hdf5(file_path):
        """Load model components from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            # Load Win matrix
            Win = f['Win'][:]
            
            # Load A matrix (handle sparse matrix)
            if 'A' in f and isinstance(f['A'], h5py.Group):
                # Load sparse matrix
                A_group = f['A']
                A_format = A_group.attrs['format']
                A_shape = tuple(A_group.attrs['shape'])
                
                # Create sparse matrix
                A = sparse.csr_matrix((A_group['data'][:], A_group['indices'][:], A_group['indptr'][:]),
                                      shape=A_shape)
                
                # Convert to the right format if needed
                if A_format != 'csr':
                    A = A.asformat(A_format)
            else:
                # Load dense matrix
                A = f['A'][:]
            
            # Load Wout matrix
            Wout = f['Wout'][:]
            
            # Load metadata if available
            metadata = {}
            if 'metadata' in f:
                meta_group = f['metadata']
                
                # Load attributes
                for key, value in meta_group.attrs.items():
                    metadata[key] = value
                
                # Load datasets
                for key in meta_group.keys():
                    metadata[key] = meta_group[key][:]
            
            return Win, A, Wout, metadata
    
    @staticmethod
    def _load_numpy(file_path):
        """Load model components from NumPy .npz file."""
        data = np.load(file_path, allow_pickle=True)
        
        # Load Win and Wout
        Win = data['Win']
        Wout = data['Wout']
        
        # Load A (handle sparse matrix)
        if 'A' in data:
            A = data['A']
        else:
            # Reconstruct sparse matrix
            A_shape = tuple(data['A_shape'])
            A_format = data['A_format'].item().decode('utf-8')
            
            A = sparse.csr_matrix((data['A_data'], data['A_indices'], data['A_indptr']),
                                 shape=A_shape)
            
            # Convert to the right format if needed
            if A_format != 'csr':
                A = A.asformat(A_format)
        
        # Extract metadata
        metadata = {}
        for key in data.keys():
            if key.startswith('meta_'):
                metadata[key[5:]] = data[key]
        
        return Win, A, Wout, metadata 