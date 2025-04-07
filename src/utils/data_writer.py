import os
import numpy as np
import pandas as pd
import json
import h5py
from typing import Dict, Any, Tuple, Optional


class DataWriter:
    """
    Responsible for saving generated time series data and metadata.
    """
    
    @staticmethod
    def save_data(time_points: np.ndarray, 
                  trajectory: np.ndarray, 
                  output_settings: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save the time series data and metadata using the specified settings.
        
        Args:
            time_points: 1D array of time points
            trajectory: 2D array of state vectors (rows are time points, columns are state variables)
            output_settings: Dictionary with output settings
                - output_path: Directory to save data
                - output_filename: Base name for output files
                - output_format: 'npy', 'csv', or 'hdf5'
                - save_metadata: Whether to save metadata alongside data
            metadata: Optional dictionary with metadata to save
            
        Returns:
            Path to the saved data file
        """
        # Extract output settings
        output_path = output_settings['output_path']
        output_filename = output_settings['output_filename']
        output_format = output_settings['output_format']
        save_metadata = output_settings.get('save_metadata', True)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Determine full output path based on format
        if output_format == 'npy':
            # For NPY format, save time points and trajectory separately
            trajectory_path = os.path.join(output_path, f"{output_filename}_trajectory.npy")
            time_path = os.path.join(output_path, f"{output_filename}_time.npy")
            
            # Save data
            np.save(trajectory_path, trajectory)
            np.save(time_path, time_points)
            
            # Save metadata if requested
            if save_metadata and metadata:
                metadata_path = os.path.join(output_path, f"{output_filename}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            return trajectory_path
            
        elif output_format == 'csv':
            # For CSV format, combine time and trajectory into a DataFrame
            output_file = os.path.join(output_path, f"{output_filename}.csv")
            
            # Create DataFrame
            df = pd.DataFrame(trajectory)
            
            # Add time column
            df.insert(0, 'time', time_points)
            
            # Rename columns
            column_names = ['time'] + [f'x{i}' for i in range(trajectory.shape[1])]
            df.columns = column_names
            
            # Save to CSV
            if save_metadata and metadata:
                # Convert metadata to string for CSV header
                metadata_str = json.dumps(metadata)
                # Save with metadata as comment
                df.to_csv(output_file, index=False, header=True, 
                          float_format='%.10g', 
                          encoding='utf-8')
                
                # Prepend metadata as comment to the CSV file
                with open(output_file, 'r') as f:
                    csv_content = f.read()
                
                with open(output_file, 'w') as f:
                    f.write(f"# Metadata: {metadata_str}\n")
                    f.write(csv_content)
            else:
                df.to_csv(output_file, index=False, header=True, 
                          float_format='%.10g', 
                          encoding='utf-8')
                
            return output_file
            
        elif output_format == 'hdf5':
            # For HDF5 format, save everything in a single file
            output_file = os.path.join(output_path, f"{output_filename}.h5")
            
            with h5py.File(output_file, 'w') as f:
                # Create datasets for time and trajectory
                f.create_dataset('time', data=time_points)
                f.create_dataset('trajectory', data=trajectory)
                
                # Save metadata as attributes if requested
                if save_metadata and metadata:
                    for key, value in metadata.items():
                        # Convert complex data structures to JSON strings
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        f.attrs[key] = value
                        
            return output_file
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
    @staticmethod
    def merge_metadata(config: Dict[str, Any], 
                       system_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Merge configuration and additional system information into metadata.
        
        Args:
            config: Configuration dictionary
            system_info: Optional additional system information
            
        Returns:
            Metadata dictionary
        """
        # Start with a copy of the configuration
        metadata = config.copy()
        
        # Add additional system information
        if system_info:
            metadata['system_info'] = system_info
            
        # Add timestamp
        from datetime import datetime
        metadata['timestamp'] = datetime.now().isoformat()
        
        return metadata 