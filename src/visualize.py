#!/usr/bin/env python3
"""
Visualization utilities for chaotic attractor data.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(data_path):
    """
    Load time series data from a file.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Tuple of (time_points, trajectory, metadata)
    """
    # Determine file format from extension
    _, ext = os.path.splitext(data_path)
    ext = ext.lower()
    
    # Load data based on file format
    if ext == '.h5' or ext == '.hdf5':
        # Load HDF5 file
        with h5py.File(data_path, 'r') as f:
            time_points = f['time'][:]
            trajectory = f['trajectory'][:]
            
            # Load metadata from attributes
            metadata = {}
            for key in f.attrs:
                value = f.attrs[key]
                # Try to parse JSON strings
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                metadata[key] = value
                
    elif ext == '.npy':
        # Assume there are companion files
        base_path = data_path.replace('_trajectory.npy', '').replace('_time.npy', '')
        trajectory_path = f"{base_path}_trajectory.npy"
        time_path = f"{base_path}_time.npy"
        metadata_path = f"{base_path}_metadata.json"
        
        # Load trajectory and time
        trajectory = np.load(trajectory_path)
        time_points = np.load(time_path)
        
        # Load metadata if available
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
    elif ext == '.csv':
        # Load CSV file
        df = pd.read_csv(data_path)
        
        # Extract time and trajectory
        time_points = df['time'].values
        trajectory = df.drop('time', axis=1).values
        
        # Extract metadata from CSV header if available
        metadata = {}
        with open(data_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('# Metadata:'):
                metadata_str = first_line[len('# Metadata:'):].strip()
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    pass
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return time_points, trajectory, metadata


def plot_time_series(time_points, trajectory, title='Time Series'):
    """
    Plot time series for each state variable.
    
    Args:
        time_points: 1D array of time points
        trajectory: 2D array of state vectors
        title: Title for the plot
    """
    n_vars = trajectory.shape[1]
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 2*n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]  # Make it iterable for single variable case
    
    for i, ax in enumerate(axes):
        ax.plot(time_points, trajectory[:, i])
        ax.set_ylabel(f'x{i}')
        ax.grid(True)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_phase_portrait(trajectory, title='Phase Portrait'):
    """
    Plot phase portrait for a 2D or a 3D system.
    
    Args:
        trajectory: 2D array of state vectors
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    n_vars = trajectory.shape[1]
    
    if n_vars == 2:
        # 2D phase portrait
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.grid(True)
        
    elif n_vars >= 3:
        # 3D phase portrait (first three variables)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', alpha=0.5)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')
        
    else:
        # 1D system - plot against time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(len(trajectory)), trajectory[:, 0], 'b-')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('x0')
        ax.grid(True)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_spatial_temporal_ks(time_points, trajectory, title='Kuramoto-Sivashinsky'):
    """
    Create a space-time plot for Kuramoto-Sivashinsky data.
    
    Args:
        time_points: 1D array of time points
        trajectory: 2D array of state vectors
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh for time and space
    x = np.linspace(0, 1, trajectory.shape[1])
    t = time_points
    
    X, T = np.meshgrid(x, t)
    
    # Plot as a pcolormesh
    c = ax.pcolormesh(X, T, trajectory, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax, label='Amplitude')
    
    ax.set_xlabel('Space (normalized)')
    ax.set_ylabel('Time')
    fig.suptitle(title)
    
    plt.tight_layout()
    
    return fig


def main():
    """Main function for visualization."""
    parser = argparse.ArgumentParser(description='Visualize chaotic attractor data')
    parser.add_argument('data_file', type=str, help='Path to the data file')
    parser.add_argument('--output', type=str, help='Path to save the figures (optional)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figures')
    parser.add_argument('--time-series', action='store_true', help='Plot time series')
    parser.add_argument('--phase-portrait', action='store_true', help='Plot phase portrait')
    parser.add_argument('--space-time', action='store_true', help='Plot space-time diagram (for KS)')
    parser.add_argument('--show', action='store_true', help='Show the plots')
    
    args = parser.parse_args()
    
    # Load the data
    try:
        time_points, trajectory, metadata = load_data(args.data_file)
        
        # Determine system type from metadata if available
        system_type = metadata.get('system_type', 'unknown')
        print(f"Loaded data for {system_type} system")
        print(f"Time range: {time_points[0]} to {time_points[-1]}")
        print(f"State dimension: {trajectory.shape[1]}")
        
        # If no specific plots are requested, show all applicable plots
        if not (args.time_series or args.phase_portrait or args.space_time):
            args.time_series = True
            args.phase_portrait = True
            if system_type in ['kuramoto_sivashinsky', 'ks']:
                args.space_time = True
        
        # Create the requested plots
        figures = []
        
        # Time series plot
        if args.time_series:
            fig = plot_time_series(time_points, trajectory, 
                                   title=f'Time Series for {system_type.capitalize()} System')
            figures.append(('time_series', fig))
        
        # Phase portrait plot
        if args.phase_portrait and trajectory.shape[1] >= 2:
            fig = plot_phase_portrait(trajectory,
                                      title=f'Phase Portrait for {system_type.capitalize()} System')
            figures.append(('phase_portrait', fig))
        
        # Space-time plot for KS
        if args.space_time and system_type in ['kuramoto_sivashinsky', 'ks']:
            fig = plot_spatial_temporal_ks(time_points, trajectory,
                                           title='Kuramoto-Sivashinsky Space-Time Diagram')
            figures.append(('space_time', fig))
        
        # Save or show the figures
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            for name, fig in figures:
                output_path = os.path.join(args.output, f"{name}.png")
                fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
                print(f"Saved figure to {output_path}")
        
        if args.show:
            plt.show()
        
        return 0
        
    except Exception as e:
        print(f"Error visualizing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 