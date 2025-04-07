import numpy as np
import h5py
import os
from scipy.integrate import solve_ivp


def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Lorenz system of differential equations.
    
    Args:
        t: Time (not used, but required by solve_ivp)
        state: System state [x, y, z]
        sigma, rho, beta: System parameters
    
    Returns:
        State derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    return [dx_dt, dy_dt, dz_dt]


def generate_lorenz_data(duration=100.0, dt_sampling=0.01, initial_state=None):
    """
    Generate time series data from the Lorenz system.
    
    Args:
        duration: Total duration of the simulation
        dt_sampling: Time step for sampling
        initial_state: Initial state vector or None for random initialization
    
    Returns:
        Tuple of (time_points, trajectory)
    """
    # Set initial state
    if initial_state is None:
        initial_state = np.random.rand(3) * 2 - 1  # Random in [-1, 1]
    
    # Create time points
    t_span = (0, duration)
    t_eval = np.arange(0, duration, dt_sampling)
    
    # Solve the ODE system
    solution = solve_ivp(
        lorenz_system,
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-10,
        atol=1e-10
    )
    
    time_points = solution.t
    trajectory = solution.y.T  # Transpose to get (time_steps, 3) shape
    
    return time_points, trajectory


def save_data_hdf5(time_points, trajectory, file_path):
    """
    Save time series data to HDF5 file.
    
    Args:
        time_points: Array of time points
        trajectory: Array of state vectors
        file_path: Path to save the file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('time_points', data=time_points)
        f.create_dataset('trajectory', data=trajectory)


def main():
    """Generate and save Lorenz data."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Generating Lorenz system data...")
    time_points, trajectory = generate_lorenz_data(
        duration=100.0,
        dt_sampling=0.01
    )
    
    print(f"Generated {len(time_points)} time points with shape {trajectory.shape}")
    
    # Save data to HDF5 file
    file_path = 'data/lorenz_data.h5'
    save_data_hdf5(time_points, trajectory, file_path)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    main() 