import numpy as np
from scipy.integrate import solve_ivp
from .base import ISimulator


class LorenzSimulator(ISimulator):
    """
    Simulator for the Lorenz '63 system.
    
    The Lorenz system is defined by:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    
    def __init__(self):
        """Initialize the Lorenz simulator"""
        self.params = None
        self.state = None
        
    def initialize(self, params, initial_state=None):
        """
        Initialize the simulator with parameters and initial state.
        
        Args:
            params: Dictionary with 'sigma', 'rho', and 'beta' parameters
            initial_state: Initial state vector [x, y, z] or None to generate random
        """
        # Validate required parameters
        required_params = ['sigma', 'rho', 'beta']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
                
        self.params = params
        
        # Set initial state
        if initial_state is not None:
            if len(initial_state) != 3:
                raise ValueError("Initial state must be a 3D vector [x, y, z]")
            self.state = np.array(initial_state, dtype=float)
        else:
            # Generate random initial state
            self.state = np.random.uniform(-10, 10, 3)
            
        return self.state
    
    def _lorenz_system(self, t, state):
        """
        The Lorenz system ODE function.
        
        Args:
            t: Time point (not used but required by scipy.integrate)
            state: State vector [x, y, z]
            
        Returns:
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        sigma = self.params['sigma']
        rho = self.params['rho']
        beta = self.params['beta']
        
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def run_transient(self, duration, dt_integration):
        """
        Run the simulation for a transient period.
        
        Args:
            duration: Duration of the transient simulation
            dt_integration: Time step for numerical integration
            
        Returns:
            The state of the system after the transient period
        """
        if self.state is None:
            raise ValueError("Simulator not initialized. Call initialize() first.")
            
        # Calculate number of steps
        t_span = (0, duration)
        
        # Run the simulation
        solution = solve_ivp(
            self._lorenz_system,
            t_span,
            self.state,
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            max_step=dt_integration
        )
        
        # Update state with the final state
        self.state = solution.y[:, -1]
        
        return self.state
    
    def run_record(self, duration, dt_integration, dt_sampling):
        """
        Run the simulation and record the trajectory.
        
        Args:
            duration: Duration of the recorded simulation
            dt_integration: Time step for numerical integration
            dt_sampling: Time step for sampling the trajectory
            
        Returns:
            Tuple of (time_points, trajectory)
        """
        if self.state is None:
            raise ValueError("Simulator not initialized. Call initialize() first.")
            
        # Calculate number of steps and create output time points
        t_span = (0, duration)
        t_eval = np.arange(0, duration, dt_sampling)
        
        # Run the simulation
        solution = solve_ivp(
            self._lorenz_system,
            t_span,
            self.state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
            max_step=dt_integration
        )
        
        # Update state with the final state
        self.state = solution.y[:, -1]
        
        # Return time points and trajectory
        # Transpose to get shape (n_timepoints, n_dimensions)
        return solution.t, solution.y.T
    
    def get_state_dimension(self):
        """Returns the dimension of the system's state vector."""
        return 3 