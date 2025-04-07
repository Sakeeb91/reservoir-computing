import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from .base import ISimulator


class KSSimulator(ISimulator):
    """
    Simulator for the Kuramoto-Sivashinsky equation.
    
    The Kuramoto-Sivashinsky equation is a nonlinear PDE:
    u_t + u*u_x + u_xx + u_xxxx = 0
    
    We use a pseudo-spectral method with periodic boundary conditions.
    """
    
    def __init__(self):
        """Initialize the Kuramoto-Sivashinsky simulator"""
        self.params = None
        self.state = None
        self.k = None  # Wavenumbers
        self.L = None  # Domain size
        self.x = None  # Spatial grid
        self.Q = None  # Number of grid points
        self.linear_term = None  # Pre-computed linear term for spectral method
        
    def initialize(self, params, initial_state=None):
        """
        Initialize the simulator with parameters and initial state.
        
        Args:
            params: Dictionary with 'L' (domain size) and 'Q' (number of grid points)
            initial_state: Initial state vector of size Q or None to generate random
        """
        # Validate required parameters
        required_params = ['L', 'Q']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        self.L = params['L']
        self.Q = params['Q']
        self.params = params
        
        # Set up spatial grid
        self.x = np.linspace(0, self.L, self.Q, endpoint=False)
        
        # Set up wavenumbers for spectral method
        self.k = 2 * np.pi * fftfreq(self.Q, self.L / self.Q)
        
        # Precompute linear term for spectral method
        self.linear_term = -self.k**2 - self.k**4
        
        # Set initial state
        if initial_state is not None:
            if len(initial_state) != self.Q:
                raise ValueError(f"Initial state must have length {self.Q}")
            self.state = np.array(initial_state, dtype=float)
        else:
            # Generate random initial state with small amplitude
            self.state = 0.1 * np.random.randn(self.Q)
            
        return self.state
    
    def _ks_rhs(self, u, u_hat):
        """
        Compute the right-hand side of the KS equation in Fourier space.
        
        Args:
            u: Real space state
            u_hat: Fourier space state
            
        Returns:
            Right-hand side in Fourier space
        """
        # Compute nonlinear term: -0.5 * d(u^2)/dx
        nonlinear = -0.5 * 1j * self.k * fft(u**2)
        
        # Compute RHS: linear + nonlinear terms
        return self.linear_term * u_hat + nonlinear
    
    def _etdrk4_step(self, u, dt):
        """
        Perform a single ETDRK4 time step.
        
        Args:
            u: Current state in real space
            dt: Time step
            
        Returns:
            New state in real space
        """
        # Precompute operators for ETDRK4 scheme
        E = np.exp(self.linear_term * dt)
        E_2 = np.exp(self.linear_term * dt/2)
        
        M = 16  # Number of points for complex means
        r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
        L = dt * self.linear_term
        
        # Calculate contour integrals
        LR = L[:, np.newaxis] + r
        Q = dt * np.mean(((np.exp(LR/2) - 1) / LR), axis=1)
        f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=1)
        f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
        f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)
        
        # Calculate u_hat (Fourier transform of u)
        u_hat = fft(u)
        
        # First step
        a = E_2 * u_hat + Q * self._ks_rhs(u, u_hat)
        a_real = np.real(ifft(a))
        
        # Second step
        b = E_2 * u_hat + Q * self._ks_rhs(a_real, a)
        b_real = np.real(ifft(b))
        
        # Third step
        c = E_2 * a + Q * (2 * self._ks_rhs(b_real, b) - self._ks_rhs(u, u_hat))
        c_real = np.real(ifft(c))
        
        # Fourth step
        u_hat_new = E * u_hat + f1 * self._ks_rhs(u, u_hat) + 2 * f2 * (self._ks_rhs(a_real, a) + self._ks_rhs(b_real, b)) + f3 * self._ks_rhs(c_real, c)
        
        # Transform back to real space
        return np.real(ifft(u_hat_new))
    
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
        n_steps = int(duration / dt_integration)
        
        # Run the simulation
        u = self.state.copy()
        for _ in range(n_steps):
            u = self._etdrk4_step(u, dt_integration)
        
        # Update state with the final state
        self.state = u
        
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
            
        # Calculate number of integration and sampling steps
        n_steps = int(duration / dt_integration)
        n_samples = int(duration / dt_sampling)
        samples_per_step = max(1, int(dt_sampling / dt_integration))
        
        # Create output arrays
        time_points = np.linspace(0, duration, n_samples)
        trajectory = np.zeros((n_samples, self.Q))
        
        # Run the simulation
        u = self.state.copy()
        sample_idx = 0
        trajectory[0] = u  # Record initial state
        
        for step in range(1, n_steps + 1):
            u = self._etdrk4_step(u, dt_integration)
            
            # Record state at sampling points
            if step % samples_per_step == 0 and sample_idx < n_samples - 1:
                sample_idx += 1
                trajectory[sample_idx] = u
        
        # Update state with the final state
        self.state = u
        
        return time_points, trajectory
    
    def get_state_dimension(self):
        """Returns the dimension of the system's state vector."""
        return self.Q 