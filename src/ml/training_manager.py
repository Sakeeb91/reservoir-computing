import numpy as np
from scipy import sparse
from scipy import linalg


class TrainingManager:
    """
    Manages the training phase of the Reservoir Computing system.
    """
    
    def __init__(self, Win, A, washout_duration, ridge_beta=1e-6, use_nonlinear_output=False, nonlinear_fraction=0.5):
        """
        Initialize the training manager.
        
        Args:
            Win: Input matrix (Dr x D)
            A: Reservoir matrix (Dr x Dr)
            washout_duration: Initial period of reservoir states to discard
            ridge_beta: Tikhonov regularization parameter
            use_nonlinear_output: Whether to apply nonlinearity to the reservoir states
            nonlinear_fraction: Fraction of nodes to apply nonlinearity to
        """
        self.Win = Win
        self.A = A
        self.washout_duration = washout_duration
        self.ridge_beta = ridge_beta
        self.use_nonlinear_output = use_nonlinear_output
        self.nonlinear_fraction = nonlinear_fraction
        
        # Determine reservoir size
        self.Dr = self.Win.shape[0]
        
        # Initialize reservoir state to zero
        self.r = np.zeros(self.Dr)
        
        # If using nonlinear output, determine which nodes to apply nonlinearity to
        if self.use_nonlinear_output:
            nonlinear_count = int(self.Dr * self.nonlinear_fraction)
            self.nonlinear_indices = np.random.choice(self.Dr, nonlinear_count, replace=False)
        else:
            self.nonlinear_indices = None
    
    def drive_reservoir(self, u):
        """
        Update the reservoir state with input u(t).
        
        Args:
            u: Input vector at time t
            
        Returns:
            Updated reservoir state r(t+1)
        """
        # Calculate r(t+1) = tanh(A*r(t) + Win*u(t))
        if sparse.issparse(self.A):
            self.r = np.tanh(self.A.dot(self.r) + self.Win.dot(u))
        else:
            self.r = np.tanh(np.dot(self.A, self.r) + np.dot(self.Win, u))
        
        return self.r
    
    def apply_nonlinearity(self, r):
        """
        Apply nonlinearity to a subset of reservoir states.
        
        Args:
            r: Reservoir state vector
            
        Returns:
            Modified reservoir state vector with nonlinearity applied
        """
        r_out = r.copy()
        r_out[self.nonlinear_indices] = r_out[self.nonlinear_indices]**2
        return r_out
    
    def train(self, U, Yd):
        """
        Train the reservoir computer using input sequence U and target sequence Yd.
        
        Args:
            U: Input sequence (T x D)
            Yd: Target sequence (T x D)
            
        Returns:
            Output weights matrix Wout (D x Dr or D x Dr')
        """
        T = U.shape[0]  # Number of time steps
        D = Yd.shape[1]  # Output dimension
        
        # Reset reservoir state
        self.r = np.zeros(self.Dr)
        
        # Warm up the reservoir (washout period)
        for i in range(self.washout_duration):
            t = i % T  # Loop through the input if washout_duration > T
            self.drive_reservoir(U[t])
        
        # Collect reservoir states
        R_collected = np.zeros((T, self.Dr))
        for i in range(T):
            self.drive_reservoir(U[i])
            R_collected[i] = self.r
        
        # Apply nonlinearity to states if requested
        if self.use_nonlinear_output and self.nonlinear_indices is not None:
            R_out = np.zeros_like(R_collected)
            for i in range(T):
                R_out[i] = self.apply_nonlinearity(R_collected[i])
            R_collected = R_out
        
        # Perform ridge regression: Wout = Yd^T * R * (R^T * R + beta * I)^(-1)
        # Transpose matrices for calculation
        R_collected = R_collected.T  # Now Dr x T
        Yd = Yd.T  # Now D x T
        
        # Compute R * R^T (Dr x Dr)
        if T >= self.Dr:
            # Direct calculation for large datasets
            RRT = np.dot(R_collected, R_collected.T)
            
            # Add regularization
            RRT += self.ridge_beta * np.eye(self.Dr)
            
            # Compute Wout
            Wout = np.dot(Yd, np.dot(R_collected.T, linalg.inv(RRT)))
        else:
            # For small datasets, use the more efficient dual formulation
            RTR = np.dot(R_collected.T, R_collected)
            RTR += self.ridge_beta * np.eye(T)
            Wout = np.dot(np.dot(Yd, linalg.inv(RTR)), R_collected.T)
        
        return Wout 