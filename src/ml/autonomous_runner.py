import numpy as np
from scipy import sparse


class AutonomousRunner:
    """
    Runs the trained reservoir in autonomous mode for prediction/generation.
    """
    
    def __init__(self, Win, A, Wout, use_nonlinear_output=False, nonlinear_indices=None):
        """
        Initialize the autonomous runner.
        
        Args:
            Win: Input matrix (Dr x D)
            A: Reservoir matrix (Dr x Dr)
            Wout: Output weights (D x Dr)
            use_nonlinear_output: Whether to apply nonlinearity to the reservoir states
            nonlinear_indices: Indices of nodes to apply nonlinearity to
        """
        self.Win = Win
        self.A = A
        self.Wout = Wout
        self.use_nonlinear_output = use_nonlinear_output
        self.nonlinear_indices = nonlinear_indices
        
        # Determine system dimensions
        self.Dr = self.Win.shape[0]  # Reservoir size
        self.D = self.Wout.shape[0]  # Data dimension
        
    def apply_nonlinearity(self, r):
        """
        Apply nonlinearity to a subset of reservoir states.
        
        Args:
            r: Reservoir state vector
            
        Returns:
            Modified reservoir state vector with nonlinearity applied
        """
        if self.nonlinear_indices is None or not self.use_nonlinear_output:
            return r
            
        r_out = r.copy()
        r_out[self.nonlinear_indices] = r_out[self.nonlinear_indices]**2
        return r_out
    
    def predict_next_step(self, r):
        """
        Predict the next system state using the current reservoir state.
        
        Args:
            r: Current reservoir state
            
        Returns:
            Tuple of (v_predicted, r_next) where:
                v_predicted: Predicted next input
                r_next: Next reservoir state
        """
        # Apply nonlinearity if needed
        r_out = self.apply_nonlinearity(r) if self.use_nonlinear_output else r
        
        # Predict next input: v_predicted = Wout * r_out
        v_predicted = np.dot(self.Wout, r_out)
        
        # Update reservoir state: r_next = tanh(A * r + Win * v_predicted)
        if sparse.issparse(self.A):
            r_next = np.tanh(self.A.dot(r) + self.Win.dot(v_predicted))
        else:
            r_next = np.tanh(np.dot(self.A, r) + np.dot(self.Win, v_predicted))
        
        return v_predicted, r_next
    
    def run_autonomous(self, initial_r, generation_length):
        """
        Run the reservoir autonomously for a specified number of steps.
        
        Args:
            initial_r: Initial reservoir state
            generation_length: Number of time steps to generate
            
        Returns:
            Tuple of (V_generated, R_generated) where:
                V_generated: Generated time series (generation_length x D)
                R_generated: Sequence of reservoir states (generation_length x Dr)
        """
        # Initialize storage for generated data
        V_generated = np.zeros((generation_length, self.D))
        R_generated = np.zeros((generation_length, self.Dr))
        
        # Set initial state
        r = initial_r.copy()
        
        # Generate time series
        for i in range(generation_length):
            # Predict next value and update reservoir state
            v_predicted, r = self.predict_next_step(r)
            
            # Store results
            V_generated[i] = v_predicted
            R_generated[i] = r
        
        return V_generated, R_generated 