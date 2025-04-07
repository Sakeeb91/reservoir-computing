import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


class ReservoirBuilder:
    """
    Constructs the reservoir matrices (Win and A) for Reservoir Computing.
    """
    
    def __init__(self, Dr, D, rho, degree, sigma_input):
        """
        Initialize the reservoir builder.
        
        Args:
            Dr: Reservoir size (number of neurons)
            D: Dimension of the input data
            rho: Desired spectral radius for matrix A
            degree: Average degree for sparse matrix A
            sigma_input: Scaling factor for Win elements
        """
        self.Dr = Dr
        self.D = D
        self.rho = rho
        self.degree = degree
        self.sigma_input = sigma_input
        
    def build_input_matrix(self):
        """
        Build the input matrix Win (Dr x D).
        
        Returns:
            Win: Input matrix
        """
        # Generate random input matrix with values in [-sigma_input, sigma_input]
        Win = 2 * self.sigma_input * np.random.rand(self.Dr, self.D) - self.sigma_input
        
        return Win
    
    def build_reservoir_matrix(self):
        """
        Build the sparse reservoir matrix A (Dr x Dr) with desired spectral radius.
        
        Returns:
            A: Reservoir matrix
        """
        # Calculate probability of connection based on desired average degree
        p = self.degree / (self.Dr - 1)
        
        # Build sparse random matrix with random weights
        A = sparse.random(self.Dr, self.Dr, density=p, format='csr', data_rvs=np.random.randn)
        
        # Calculate spectral radius
        eigenvalues = splinalg.eigs(A, k=1, which='LM', return_eigenvectors=False)
        current_spectral_radius = np.abs(eigenvalues[0])
        
        # Rescale to desired spectral radius
        A = A * (self.rho / current_spectral_radius)
        
        return A
    
    def build_matrices(self):
        """
        Build both input and reservoir matrices.
        
        Returns:
            Tuple of (Win, A)
        """
        Win = self.build_input_matrix()
        A = self.build_reservoir_matrix()
        
        return Win, A 