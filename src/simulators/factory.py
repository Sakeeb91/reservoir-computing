from .lorenz import LorenzSimulator
from .kuramoto_sivashinsky import KSSimulator


class SimulatorFactory:
    """
    Factory class for creating simulator instances based on system type.
    """
    
    @staticmethod
    def create_simulator(system_type):
        """
        Create and return a simulator instance for the specified system type.
        
        Args:
            system_type: String identifier of the system ('lorenz' or 'kuramoto_sivashinsky')
            
        Returns:
            An instance of a simulator implementing the ISimulator interface
            
        Raises:
            ValueError: If the system_type is not supported
        """
        system_type = system_type.lower()
        
        if system_type == 'lorenz':
            return LorenzSimulator()
        elif system_type in ['kuramoto_sivashinsky', 'ks']:
            return KSSimulator()
        else:
            supported_systems = ['lorenz', 'kuramoto_sivashinsky', 'ks']
            raise ValueError(f"Unsupported system type: {system_type}. "
                             f"Supported types are: {', '.join(supported_systems)}") 