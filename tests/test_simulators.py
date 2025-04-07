#!/usr/bin/env python3
"""
Tests for chaotic simulators.
"""

import unittest
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path to import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulators import SimulatorFactory, LorenzSimulator, KSSimulator


class TestLorenzSimulator(unittest.TestCase):
    """Tests for the Lorenz simulator."""
    
    def setUp(self):
        """Set up the test fixture."""
        self.simulator = LorenzSimulator()
        self.params = {
            'sigma': 10.0,
            'rho': 28.0,
            'beta': 8/3
        }
        
    def test_initialization(self):
        """Test simulator initialization."""
        state = self.simulator.initialize(self.params)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), 3)
        
        # Test with fixed initial state
        initial_state = np.array([1.0, 2.0, 3.0])
        state = self.simulator.initialize(self.params, initial_state)
        np.testing.assert_array_equal(state, initial_state)
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Missing parameter
        params = {'sigma': 10.0, 'rho': 28.0}  # Missing beta
        with self.assertRaises(ValueError):
            self.simulator.initialize(params)
            
    def test_transient_simulation(self):
        """Test transient simulation."""
        self.simulator.initialize(self.params)
        state = self.simulator.run_transient(0.1, 0.01)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), 3)
        
    def test_record_simulation(self):
        """Test recorded simulation."""
        self.simulator.initialize(self.params)
        time_points, trajectory = self.simulator.run_record(0.1, 0.01, 0.02)
        
        # Check time points
        self.assertIsInstance(time_points, np.ndarray)
        self.assertGreater(len(time_points), 0)
        
        # Check trajectory
        self.assertIsInstance(trajectory, np.ndarray)
        self.assertEqual(trajectory.shape[0], len(time_points))
        self.assertEqual(trajectory.shape[1], 3)
        
        # Check time step
        dt = time_points[1] - time_points[0]
        self.assertAlmostEqual(dt, 0.02, places=4)
        
    def test_chaotic_behavior(self):
        """Test that the system exhibits chaotic behavior."""
        # Initialize with two slightly different initial conditions
        initial_state1 = np.array([0.1, 0.1, 0.1])
        initial_state2 = np.array([0.100001, 0.1, 0.1])  # Small perturbation
        
        self.simulator.initialize(self.params, initial_state1)
        _, trajectory1 = self.simulator.run_record(5.0, 0.01, 0.1)
        
        self.simulator.initialize(self.params, initial_state2)
        _, trajectory2 = self.simulator.run_record(5.0, 0.01, 0.1)
        
        # Check that trajectories diverge
        # (The final states should be different due to chaos)
        self.assertFalse(np.allclose(trajectory1[-1], trajectory2[-1]))


class TestKSSimulator(unittest.TestCase):
    """Tests for the Kuramoto-Sivashinsky simulator."""
    
    def setUp(self):
        """Set up the test fixture."""
        self.simulator = KSSimulator()
        self.params = {
            'L': 60.0,
            'Q': 64
        }
        
    def test_initialization(self):
        """Test simulator initialization."""
        state = self.simulator.initialize(self.params)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), self.params['Q'])
        
        # Test with fixed initial state
        initial_state = np.random.randn(self.params['Q'])
        state = self.simulator.initialize(self.params, initial_state)
        np.testing.assert_array_equal(state, initial_state)
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Missing parameter
        params = {'L': 60.0}  # Missing Q
        with self.assertRaises(ValueError):
            self.simulator.initialize(params)
            
    def test_transient_simulation(self):
        """Test transient simulation."""
        self.simulator.initialize(self.params)
        state = self.simulator.run_transient(0.1, 0.01)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), self.params['Q'])
        
    def test_record_simulation(self):
        """Test recorded simulation."""
        self.simulator.initialize(self.params)
        time_points, trajectory = self.simulator.run_record(0.1, 0.01, 0.02)
        
        # Check time points
        self.assertIsInstance(time_points, np.ndarray)
        self.assertGreater(len(time_points), 0)
        
        # Check trajectory
        self.assertIsInstance(trajectory, np.ndarray)
        self.assertEqual(trajectory.shape[0], len(time_points))
        self.assertEqual(trajectory.shape[1], self.params['Q'])
        
        # Check time step
        dt = time_points[1] - time_points[0]
        self.assertAlmostEqual(dt, 0.02, places=4)


class TestSimulatorFactory(unittest.TestCase):
    """Tests for the simulator factory."""
    
    def test_lorenz_creation(self):
        """Test creating a Lorenz simulator."""
        factory = SimulatorFactory()
        simulator = factory.create_simulator('lorenz')
        self.assertIsInstance(simulator, LorenzSimulator)
        
    def test_ks_creation(self):
        """Test creating a KS simulator."""
        factory = SimulatorFactory()
        simulator = factory.create_simulator('kuramoto_sivashinsky')
        self.assertIsInstance(simulator, KSSimulator)
        
        # Test the shortened alias
        simulator = factory.create_simulator('ks')
        self.assertIsInstance(simulator, KSSimulator)
        
    def test_invalid_system(self):
        """Test creating an invalid system type."""
        factory = SimulatorFactory()
        with self.assertRaises(ValueError):
            factory.create_simulator('invalid_system')


if __name__ == '__main__':
    unittest.main() 