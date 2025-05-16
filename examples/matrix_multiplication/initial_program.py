"""
Initial implementation of tensor decomposition for matrix multiplication

This file implements a framework for discovering efficient matrix multiplication
algorithms through tensor decomposition, as described in the AlphaEvolve paper.

When matrix multiplication is viewed as a 3D tensor problem, finding a low-rank
decomposition of this tensor directly translates to finding an algorithm with
fewer scalar multiplications.
"""
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Callable


# EVOLVE-BLOCK-START
class TensorDecomposition:
    """
    Framework for finding low-rank tensor decompositions for matrix multiplication.
    
    This implements a basic gradient-based optimization approach to find
    decompositions that represent efficient matrix multiplication algorithms.
    """
    
    def __init__(self, target_shape: Tuple[int, int, int], rank: int, config: Dict[str, Any] = None):
        """
        Initialize the tensor decomposition framework.
        
        Args:
            target_shape: Tuple (m, n, p) for the matrix multiplication problem mxn * nxp
            rank: Rank of the decomposition to find (number of terms in the decomposition)
            config: Configuration dictionary with various parameters
        """
        self.m, self.n, self.p = target_shape
        self.rank = rank
        
        # Default configuration
        self.config = {
            "training_steps": 5000,
            "learning_rate": 0.1,
            "init_scale": 0.5,
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Set device (use GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create target tensor representing matrix multiplication
        self.target_tensor = self._create_target_tensor().to(self.device)
    
    def _create_target_tensor(self) -> torch.Tensor:
        """
        Create the target tensor representing matrix multiplication.
        
        For matrix multiplication C = A * B, the tensor T has elements:
        T[i, j, k] = 1 if C[i, j] depends on A[i, k] and B[k, j], otherwise 0
        
        Returns:
            3D tensor of shape (m, p, n) representing matrix multiplication
        """
        tensor = torch.zeros((self.m, self.p, self.n), dtype=torch.cfloat)
        
        for i in range(self.m):
            for j in range(self.p):
                for k in range(self.n):
                    tensor[i, j, k] = 1.0
        
        return tensor
    
    def _initialize_decomposition(self, seed: int = 42) -> List[torch.Tensor]:
        """
        Initialize the decomposition parameters.
        
        For matrix multiplication tensor of shape (m, p, n), we need
        rank terms, each consisting of 3 components:
        - U factors of shape (m, rank)
        - V factors of shape (p, rank)
        - W factors of shape (n, rank)
        
        Args:
            seed: Random seed for initialization
            
        Returns:
            List of decomposition factors [u_factors, v_factors, w_factors]
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Initialize factors with normal distribution
        init_scale = self.config["init_scale"]
        
        # Complex initialization
        u_factors = torch.complex(
            torch.randn(self.m, self.rank) * init_scale,
            torch.randn(self.m, self.rank) * init_scale
        ).to(self.device)
        
        v_factors = torch.complex(
            torch.randn(self.p, self.rank) * init_scale,
            torch.randn(self.p, self.rank) * init_scale
        ).to(self.device)
        
        w_factors = torch.complex(
            torch.randn(self.n, self.rank) * init_scale,
            torch.randn(self.n, self.rank) * init_scale
        ).to(self.device)
        
        # Make parameters require gradient
        u_factors.requires_grad_(True)
        v_factors.requires_grad_(True)
        w_factors.requires_grad_(True)
        
        return [u_factors, v_factors, w_factors]
    
    def _decomposition_to_tensor(self, decomposition: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert decomposition factors back to the full tensor.
        
        Args:
            decomposition: List of decomposition factors [u_factors, v_factors, w_factors]
            
        Returns:
            Reconstructed tensor of shape (m, p, n)
        """
        u_factors, v_factors, w_factors = decomposition
        
        # For each rank, we compute the outer product of the corresponding vectors
        reconstructed = torch.zeros((self.m, self.p, self.n), dtype=torch.cfloat).to(self.device)
        
        # Batch implementation
        for r in range(self.rank):
            u = u_factors[:, r].reshape(self.m, 1, 1)
            v = v_factors[:, r].reshape(1, self.p, 1)
            w = w_factors[:, r].reshape(1, 1, self.n)
            
            outer_product = u * v * w
            reconstructed += outer_product
        
        return reconstructed
    
    def _loss_fn(self, decomposition: List[torch.Tensor], global_step: int = 0) -> torch.Tensor:
        """
        Computes loss on learned decomposition.
        
        Args:
            decomposition: List of decomposition factors
            global_step: Current optimization step
            
        Returns:
            Loss value
        """
        # Compute reconstruction loss
        rec_tensor = self._decomposition_to_tensor(decomposition)
        
        # Define the loss as the L2 reconstruction error
        rec_loss = l2_loss_complex(self.target_tensor, rec_tensor)
        
        return rec_loss
    
    def optimize(self, rng_seed: int = 42) -> Dict[str, Any]:
        """
        Run optimization to find a low-rank decomposition.
        
        Args:
            rng_seed: Random seed for initialization
            
        Returns:
            Dictionary with optimization results
        """
        # Initialize random state
        torch.manual_seed(rng_seed)
        
        # Initialize decomposition
        decomposition = self._initialize_decomposition(seed=rng_seed)
        
        # Create optimizer
        optimizer = torch.optim.Adam(decomposition, lr=self.config["learning_rate"])
        
        # Training loop
        training_steps = self.config["training_steps"]
        best_loss = float('inf')
        best_decomposition = None
        
        for step in range(training_steps):
            # Forward pass
            loss = self._loss_fn(decomposition, step)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track progress
            if step % 500 == 0:
                with torch.no_grad():
                    print(f"Step {step}, Loss: {loss.item()}")
            
            # Track best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_decomposition = [f.detach().clone() for f in decomposition]
        
        # Use best decomposition
        if best_decomposition is not None:
            decomposition = best_decomposition
        
        # Return final results
        with torch.no_grad():
            reconstruction = self._decomposition_to_tensor(decomposition)
            reconstruction_error = l2_loss_complex(self.target_tensor, reconstruction).item()
            
            # Extract algorithm from decomposition
            algorithm = self._extract_algorithm(decomposition)
            
            return {
                "decomposition": [f.cpu().numpy() for f in decomposition],
                "reconstruction": reconstruction.cpu().numpy(),
                "reconstruction_error": float(reconstruction_error),
                "algorithm": algorithm,
            }
    
    def _extract_algorithm(self, decomposition: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Extract a concrete matrix multiplication algorithm from the decomposition.
        
        Args:
            decomposition: List of decomposition factors
            
        Returns:
            Dictionary describing the matrix multiplication algorithm
        """
        u_factors, v_factors, w_factors = decomposition
        
        # Round values to identify potential patterns (for interpretability)
        # This helps in recognizing when the decomposition corresponds to a known algorithm
        
        # Extract scalar multiplications (one per rank)
        multiplications = []
        for r in range(self.rank):
            u_vec = u_factors[:, r].cpu().numpy()
            v_vec = v_factors[:, r].cpu().numpy()
            w_vec = w_factors[:, r].cpu().numpy()
            
            # Each term represents: (u-linear combination of A rows) * (v-linear combination of B columns)
            # This produces one scalar multiplication that contributes to the final result
            multiplications.append({
                "u_coeffs": u_vec.tolist(),
                "v_coeffs": v_vec.tolist(),
                "w_coeffs": w_vec.tolist(),
            })
        
        return {
            "rank": self.rank,
            "shape": (self.m, self.n, self.p),
            "multiplications": multiplications,
        }


def l2_loss_complex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Elementwise L2 loss for complex numbers.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        L2 loss
    """
    return torch.mean(torch.abs(x - y) ** 2)


def sweep():
    """Define hyperparameter sweep configurations."""
    from openevolve.utils import hyper
    
    return hyper.zipit([
        hyper.uniform('init_scale', hyper.interval(0.2, 1.5)),
        hyper.uniform('learning_rate', hyper.interval(0.05, 0.3)),
    ])
# EVOLVE-BLOCK-END


def test_tensor_decomposition():
    """
    Test the tensor decomposition framework on a simple example.
    """
    # Test 2x2x2 matrix multiplication (Strassen's algorithm should give rank 7)
    td = TensorDecomposition(target_shape=(2, 2, 2), rank=7)
    result = td.optimize(rng_seed=42)
    
    # Check reconstruction error
    reconstruction_error = result["reconstruction_error"]
    print(f"Reconstruction error: {reconstruction_error}")
    
    # If error is small, decomposition is valid
    is_valid = reconstruction_error < 1e-3
    print(f"Found valid decomposition: {is_valid}")
    
    return is_valid


if __name__ == "__main__":
    # Test tensor decomposition
    test_tensor_decomposition()
