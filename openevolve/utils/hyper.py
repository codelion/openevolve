"""
Hyperparameter utilities for tensor decomposition and other optimization problems.

This module provides utilities for hyperparameter sweeping and sampling, which are
essential for the tensor decomposition approach in the AlphaEvolve paper.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Callable


class Interval:
    """
    Represents a continuous interval [min_val, max_val].
    """
    
    def __init__(self, min_val: float, max_val: float):
        """
        Initialize an interval.
        
        Args:
            min_val: Minimum value of the interval
            max_val: Maximum value of the interval
        """
        self.min_val = min_val
        self.max_val = max_val
    
    def sample(self) -> float:
        """
        Sample a value uniformly from the interval.
        
        Returns:
            Random value in [min_val, max_val]
        """
        return np.random.uniform(self.min_val, self.max_val)
    
    def __str__(self) -> str:
        return f"[{self.min_val}, {self.max_val}]"


class DiscreteInterval:
    """
    Represents a discrete interval [min_val, max_val].
    """
    
    def __init__(self, min_val: int, max_val: int):
        """
        Initialize a discrete interval.
        
        Args:
            min_val: Minimum value of the interval
            max_val: Maximum value of the interval
        """
        self.min_val = min_val
        self.max_val = max_val
    
    def sample(self) -> int:
        """
        Sample a value uniformly from the discrete interval.
        
        Returns:
            Random integer in [min_val, max_val]
        """
        return np.random.randint(self.min_val, self.max_val + 1)
    
    def __str__(self) -> str:
        return f"[{self.min_val}, {self.max_val}]"


class Choice:
    """
    Represents a choice from a list of values.
    """
    
    def __init__(self, values: List[Any]):
        """
        Initialize a choice.
        
        Args:
            values: List of values to choose from
        """
        self.values = values
    
    def sample(self) -> Any:
        """
        Sample a value uniformly from the list of values.
        
        Returns:
            Random value from the list
        """
        return np.random.choice(self.values)
    
    def __str__(self) -> str:
        return f"{self.values}"


def interval(min_val: float, max_val: float) -> Interval:
    """
    Create a continuous interval.
    
    Args:
        min_val: Minimum value of the interval
        max_val: Maximum value of the interval
        
    Returns:
        Interval object
    """
    return Interval(min_val, max_val)


def discrete_interval(min_val: int, max_val: int) -> DiscreteInterval:
    """
    Create a discrete interval.
    
    Args:
        min_val: Minimum value of the interval
        max_val: Maximum value of the interval
        
    Returns:
        DiscreteInterval object
    """
    return DiscreteInterval(min_val, max_val)


def choice(values: List[Any]) -> Choice:
    """
    Create a choice.
    
    Args:
        values: List of values to choose from
        
    Returns:
        Choice object
    """
    return Choice(values)


def uniform(name: str, interval: Interval) -> Dict[str, Callable[[], float]]:
    """
    Create a uniform distribution over an interval.
    
    Args:
        name: Name of the hyperparameter
        interval: Interval to sample from
        
    Returns:
        Dictionary mapping hyperparameter name to sampling function
    """
    return {name: interval.sample}


def log_uniform(name: str, interval: Interval) -> Dict[str, Callable[[], float]]:
    """
    Create a log-uniform distribution over an interval.
    
    Args:
        name: Name of the hyperparameter
        interval: Interval to sample from
        
    Returns:
        Dictionary mapping hyperparameter name to sampling function
    """
    def sample():
        return np.exp(np.random.uniform(np.log(interval.min_val), np.log(interval.max_val)))
    
    return {name: sample}


def categorical(name: str, choice: Choice) -> Dict[str, Callable[[], Any]]:
    """
    Create a categorical distribution over a set of values.
    
    Args:
        name: Name of the hyperparameter
        choice: Choice to sample from
        
    Returns:
        Dictionary mapping hyperparameter name to sampling function
    """
    return {name: choice.sample}


def zipit(dicts: List[Dict[str, Callable[[], Any]]]) -> Callable[[], Dict[str, Any]]:
    """
    Combine multiple hyperparameter dictionaries into a single dictionary.
    
    Args:
        dicts: List of hyperparameter dictionaries
        
    Returns:
        Function that samples from all hyperparameters
    """
    def sample():
        result = {}
        for d in dicts:
            for name, sampler in d.items():
                result[name] = sampler()
        return result
    
    return sample


def grid_search(sweep_configs: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Create a grid search over hyperparameters.
    
    Args:
        sweep_configs: Dictionary mapping hyperparameter names to lists of values
        
    Returns:
        List of hyperparameter configurations
    """
    keys = list(sweep_configs.keys())
    
    if not keys:
        return [{}]
    
    result = []
    
    # Recursive helper function to build grid
    def build_grid(index: int, current: Dict[str, Any]):
        if index == len(keys):
            result.append(current.copy())
            return
        
        key = keys[index]
        for value in sweep_configs[key]:
            current[key] = value
            build_grid(index + 1, current)
    
    build_grid(0, {})
    return result


def random_search(sweep_configs: Dict[str, Union[Interval, DiscreteInterval, Choice]], 
                 num_samples: int) -> List[Dict[str, Any]]:
    """
    Create a random search over hyperparameters.
    
    Args:
        sweep_configs: Dictionary mapping hyperparameter names to distributions
        num_samples: Number of samples to generate
        
    Returns:
        List of hyperparameter configurations
    """
    result = []
    
    for _ in range(num_samples):
        sample = {}
        for key, dist in sweep_configs.items():
            sample[key] = dist.sample()
        result.append(sample)
    
    return result
