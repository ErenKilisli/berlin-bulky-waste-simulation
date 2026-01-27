"""
Utility Functions

Helper functions for:
- Youth ratio calculations
- Reuse probability calculations
- CO2 emission estimates
- Data validation
- Statistical utilities
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_youth_ratio(
    age_columns: pd.DataFrame,
    total_population: Union[pd.Series, float]
) -> pd.Series:
    """
    Calculate youth ratio (18-45 age group) from population data.
    
    Args:
        age_columns: DataFrame with age group columns
        total_population: Total population (Series or scalar)
        
    Returns:
        Series with youth ratios
    """
    youth_cols = [col for col in age_columns.columns if any(
        str(age) in col for age in range(18, 46)
    )]
    
    if not youth_cols:
        logger.warning("No youth age columns found, using default 0.30")
        return pd.Series(0.30, index=age_columns.index)
    
    youth_population = age_columns[youth_cols].sum(axis=1)
    youth_ratio = youth_population / total_population
    
    # Clip to reasonable range
    youth_ratio = youth_ratio.clip(0.10, 0.50)
    
    return youth_ratio


def calculate_reuse_probability(
    youth_ratio: float,
    scenario_multiplier: float,
    item_attractiveness: float,
    natural_reuse_prob: float = 0.05
) -> float:
    """
    Calculate reuse probability using the model formula.
    
    P_reuse = (Youth Ratio × Scenario Multiplier) × Item Attractiveness
    
    Args:
        youth_ratio: District youth ratio (0.0-1.0)
        scenario_multiplier: Scenario adoption factor (0.0-1.0)
        item_attractiveness: Item category attractiveness (0.0-1.0)
        natural_reuse_prob: Baseline reuse without app
        
    Returns:
        Reuse probability (0.0-1.0)
    """
    if scenario_multiplier == 0.0:
        # Baseline scenario
        return natural_reuse_prob
    
    p_reuse = (youth_ratio * scenario_multiplier) * item_attractiveness
    
    # Ensure within bounds
    p_reuse = max(natural_reuse_prob, min(p_reuse, 0.95))
    
    return p_reuse


def estimate_co2_emissions(
    num_collections: int,
    co2_per_collection: float = 2.5,
    vehicle_efficiency: float = 1.0
) -> float:
    """
    Estimate CO2 emissions from waste collection operations.
    
    Args:
        num_collections: Number of collection events
        co2_per_collection: CO2 kg per collection (default: 2.5 kg)
        vehicle_efficiency: Efficiency factor (1.0 = standard)
        
    Returns:
        Total CO2 emissions in kg
    """
    total_co2 = num_collections * co2_per_collection * vehicle_efficiency
    
    return max(0.0, total_co2)


def validate_probability(value: float, name: str = "probability") -> float:
    """
    Validate that a value is a valid probability (0.0-1.0).
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Returns:
        Validated probability
        
    Raises:
        ValueError: If value is not in [0, 1]
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
    
    return value


def validate_positive(value: float, name: str = "value") -> float:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    
    return value


def calculate_confidence_interval(
    data: Union[List, np.ndarray, pd.Series],
    confidence: float = 0.95
) -> tuple:
    """
    Calculate confidence interval for data.
    
    Args:
        data: Numeric data
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    from scipy import stats
    
    data_array = np.array(data)
    n = len(data_array)
    mean = np.mean(data_array)
    std_err = stats.sem(data_array)
    
    ci = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - ci, mean + ci


def exponential_arrival_times(rate: float, n_events: int, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate exponential inter-arrival times.
    
    Args:
        rate: Arrival rate (events per time unit)
        n_events: Number of events to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of arrival times
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    inter_arrivals = np.random.exponential(1.0 / rate, size=n_events)
    arrival_times = np.cumsum(inter_arrivals)
    
    return arrival_times


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.25)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string (e.g., "25.0%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_large_number(value: float, decimals: int = 0) -> str:
    """
    Format large numbers with thousand separators.
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:,.{decimals}f}"


class SimulationTimer:
    """
    Context manager for timing simulation runs.
    """
    
    def __init__(self, name: str = "Simulation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.info(f"{self.name} started...")
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {elapsed:.2f} seconds")


def save_simulation_metadata(
    output_path: str,
    parameters: dict,
    results_summary: dict
):
    """
    Save simulation metadata to JSON file.
    
    Args:
        output_path: Path to save metadata
        parameters: Simulation parameters
        results_summary: Summary of results
    """
    import json
    from datetime import datetime
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "parameters": parameters,
        "results_summary": results_summary
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {output_path}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...\n")
    
    # Test youth ratio calculation
    youth_ratio = calculate_reuse_probability(
        youth_ratio=0.40,
        scenario_multiplier=0.5,
        item_attractiveness=0.8
    )
    print(f"Reuse probability: {format_percentage(youth_ratio)}")
    
    # Test CO2 calculation
    co2 = estimate_co2_emissions(num_collections=100)
    print(f"CO2 emissions: {format_large_number(co2)} kg")
    
    # Test confidence interval
    data = np.random.normal(0.30, 0.05, 100)
    mean, lower, upper = calculate_confidence_interval(data)
    print(f"CI: {format_percentage(mean)} ({format_percentage(lower)} - {format_percentage(upper)})")