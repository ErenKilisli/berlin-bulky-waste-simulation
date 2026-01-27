"""
Berlin Bulky Waste Simulation Package

A Discrete Event Simulation framework for analyzing illegal bulky waste
management in Berlin and evaluating digital reuse platform interventions.
"""

__version__ = "0.1.0"
__author__ = "BSBI MSc IT Management Team"

from .data_loader import DataLoader
from .simulation import BulkyWasteSimulation, WasteItem
from .analysis import ResultsAnalyzer
from .utils import (
    calculate_youth_ratio,
    calculate_reuse_probability,
    estimate_co2_emissions,
)

__all__ = [
    "DataLoader",
    "BulkyWasteSimulation",
    "WasteItem",
    "ResultsAnalyzer",
    "calculate_youth_ratio",
    "calculate_reuse_probability",
    "estimate_co2_emissions",
]