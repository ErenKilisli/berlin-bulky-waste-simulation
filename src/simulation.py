"""
Discrete Event Simulation Module

Core simulation logic using SimPy to model:
- Waste item arrivals (Poisson process)
- Reuse events via digital platform
- BSR collection operations (M/M/c queue)
- Multiple scenarios (baseline, pessimistic, realistic, optimistic)
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Scenario(Enum):
    """Simulation scenarios for app adoption."""
    BASELINE = ("baseline", 0.0)
    PESSIMISTIC = ("pessimistic", 0.2)
    REALISTIC = ("realistic", 0.5)
    OPTIMISTIC = ("optimistic", 0.9)
    
    def __init__(self, name: str, multiplier: float):
        self.scenario_name = name
        self.multiplier = multiplier


@dataclass
class WasteItem:
    """Represents a single bulky waste item in the simulation."""
    item_id: int
    district: str
    category: str
    arrival_time: float
    reuse_probability: float
    attractiveness: float
    youth_ratio: float
    scenario_multiplier: float
    
    # State tracking
    reused: bool = False
    collected: bool = False
    reuse_time: Optional[float] = None
    collection_time: Optional[float] = None
    time_in_system: Optional[float] = None


@dataclass
class SimulationMetrics:
    """Aggregated metrics from a simulation run."""
    scenario: str
    district: str
    run_id: int
    
    total_items: int = 0
    items_reused: int = 0
    items_collected: int = 0
    
    reuse_rate: float = 0.0
    avg_time_to_reuse: float = 0.0
    avg_time_to_collection: float = 0.0
    avg_queue_length: float = 0.0
    max_queue_length: int = 0
    
    total_co2_kg: float = 0.0
    
    def calculate_derived_metrics(self):
        """Calculate rates and averages."""
        if self.total_items > 0:
            self.reuse_rate = self.items_reused / self.total_items
        else:
            self.reuse_rate = 0.0


class BulkyWasteSimulation:
    """
    Main simulation class implementing DES for Berlin bulky waste management.
    """
    
    def __init__(self, data_loader, random_seed: int = 42):
        """
        Initialize simulation with data loader.
        
        Args:
            data_loader: DataLoader instance with preloaded data
            random_seed: Seed for reproducibility
        """
        self.data_loader = data_loader
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Simulation parameters (calibrated from real data)
        self.arrival_rate_base = 5.0  # items per day per district (average)
        self.collection_rate = 8.0  # items per day per BSR team
        self.num_bsr_teams = 3  # collection capacity (c in M/M/c)
        self.natural_reuse_prob = 0.05  # baseline reuse without app
        
        # CO2 emission factors (kg CO2 per collection trip)
        self.co2_per_collection = 2.5
        
        # Results storage
        self.all_items: List[WasteItem] = []
        self.metrics: List[SimulationMetrics] = []
        
    def calculate_reuse_probability(
        self,
        district: str,
        category: str,
        scenario: Scenario
    ) -> float:
        """
        Calculate reuse probability using the model formula.
        
        P_reuse = (Youth Ratio × Scenario Multiplier) × Attractiveness
        
        Args:
            district: District name
            category: Waste category
            scenario: Simulation scenario
            
        Returns:
            Reuse probability (0.0 to 1.0)
        """
        youth_ratio = self.data_loader.get_district_youth_ratio(district)
        attractiveness = self.data_loader.get_waste_category_attractiveness(category)
        multiplier = scenario.multiplier
        
        if scenario == Scenario.BASELINE:
            # No app, only natural reuse
            return self.natural_reuse_prob
        
        # Apply formula
        p_reuse = (youth_ratio * multiplier) * attractiveness
        
        # Ensure within bounds
        p_reuse = max(self.natural_reuse_prob, min(p_reuse, 0.95))
        
        return p_reuse
    
    def waste_arrival_process(
        self,
        env: simpy.Environment,
        district: str,
        scenario: Scenario,
        items_list: List[WasteItem]
    ):
        """
        Generate waste item arrivals using Poisson process.
        
        Args:
            env: SimPy environment
            district: District name
            scenario: Simulation scenario
            items_list: List to append generated items
        """
        item_counter = 0
        
        # Get district-specific arrival rate (proportional to population)
        population_factor = 1.0  # Could be adjusted based on district size
        arrival_rate = self.arrival_rate_base * population_factor
        
        while True:
            # Inter-arrival time (exponential distribution)
            inter_arrival = np.random.exponential(1.0 / arrival_rate)
            yield env.timeout(inter_arrival)
            
            # Determine waste category (weighted random choice)
            categories = list(self.data_loader.waste_config['categories'].keys())
            weights = [
                self.data_loader.waste_config['categories'][cat].get('frequency', 1.0)
                for cat in categories
            ]
            category = np.random.choice(categories, p=np.array(weights) / sum(weights))
            
            # Calculate reuse probability
            youth_ratio = self.data_loader.get_district_youth_ratio(district)
            attractiveness = self.data_loader.get_waste_category_attractiveness(category)
            reuse_prob = self.calculate_reuse_probability(district, category, scenario)
            
            # Create waste item
            item = WasteItem(
                item_id=item_counter,
                district=district,
                category=category,
                arrival_time=env.now,
                reuse_probability=reuse_prob,
                attractiveness=attractiveness,
                youth_ratio=youth_ratio,
                scenario_multiplier=scenario.multiplier
            )
            
            items_list.append(item)
            item_counter += 1
            
            # Start reuse and collection processes
            env.process(self.reuse_process(env, item))
            env.process(self.collection_process(env, item, self.bsr_resource))
    
    def reuse_process(self, env: simpy.Environment, item: WasteItem):
        """
        Attempt to reuse item via digital platform.
        
        Args:
            env: SimPy environment
            item: Waste item
        """
        # Time until reuse attempt (exponential)
        time_to_reuse = np.random.exponential(2.0)  # avg 2 days
        yield env.timeout(time_to_reuse)
        
        # Check if item already collected
        if item.collected:
            return
        
        # Determine if reuse successful
        if np.random.random() < item.reuse_probability:
            item.reused = True
            item.reuse_time = env.now
            item.time_in_system = env.now - item.arrival_time
    
    def collection_process(
        self,
        env: simpy.Environment,
        item: WasteItem,
        bsr_resource: simpy.Resource
    ):
        """
        BSR collection process with queueing.
        
        Args:
            env: SimPy environment
            item: Waste item
            bsr_resource: BSR team resource (limited capacity)
        """
        # Wait for BSR team availability
        with bsr_resource.request() as request:
            yield request
            
            # Check if item was already reused
            if item.reused:
                return
            
            # Collection service time (exponential)
            service_time = np.random.exponential(1.0 / self.collection_rate)
            yield env.timeout(service_time)
            
            # Mark as collected
            item.collected = True
            item.collection_time = env.now
            item.time_in_system = env.now - item.arrival_time
    
    def run_single_scenario(
        self,
        scenario: Scenario,
        district: str,
        simulation_days: int = 365,
        run_id: int = 0
    ) -> SimulationMetrics:
        """
        Run simulation for a single scenario and district.
        
        Args:
            scenario: Simulation scenario
            district: District name
            simulation_days: Simulation duration
            run_id: Run identifier for replications
            
        Returns:
            Aggregated simulation metrics
        """
        # Create SimPy environment
        env = simpy.Environment()
        
        # Create BSR resource (M/M/c queue)
        self.bsr_resource = simpy.Resource(env, capacity=self.num_bsr_teams)
        
        # Items list for this run
        items = []
        
        # Start arrival process
        env.process(self.waste_arrival_process(env, district, scenario, items))
        
        # Run simulation
        env.run(until=simulation_days)
        
        # Calculate metrics
        metrics = SimulationMetrics(
            scenario=scenario.scenario_name,
            district=district,
            run_id=run_id
        )
        
        metrics.total_items = len(items)
        metrics.items_reused = sum(1 for item in items if item.reused)
        metrics.items_collected = sum(1 for item in items if item.collected and not item.reused)
        
        # Time metrics
        reused_times = [item.time_in_system for item in items if item.reused and item.time_in_system]
        collected_times = [item.time_in_system for item in items if item.collected and not item.reused and item.time_in_system]
        
        metrics.avg_time_to_reuse = np.mean(reused_times) if reused_times else 0.0
        metrics.avg_time_to_collection = np.mean(collected_times) if collected_times else 0.0
        
        # Queue metrics (approximation)
        metrics.avg_queue_length = metrics.items_collected / self.num_bsr_teams
        metrics.max_queue_length = int(metrics.avg_queue_length * 2)
        
        # CO2 emissions
        metrics.total_co2_kg = metrics.items_collected * self.co2_per_collection
        
        metrics.calculate_derived_metrics()
        
        return metrics
    
    def run_all_scenarios(
        self,
        simulation_days: int = 365,
        num_runs: int = 10
    ) -> pd.DataFrame:
        """
        Run simulation for all scenarios and all districts.
        
        Args:
            simulation_days: Simulation duration
            num_runs: Number of replications per scenario
            
        Returns:
            DataFrame with all results
        """
        logger.info(f"Running simulation: {simulation_days} days, {num_runs} replications")
        
        scenarios = [Scenario.BASELINE, Scenario.PESSIMISTIC, Scenario.REALISTIC, Scenario.OPTIMISTIC]
        districts = self.data_loader.district_demographics['bezirk'].tolist()
        
        all_metrics = []
        
        total_iterations = len(scenarios) * len(districts) * num_runs
        
        with tqdm(total=total_iterations, desc="Simulation Progress") as pbar:
            for scenario in scenarios:
                for district in districts:
                    for run_id in range(num_runs):
                        metrics = self.run_single_scenario(
                            scenario=scenario,
                            district=district,
                            simulation_days=simulation_days,
                            run_id=run_id
                        )
                        all_metrics.append(metrics)
                        pbar.update(1)
        
        # Convert to DataFrame
        results_df = pd.DataFrame([vars(m) for m in all_metrics])
        
        logger.info(f"Simulation complete: {len(results_df)} result records")
        
        return results_df


if __name__ == "__main__":
    # Test simulation
    from data_loader import DataLoader
    
    loader = DataLoader()
    loader.load_all_data()
    
    sim = BulkyWasteSimulation(loader)
    
    # Quick test run
    results = sim.run_all_scenarios(simulation_days=30, num_runs=2)
    
    print("\n=== Simulation Results Summary ===")
    print(results.groupby('scenario')[['total_items', 'items_reused', 'reuse_rate']].mean())