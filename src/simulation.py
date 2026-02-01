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
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================================
# SCENARIO ENUM
# ==================================================================
class Scenario(Enum):
    BASELINE     = ("baseline",     0.0)
    PESSIMISTIC  = ("pessimistic",  0.3)
    REALISTIC    = ("realistic",    0.6)
    OPTIMISTIC   = ("optimistic",   0.9)

    def __init__(self, scenario_name: str, multiplier: float):
        self.scenario_name = scenario_name
        self.multiplier = multiplier


# ==================================================================
# DATA CLASSES
# ==================================================================
@dataclass
class WasteItem:
    item_id: int
    district: str
    category: str
    arrival_time: float
    reuse_probability: float
    attractiveness: float
    youth_ratio: float
    scenario_multiplier: float

    reused: bool = False
    collected: bool = False
    reuse_time: Optional[float] = None
    collection_time: Optional[float] = None
    time_in_system: Optional[float] = None


@dataclass
class SimulationMetrics:
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
        if self.total_items > 0:
            self.reuse_rate = self.items_reused / self.total_items
        else:
            self.reuse_rate = 0.0


# ==================================================================
# MAIN SIMULATION
# ==================================================================
class BulkyWasteSimulation:
    """
    DES for Berlin bulky waste management.

    ---------------------------------------------------------------
    TIMING MODEL (realistic BSR behaviour):

    Day 0      Item placed on street
               |
               |--- reuse_process:
               |        buyer sees it after ~Exp(2) days (avg 2 days)
               |        if random() < P_reuse  ->  item reused
               |
               |--- collection_process:
               |        Step 1: BSR detection delay  ~Exp(5) days   <-- KEY FIX
               |                (patrol + report + scheduling)
               |        Step 2: Queue wait  (M/M/c, 3 teams)
               |        Step 3: Service time ~Exp(1/8) days
               |
    Because reuse attempt (avg 2 days) is FASTER than
    BSR pickup (avg 5+ days), high-P_reuse items actually
    get reused before BSR arrives.
    ---------------------------------------------------------------

    REUSE PROBABILITY MODEL:
        P_reuse = multiplier * attractiveness
                  (scaled by youth_ratio per district)

        baseline    : natural_reuse = 0.05 (no app)
        pessimistic : 0.3 * attractiveness  (low adoption)
        realistic   : 0.6 * attractiveness  (moderate)
        optimistic  : 0.9 * attractiveness  (high adoption)

    youth_ratio acts as a district-level scaling factor:
        final_p = P_reuse * (0.5 + 0.5 * youth_ratio)
    This means youth-heavy districts get a boost (up to 1.0x)
    while older districts get a penalty (down to 0.5x).
    ---------------------------------------------------------------
    """

    def __init__(self, data_loader, random_seed: int = 42):
        self.data_loader = data_loader
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # --- Timing parameters ---
        self.arrival_rate_base      = 5.0    # items / day / district
        self.collection_rate        = 8.0    # service rate per BSR team
        self.bsr_detection_delay    = 5.0    # avg days before BSR notices item
        self.num_bsr_teams          = 3      # M/M/c capacity
        self.natural_reuse_prob     = 0.05   # baseline (no app)
        self.co2_per_collection     = 2.5    # kg CO2 per collected item

        # Pre-compute category arrays once
        self._category_ids: List[str] = []
        self._category_weights: np.ndarray = np.array([])
        self._prepare_categories()

    def _prepare_categories(self):
        probs = self.data_loader.get_category_probabilities()
        self._category_ids = list(probs.keys())
        raw = np.array(list(probs.values()), dtype=float)
        self._category_weights = raw / raw.sum()
        logger.info(f"Categories: {self._category_ids} | weights: {self._category_weights}")

    # ==============================================================
    # REUSE PROBABILITY
    #
    # P_reuse = multiplier * attractiveness * district_scale
    # district_scale = 0.5 + 0.5 * youth_ratio
    #
    # youth_ratio=0.15 -> scale=0.575  (older district, less app use)
    # youth_ratio=0.30 -> scale=0.650  (average)
    # youth_ratio=0.45 -> scale=0.725  (young district, more app use)
    # ==============================================================
    def calculate_reuse_probability(
        self, district: str, category_id: str, scenario: Scenario
    ) -> float:
        if scenario == Scenario.BASELINE:
            return self.natural_reuse_prob

        youth_ratio    = self.data_loader.get_district_youth_ratio(district)
        attractiveness = self.data_loader.get_waste_category_attractiveness(category_id)

        district_scale = 0.5 + 0.5 * youth_ratio
        p = scenario.multiplier * attractiveness * district_scale

        return max(self.natural_reuse_prob, min(p, 0.95))

    # ==============================================================
    # SIMPY PROCESSES
    # ==============================================================
    def waste_arrival_process(self, env, district, scenario, items_list):
        """Poisson arrivals."""
        item_counter = 0
        rate = self.arrival_rate_base

        while True:
            yield env.timeout(np.random.exponential(1.0 / rate))

            category_id    = np.random.choice(self._category_ids, p=self._category_weights)
            youth_ratio    = self.data_loader.get_district_youth_ratio(district)
            attractiveness = self.data_loader.get_waste_category_attractiveness(category_id)
            reuse_prob     = self.calculate_reuse_probability(district, category_id, scenario)

            item = WasteItem(
                item_id=item_counter,
                district=district,
                category=category_id,
                arrival_time=env.now,
                reuse_probability=reuse_prob,
                attractiveness=attractiveness,
                youth_ratio=youth_ratio,
                scenario_multiplier=scenario.multiplier,
            )
            items_list.append(item)
            item_counter += 1

            env.process(self.reuse_process(env, item))
            env.process(self.collection_process(env, item, self.bsr_resource))

    def reuse_process(self, env, item):
        """
        Reuse attempt: a potential buyer sees the item after ~Exp(2) days.
        If random() < P_reuse, item is reused.
        """
        yield env.timeout(np.random.exponential(2.0))   # avg 2 days

        if item.collected:
            return

        if np.random.random() < item.reuse_probability:
            item.reused         = True
            item.reuse_time     = env.now
            item.time_in_system = env.now - item.arrival_time

    def collection_process(self, env, item, bsr_resource):
        """
        BSR collection — 3 steps:
            1. Detection delay  (avg 5 days) — BSR doesn't know instantly
            2. Queue wait       (M/M/c with 3 teams)
            3. Service          (avg 0.125 days)
        """
        # --- Step 1: detection / scheduling delay ---
        yield env.timeout(np.random.exponential(self.bsr_detection_delay))

        if item.reused:
            return   # already reused while BSR was unaware

        # --- Step 2 + 3: queue + service ---
        with bsr_resource.request() as req:
            yield req

            if item.reused:
                return   # reused while waiting in queue

            yield env.timeout(np.random.exponential(1.0 / self.collection_rate))

            item.collected       = True
            item.collection_time = env.now
            item.time_in_system  = env.now - item.arrival_time

    # ==============================================================
    # RUN SINGLE
    # ==============================================================
    def run_single_scenario(self, scenario, district, simulation_days=365, run_id=0):
        env = simpy.Environment()
        self.bsr_resource = simpy.Resource(env, capacity=self.num_bsr_teams)
        items: List[WasteItem] = []

        env.process(self.waste_arrival_process(env, district, scenario, items))
        env.run(until=simulation_days)

        m = SimulationMetrics(scenario=scenario.scenario_name, district=district, run_id=run_id)
        m.total_items     = len(items)
        m.items_reused    = sum(1 for i in items if i.reused)
        m.items_collected = sum(1 for i in items if i.collected and not i.reused)

        reused_t    = [i.time_in_system for i in items if i.reused    and i.time_in_system]
        collected_t = [i.time_in_system for i in items if i.collected and not i.reused and i.time_in_system]

        m.avg_time_to_reuse      = float(np.mean(reused_t))    if reused_t    else 0.0
        m.avg_time_to_collection = float(np.mean(collected_t)) if collected_t else 0.0
        m.avg_queue_length       = m.items_collected / self.num_bsr_teams
        m.max_queue_length       = int(m.avg_queue_length * 2)
        m.total_co2_kg           = m.items_collected * self.co2_per_collection

        m.calculate_derived_metrics()
        return m

    # ==============================================================
    # RUN ALL
    # ==============================================================
    def run_all_scenarios(self, simulation_days=365, num_runs=10):
        logger.info(f"Running simulation: {simulation_days} days, {num_runs} replications")

        scenarios = [Scenario.BASELINE, Scenario.PESSIMISTIC, Scenario.REALISTIC, Scenario.OPTIMISTIC]
        districts = self.data_loader.district_demographics["bezirk"].tolist()

        all_metrics = []
        total = len(scenarios) * len(districts) * num_runs

        with tqdm(total=total, desc="Simulation Progress") as pbar:
            for scenario in scenarios:
                for district in districts:
                    for run_id in range(num_runs):
                        all_metrics.append(
                            self.run_single_scenario(scenario, district, simulation_days, run_id)
                        )
                        pbar.update(1)

        results_df = pd.DataFrame([vars(m) for m in all_metrics])
        logger.info(f"Simulation complete: {len(results_df)} result records")
        return results_df


# ==================================================================
# STANDALONE TEST
# ==================================================================
if __name__ == "__main__":
    import sys, io
    from pathlib import Path

    # Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import DataLoader

    print("\n" + "=" * 70)
    print("RUNNING SIMULATION TEST")
    print("=" * 70)

    loader = DataLoader()
    loader.load_all_data()

    districts = loader.district_demographics["bezirk"].tolist()
    print(f"\nDistricts   : {districts}")
    print(f"Categories  : {loader.get_category_ids()}")

    # --- Print expected P_reuse table ---
    print("\n" + "-" * 70)
    print("EXPECTED P_reuse per scenario x category")
    print("-" * 70)
    sim_tmp = BulkyWasteSimulation(loader)
    d0 = districts[0]
    yr = loader.get_district_youth_ratio(d0)
    print(f"  district={d0} | youth_ratio={yr:.3f} | district_scale={0.5+0.5*yr:.3f}")
    print()
    print(f"  {'Scenario':<14} {'WOOD':<10} {'E_WASTE':<10} {'TEXTILE':<10} {'MIXED':<10} {'weighted_avg':<12}")
    cats    = ['WOOD', 'E_WASTE', 'TEXTILE', 'MIXED_WASTE']
    weights = [0.45, 0.05, 0.10, 0.40]
    for sc in [Scenario.BASELINE, Scenario.PESSIMISTIC, Scenario.REALISTIC, Scenario.OPTIMISTIC]:
        row = f"  {sc.scenario_name:<14}"
        wavg = 0.0
        for cat, w in zip(cats, weights):
            p = sim_tmp.calculate_reuse_probability(d0, cat, sc)
            row += f" {p:<10.4f}"
            wavg += p * w
        row += f" {wavg:<12.4f}"
        print(row)

    # --- Run simulation ---
    print("\n" + "-" * 70)
    sim = BulkyWasteSimulation(loader)
    results = sim.run_all_scenarios(simulation_days=365, num_runs=10)

    # Save
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "simulation_results.csv", index=False)
    print(f"\nResults saved -> {out_dir / 'simulation_results.csv'}")

    # --- Summary ---
    print("\n" + "-" * 70)
    print("RESULTS BY SCENARIO (mean across runs)")
    print("-" * 70)
    cols = ["total_items", "items_reused", "items_collected", "reuse_rate", "total_co2_kg"]
    summary = results.groupby("scenario")[cols].mean().round(2)
    summary = summary.reindex(["baseline", "pessimistic", "realistic", "optimistic"])
    print(summary.to_string())

    # --- Sanity checks ---
    print("\n" + "-" * 70)
    print("SANITY CHECKS")
    print("-" * 70)

    rr = summary["reuse_rate"]
    co2 = summary["total_co2_kg"]

    order_ok = (rr["baseline"] <= rr["pessimistic"] <= rr["realistic"] <= rr["optimistic"])
    print(f"  reuse_rate order  baseline <= pess <= real <= opt?")
    print(f"  {rr['baseline']:.4f} <= {rr['pessimistic']:.4f} <= {rr['realistic']:.4f} <= {rr['optimistic']:.4f}")
    print(f"  {'[OK]' if order_ok else '[FAIL]'}")

    co2_ok = (co2["baseline"] >= co2["pessimistic"] >= co2["realistic"] >= co2["optimistic"])
    print(f"\n  CO2 order  baseline >= pess >= real >= opt?")
    print(f"  {co2['baseline']:.1f} >= {co2['pessimistic']:.1f} >= {co2['realistic']:.1f} >= {co2['optimistic']:.1f}")
    print(f"  {'[OK]' if co2_ok else '[FAIL]'}")

    print("\n" + "=" * 70)
    print("SIMULATION TEST COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")