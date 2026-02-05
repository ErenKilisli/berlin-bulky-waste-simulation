# Quick Start Guide - Berlin Bulky Waste Simulation

## Running the Complete Simulation

To run the entire simulation pipeline from start to finish:

```bash
python run_complete_simulation.py
```

This single command will:
1. ✅ Load all data from `data/raw/`
2. ✅ Run 480 simulations (4 scenarios × 12 districts × 10 runs)
3. ✅ Save results to `outputs/logs/simulation_results.csv`
4. ✅ Generate 7 visualizations in `outputs/visualizations/`
5. ✅ Create summary statistics table

**Expected runtime:** ~5 minutes

---

## Running Individual Components

### 1. Data Loading Only
```python
from src.data_loader import DataLoader

loader = DataLoader()
loader.load_all_data()
```

### 2. Simulation Only
```bash
python src/simulation.py
```

### 3. Visualizations Only
```bash
python src/visualizations.py
```
*Note: Requires `outputs/logs/simulation_results.csv` to exist*

---

## Output Files

### Simulation Results
- **Location:** `outputs/logs/simulation_results.csv`
- **Size:** ~60 KB
- **Records:** 480 (one per simulation run)
- **Columns:** scenario, district, run_id, total_items, items_reused, items_collected, reuse_rate, avg_time_to_reuse, avg_time_to_collection, avg_queue_length, max_queue_length, total_co2_kg

### Visualizations
All saved to `outputs/visualizations/`:

1. **1_scenario_comparison.png** - Reuse rates and CO₂ by scenario
2. **2_district_heatmap.png** - Reuse rates by district (realistic scenario)
3. **3_co2_reduction.png** - CO₂ savings vs baseline
4. **4_item_flow.png** - Items reused vs collected
5. **5_variability_boxplot.png** - Statistical variability
6. **6_sdg_impact.png** - SDG contribution scores
7. **summary_table.csv** - Statistical summary (mean ± std)

---

## Customizing Simulation Parameters

Edit `run_complete_simulation.py` or `src/simulation.py`:

```python
# Simulation duration
simulation_days = 365  # Default: 1 year

# Number of replications
num_runs = 10  # Default: 10 runs per scenario

# Random seed (for reproducibility)
random_seed = 42  # Default: 42
```

### Model Parameters (in `BulkyWasteSimulation.__init__`)
```python
self.arrival_rate_base = 5.0      # items/day/district
self.collection_rate = 8.0        # service rate per BSR team
self.bsr_detection_delay = 5.0    # avg days before BSR notices
self.num_bsr_teams = 3            # M/M/c capacity
self.natural_reuse_prob = 0.05    # baseline reuse (no app)
self.co2_per_collection = 2.5     # kg CO₂ per item
```

---

## Scenarios Explained

| Scenario | Description | Multiplier | Expected Reuse Rate |
|----------|-------------|------------|---------------------|
| **Baseline** | No digital platform | 0.0 | ~3-4% |
| **Pessimistic** | Low app adoption | 0.3 | ~7-8% |
| **Realistic** | Moderate adoption | 0.6 | ~13-14% |
| **Optimistic** | High adoption | 0.9 | ~19-20% |

Reuse probability formula:
```
P_reuse = multiplier × attractiveness × district_scale
district_scale = 0.5 + 0.5 × youth_ratio
```

---

## Troubleshooting

### Error: "Results file not found"
**Solution:** Run simulation first:
```bash
python src/simulation.py
```

### Error: "Data file not found"
**Solution:** Ensure all data files exist in `data/raw/`:
- `ordnungsamt_2023.json`
- `population_2024.csv`
- `berlin_map.geojson`
- `waste_config.json`

### Slow Performance
**Solution:** Reduce number of runs:
```python
num_runs = 5  # Instead of 10
```

---

## Interpreting Results

### Reuse Rate
- **Baseline ~4%:** Natural reuse without digital platform
- **Target 13%:** Realistic scenario (moderate app adoption)
- **Best case 19%:** Optimistic scenario (high adoption)

### CO₂ Reduction
- **Realistic:** ~426 kg/year/district (9.8% reduction)
- **Optimistic:** ~711 kg/year/district (16.4% reduction)
- **City-wide (12 districts):** 5,112-8,532 kg/year

### Item Flow
- **Baseline:** 1,735 items collected by BSR
- **Realistic:** 1,565 items collected (170 fewer)
- **Optimistic:** 1,451 items collected (284 fewer)

---

## Academic Use

### For Thesis/Report
1. Use `EXECUTION_SUMMARY.md` for methodology section
2. Include visualizations from `outputs/visualizations/`
3. Reference `summary_table.csv` for statistical tables
4. Cite data sources from README.md

### For Presentations
- **Main slide:** `1_scenario_comparison.png`
- **Environmental impact:** `3_co2_reduction.png`
- **Geographic variation:** `2_district_heatmap.png`
- **SDG contribution:** `6_sdg_impact.png`

---

## Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

Key packages:
- `simpy==4.1.1` - Discrete event simulation
- `pandas` - Data processing
- `matplotlib` - Visualization
- `seaborn` - Statistical plots
- `numpy` - Numerical operations
- `tqdm` - Progress bars

---

## Contact & Support

For questions about the simulation:
1. Check `README.md` for project overview
2. Review `EXECUTION_SUMMARY.md` for results
3. Examine source code in `src/` directory

**Project:** Berlin Bulky Waste Simulation  
**Institution:** Berlin School of Business & Innovation (BSBI)  
**Programme:** MSc Information Technology Management  
**Module:** Digital Economy & Transformation
