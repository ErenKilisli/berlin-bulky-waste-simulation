# Berlin Bulky Waste Simulation - Execution Summary

**Date:** 2026-02-05  
**Status:** ✅ COMPLETED SUCCESSFULLY

---

## Execution Overview

The complete Berlin Bulky Waste Simulation pipeline has been executed from start to finish, including:
1. Data loading and validation
2. Discrete Event Simulation (DES) across all scenarios
3. Statistical analysis
4. Comprehensive visualization generation

---

## Simulation Parameters

- **Duration:** 365 days (1 year)
- **Replications:** 10 runs per scenario
- **Districts:** 12 Berlin districts
- **Scenarios:** 4 (baseline, pessimistic, realistic, optimistic)
- **Total Simulations:** 480 individual simulation runs

---

## Key Results Summary

### Reuse Rates by Scenario
- **Baseline:** 3.6% (no digital intervention)
- **Pessimistic:** 7.4% (low app adoption)
- **Realistic:** 13.2% (moderate app adoption)
- **Optimistic:** 19.3% (high app adoption)

### CO₂ Emissions (kg/year/district)
- **Baseline:** 4,338 kg
- **Pessimistic:** 4,165 kg (-173 kg, -4.0%)
- **Realistic:** 3,912 kg (-426 kg, -9.8%)
- **Optimistic:** 3,627 kg (-711 kg, -16.4%)

### Item Flow (items/year/district)
| Scenario | Total Items | Items Reused | Items Collected |
|----------|-------------|--------------|-----------------|
| Baseline | 1,825 | 65 | 1,735 |
| Pessimistic | 1,825 | 135 | 1,666 |
| Realistic | 1,828 | 241 | 1,565 |
| Optimistic | 1,824 | 353 | 1,451 |

---

## Validation Checks

✅ **Reuse Rate Ordering:** baseline ≤ pessimistic ≤ realistic ≤ optimistic  
✅ **CO₂ Ordering:** baseline ≥ pessimistic ≥ realistic ≥ optimistic  
✅ **All sanity checks passed**

---

## Generated Outputs

### 1. Simulation Data
📊 **Location:** `outputs/logs/simulation_results.csv`  
- 480 simulation records
- Complete metrics for each run

### 2. Visualizations
📈 **Location:** `outputs/visualizations/`

1. **1_scenario_comparison.png** - Main policy comparison showing reuse rates and CO₂ impact
2. **2_district_heatmap.png** - Geographic variation in reuse rates across Berlin districts
3. **3_co2_reduction.png** - CO₂ emissions reduction compared to baseline
4. **4_item_flow.png** - Stacked bar chart showing items reused vs collected
5. **5_variability_boxplot.png** - Statistical variability across simulation runs
6. **6_sdg_impact.png** - Contribution to SDG 11 & SDG 12 goals
7. **summary_table.csv** - Statistical summary with means and standard deviations

---

## Key Insights

### Environmental Impact
- The **realistic scenario** achieves a **9.8% reduction** in CO₂ emissions
- The **optimistic scenario** achieves a **16.4% reduction** in CO₂ emissions
- This demonstrates significant environmental benefits from digital reuse platforms

### Waste Reduction
- Digital platform can increase reuse rates from **3.6% to 13.2%** (realistic)
- This represents a **3.7x increase** in item reuse
- Reduces municipal collection burden by **170-285 items/year/district**

### District Variation
- Reuse rates vary by district (11-15% in realistic scenario)
- Districts with higher youth ratios show better reuse performance
- Treptow-Köpenick shows highest reuse potential (up to 15.2%)

### SDG Contribution
- **Baseline:** SDG Impact Score = 1.8
- **Realistic:** SDG Impact Score = 11.5
- **Optimistic:** SDG Impact Score = 17.9
- Strong contribution to SDG 11 (Sustainable Cities) and SDG 12 (Responsible Consumption)

---

## Methodology Validation

The simulation successfully implements:
- ✅ **Discrete Event Simulation (DES)** using SimPy
- ✅ **M/M/c queueing model** for BSR collection capacity
- ✅ **Poisson arrival process** for waste items
- ✅ **Exponential service times** for collection operations
- ✅ **District-specific parameters** (youth ratio, attractiveness)
- ✅ **Multiple scenario analysis** with proper ordering

---

## Technical Details

### Simulation Model
- **Framework:** SimPy 4.1.1
- **Random Seed:** 42 (for reproducibility)
- **Arrival Rate:** 5 items/day/district
- **BSR Teams:** 3 teams per district (M/M/c queue)
- **Detection Delay:** 5 days average (exponential)
- **Collection Rate:** 8 items/day/team

### Data Sources
- Ordnungsamt-Online (illegal dumping reports)
- Amt für Statistik Berlin-Brandenburg (population data)
- Berliner Stadtreinigung (waste composition)
- Berlin LOR geospatial boundaries

---

## Next Steps

The simulation results are now ready for:
1. **Academic reporting** - Use visualizations and summary table in thesis/report
2. **Policy recommendations** - Present findings to stakeholders
3. **Further analysis** - Sensitivity analysis, parameter optimization
4. **Publication** - Academic papers or policy briefs

---

## Files Generated

```
outputs/
├── logs/
│   └── simulation_results.csv          (60 KB, 480 records)
└── visualizations/
    ├── 1_scenario_comparison.png       (174 KB)
    ├── 2_district_heatmap.png          (612 KB)
    ├── 3_co2_reduction.png             (124 KB)
    ├── 4_item_flow.png                 (115 KB)
    ├── 5_variability_boxplot.png       (169 KB)
    ├── 6_sdg_impact.png                (90 KB)
    └── summary_table.csv               (486 bytes)
```

---

## Conclusion

✅ **Simulation completed successfully**  
✅ **All visualizations generated**  
✅ **Results validated and consistent**  
✅ **Ready for academic use**

The simulation demonstrates that a digital reuse platform ("zu verschenken" app) can significantly reduce illegal bulky waste accumulation in Berlin, achieving 9.8-16.4% CO₂ emission reductions and 3.7-5.4x increases in reuse rates.

---

**Generated by:** Berlin Bulky Waste Simulation Pipeline  
**Execution Time:** ~5 minutes  
**Total Simulations:** 480 runs across 4 scenarios, 12 districts, 10 replications
