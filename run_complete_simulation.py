"""
Complete Berlin Bulky Waste Simulation Pipeline
================================================
This script runs the entire simulation from start to finish:
1. Load and validate data
2. Run simulation across all scenarios
3. Generate all visualizations
4. Create summary reports
"""

import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import DataLoader
from simulation import BulkyWasteSimulation
import visualizations
import pandas as pd

def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    print_header("BERLIN BULKY WASTE SIMULATION - COMPLETE PIPELINE")
    
    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    print_header("STEP 1: LOADING DATA")
    print("Loading municipal datasets...")
    
    try:
        loader = DataLoader()
        loader.load_all_data()
        
        districts = loader.district_demographics["bezirk"].tolist()
        categories = loader.get_category_ids()
        
        print(f"✓ Loaded {len(districts)} districts: {districts}")
        print(f"✓ Loaded {len(categories)} waste categories: {categories}")
        print("✓ Data loading complete!")
        
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return 1
    
    # ========================================================================
    # STEP 2: SIMULATION SETUP
    # ========================================================================
    print_header("STEP 2: SIMULATION SETUP")
    
    simulation_days = 365
    num_runs = 10
    
    print(f"Simulation parameters:")
    print(f"  - Duration: {simulation_days} days (1 year)")
    print(f"  - Replications: {num_runs} runs per scenario")
    print(f"  - Scenarios: baseline, pessimistic, realistic, optimistic")
    print(f"  - Districts: {len(districts)}")
    print(f"  - Total simulations: {4 * len(districts) * num_runs} = {4 * len(districts) * num_runs}")
    
    # ========================================================================
    # STEP 3: RUN SIMULATION
    # ========================================================================
    print_header("STEP 3: RUNNING SIMULATION")
    
    try:
        sim = BulkyWasteSimulation(loader, random_seed=42)
        results = sim.run_all_scenarios(
            simulation_days=simulation_days,
            num_runs=num_runs
        )
        
        print(f"\n✓ Simulation complete!")
        print(f"✓ Generated {len(results)} result records")
        
    except Exception as e:
        print(f"❌ ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STEP 4: SAVE RESULTS
    # ========================================================================
    print_header("STEP 4: SAVING RESULTS")
    
    try:
        output_dir = Path(__file__).resolve().parent / "outputs" / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "simulation_results.csv"
        results.to_csv(results_path, index=False)
        
        print(f"✓ Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ ERROR saving results: {e}")
        return 1
    
    # ========================================================================
    # STEP 5: RESULTS SUMMARY
    # ========================================================================
    print_header("STEP 5: RESULTS SUMMARY")
    
    try:
        cols = ["total_items", "items_reused", "items_collected", "reuse_rate", "total_co2_kg"]
        summary = results.groupby("scenario")[cols].mean().round(2)
        summary = summary.reindex(["baseline", "pessimistic", "realistic", "optimistic"])
        
        print("\nMean results by scenario:")
        print(summary.to_string())
        
        # Sanity checks
        print("\n" + "-" * 80)
        print("SANITY CHECKS")
        print("-" * 80)
        
        rr = summary["reuse_rate"]
        co2 = summary["total_co2_kg"]
        
        order_ok = (rr["baseline"] <= rr["pessimistic"] <= rr["realistic"] <= rr["optimistic"])
        print(f"\nReuse rate ordering (baseline ≤ pessimistic ≤ realistic ≤ optimistic):")
        print(f"  {rr['baseline']:.4f} ≤ {rr['pessimistic']:.4f} ≤ {rr['realistic']:.4f} ≤ {rr['optimistic']:.4f}")
        print(f"  {'✓ [OK]' if order_ok else '✗ [FAIL]'}")
        
        co2_ok = (co2["baseline"] >= co2["pessimistic"] >= co2["realistic"] >= co2["optimistic"])
        print(f"\nCO₂ ordering (baseline ≥ pessimistic ≥ realistic ≥ optimistic):")
        print(f"  {co2['baseline']:.1f} ≥ {co2['pessimistic']:.1f} ≥ {co2['realistic']:.1f} ≥ {co2['optimistic']:.1f}")
        print(f"  {'✓ [OK]' if co2_ok else '✗ [FAIL]'}")
        
    except Exception as e:
        print(f"❌ ERROR generating summary: {e}")
        return 1
    
    # ========================================================================
    # STEP 6: GENERATE VISUALIZATIONS
    # ========================================================================
    print_header("STEP 6: GENERATING VISUALIZATIONS")
    
    try:
        print("Creating visualizations...")
        
        # Run all visualization functions
        visualizations.plot_scenario_comparison(results)
        visualizations.plot_district_heatmap(results)
        visualizations.plot_co2_reduction(results)
        visualizations.plot_item_flow(results)
        visualizations.plot_variability(results)
        visualizations.create_summary_table(results)
        visualizations.plot_sdg_impact(results)
        
        viz_dir = Path(__file__).resolve().parent / "outputs" / "visualizations"
        print(f"\n✓ All visualizations saved to: {viz_dir}")
        
    except Exception as e:
        print(f"❌ ERROR generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print_header("SIMULATION PIPELINE COMPLETED SUCCESSFULLY! ✓")
    
    print("\nGenerated outputs:")
    print(f"  📊 Simulation results: {results_path}")
    print(f"  📈 Visualizations: {viz_dir}")
    print(f"  📋 Summary table: {viz_dir / 'summary_table.csv'}")
    
    print("\nGenerated visualizations:")
    print("  1. 1_scenario_comparison.png - Main policy comparison")
    print("  2. 2_district_heatmap.png - Geographic variation")
    print("  3. 3_co2_reduction.png - CO₂ savings vs baseline")
    print("  4. 4_item_flow.png - Item flow (reused vs collected)")
    print("  5. 5_variability_boxplot.png - Statistical variability")
    print("  6. 6_sdg_impact.png - SDG contribution scores")
    print("  7. summary_table.csv - Statistical summary table")
    
    print("\n" + "=" * 80)
    print("You can now review the results and visualizations!")
    print("=" * 80 + "\n")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
