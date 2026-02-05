"""
Task 4 Visualizations: Berlin Bulky Waste Simulation Results
Comprehensive charts for policy insights and SDG reporting
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Load results
BASE_DIR = Path(__file__).resolve().parent.parent
results_path = BASE_DIR / "outputs" / "logs" / "simulation_results.csv"

# Create output directory
output_dir = BASE_DIR / "outputs" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("TASK 4: VISUALIZATION GENERATION")
print("="*70)

def load_data():
    if not results_path.exists():
        print(f"❌ HATA: Sonuç dosyası bulunamadı: {results_path}")
        print("Lütfen önce 'python src/simulation.py' komutunu çalıştırın.")
        sys.exit(1)
    return pd.read_csv(results_path)

# ============================================================================
# 1. SCENARIO COMPARISON - REUSE RATE & CO2
# ============================================================================
def plot_scenario_comparison(df):
    """Main policy comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Aggregate by scenario
    agg = df.groupby('scenario').agg({
        'reuse_rate': 'mean',
        'total_co2_kg': 'mean',
        'items_reused': 'mean',
        'items_collected': 'mean'
    }).reset_index()
    
    # Sıralamayı garantiye al
    scenario_order = ['baseline', 'pessimistic', 'realistic', 'optimistic']
    agg['scenario'] = pd.Categorical(agg['scenario'], categories=scenario_order, ordered=True)
    agg = agg.sort_values('scenario')
    
    scenarios = agg['scenario'].tolist()
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    # Left: Reuse Rate
    bars1 = ax1.bar(scenarios, agg['reuse_rate'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Reuse Rate', fontsize=12, fontweight='bold')
    ax1.set_title('A) Impact on Reuse Rate by Scenario', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 0.25)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right: CO2 Emissions
    bars2 = ax2.bar(scenarios, agg['total_co2_kg'], color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Total $CO_2$ Emissions (kg/year/district)', fontsize=12, fontweight='bold')
    ax2.set_title('B) $CO_2$ Impact by Scenario', fontsize=14, fontweight='bold')
    # ax2.set_ylim(0, 5000) # Dinamik olsun diye kapattım
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_scenario_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: 1_scenario_comparison.png")
    plt.close()

# ============================================================================
# 2. DISTRICT-LEVEL HEATMAP
# ============================================================================
def plot_district_heatmap(df):
    """Geographic variation in reuse rates"""
    # Sadece realistic senaryoyu al
    subset = df[df.scenario == 'realistic']
    
    pivot = subset.pivot_table(
        index='district', 
        columns='run_id', 
        values='reuse_rate'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                cbar_kws={'label': 'Reuse Rate'}, ax=ax, linewidths=0.5)
    ax.set_title('Reuse Rate by District (Realistic Scenario, 10 Runs)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Simulation Run', fontsize=12)
    ax.set_ylabel('Berlin District', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_district_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: 2_district_heatmap.png")
    plt.close()

# ============================================================================
# 3. CO2 REDUCTION POTENTIAL
# ============================================================================
def plot_co2_reduction(df):
    """Show CO2 savings vs baseline"""
    agg = df.groupby('scenario')['total_co2_kg'].mean().reset_index()
    
    # Baseline değerini güvenli şekilde al
    try:
        baseline_co2 = agg[agg.scenario == 'baseline']['total_co2_kg'].values[0]
    except IndexError:
        print("Uyarı: Baseline senaryosu bulunamadı, grafik atlanıyor.")
        return

    scenarios = ['pessimistic', 'realistic', 'optimistic']
    reductions = []
    valid_scenarios = []
    
    for s in scenarios:
        if s in agg['scenario'].values:
            scenario_co2 = agg[agg.scenario == s]['total_co2_kg'].values[0]
            reduction = baseline_co2 - scenario_co2
            reductions.append(reduction)
            valid_scenarios.append(s)
    
    if not valid_scenarios:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(valid_scenarios, reductions, color=['#ff7f0e', '#2ca02c', '#1f77b4'], 
                  alpha=0.8, edgecolor='black')
    ax.set_ylabel('$CO_2$ Reduction (kg/year/district)', fontsize=12, fontweight='bold')
    ax.set_title('$CO_2$ Emissions Reduction Compared to Baseline', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = (height / baseline_co2) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)} kg\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_co2_reduction.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: 3_co2_reduction.png")
    plt.close()

# ============================================================================
# 4. TIME SERIES (ITEM FLOW)
# ============================================================================
def plot_item_flow(df):
    """Stacked bar showing items generated, reused, collected"""
    agg = df.groupby('scenario').agg({
        'total_items': 'mean',
        'items_reused': 'mean',
        'items_collected': 'mean'
    }).reset_index()
    
    # Sıralama
    scenario_order = ['baseline', 'pessimistic', 'realistic', 'optimistic']
    agg['scenario'] = pd.Categorical(agg['scenario'], categories=scenario_order, ordered=True)
    agg = agg.sort_values('scenario')
    
    scenarios = agg['scenario'].tolist()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    reused = agg['items_reused'].values
    collected = agg['items_collected'].values
    
    ax.bar(x, reused, width, label='Reused (via app)', color='#2ca02c', alpha=0.8)
    ax.bar(x, collected, width, bottom=reused, label='Collected (BSR)', 
           color='#d62728', alpha=0.8)
    
    ax.set_ylabel('Items per Year per District', fontsize=12, fontweight='bold')
    ax.set_title('Item Flow: Reused vs Collected', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_item_flow.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: 4_item_flow.png")
    plt.close()

# ============================================================================
# 5. BOXPLOT - VARIABILITY ACROSS RUNS
# ============================================================================
def plot_variability(df):
    """Show confidence in results via boxplots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    order = ['baseline', 'pessimistic', 'realistic', 'optimistic']
    palette = {'baseline': '#d62728', 'pessimistic': '#ff7f0e', 'realistic': '#2ca02c', 'optimistic': '#1f77b4'}
    
    # Reuse rate variability
    sns.boxplot(data=df, x='scenario', y='reuse_rate', order=order, palette=palette, ax=ax1)
    ax1.set_ylabel('Reuse Rate', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Scenario', fontsize=12)
    ax1.set_title('A) Reuse Rate Variability', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # CO2 variability
    sns.boxplot(data=df, x='scenario', y='total_co2_kg', order=order, palette=palette, ax=ax2)
    ax2.set_ylabel('$CO_2$ Emissions (kg)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Scenario', fontsize=12)
    ax2.set_title('B) $CO_2$ Emissions Variability', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_variability_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: 5_variability_boxplot.png")
    plt.close()

# ============================================================================
# 6. SUMMARY TABLE
# ============================================================================
def create_summary_table(df):
    """Professional table for academic report"""
    agg = df.groupby('scenario').agg({
        'total_items': ['mean', 'std'],
        'items_reused': ['mean', 'std'],
        'items_collected': ['mean', 'std'],
        'reuse_rate': ['mean', 'std'],
        'total_co2_kg': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    
    # Save as CSV
    agg.to_csv(output_dir / 'summary_table.csv', index=False)
    print(f"[OK] Saved: summary_table.csv")
    
    return agg

# ============================================================================
# 7. SDG IMPACT VISUALIZATION
# ============================================================================
def plot_sdg_impact(df):
    """Connect results to SDG goals"""
    agg = df.groupby('scenario').agg({
        'reuse_rate': 'mean',
        'total_co2_kg': 'mean'
    }).reset_index()
    
    try:
        baseline_co2 = agg[agg.scenario == 'baseline']['total_co2_kg'].values[0]
    except:
        return

    scenarios = ['baseline', 'pessimistic', 'realistic', 'optimistic']
    sdg_scores = []
    
    for s in scenarios:
        if s in agg['scenario'].values:
            co2 = agg[agg.scenario == s]['total_co2_kg'].values[0]
            reuse = agg[agg.scenario == s]['reuse_rate'].values[0]
            
            # Composite SDG score (normalized)
            co2_reduction = (baseline_co2 - co2) / baseline_co2
            score = (co2_reduction * 0.5 + reuse * 0.5) * 100
            sdg_scores.append(score)
        else:
            sdg_scores.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(scenarios, sdg_scores, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'],
                   alpha=0.8, edgecolor='black')
    ax.set_xlabel('SDG Impact Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_title('Contribution to SDG 11 & SDG 12', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '6_sdg_impact.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: 6_sdg_impact.png")
    plt.close()

# ============================================================================
# EXECUTE ALL VISUALIZATIONS
# ============================================================================
if __name__ == "__main__":
    df = load_data()
    
    plot_scenario_comparison(df)
    plot_district_heatmap(df)
    plot_co2_reduction(df)
    plot_item_flow(df)
    plot_variability(df)
    create_summary_table(df)
    plot_sdg_impact(df)
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print(f"Location: {output_dir}")
    print("="*70)