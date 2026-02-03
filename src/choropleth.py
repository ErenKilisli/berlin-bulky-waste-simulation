"""
Choropleth Map: Berlin Bulky Waste Reuse Rate by District
Standalone script to create geographic visualization
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
simulation_results = BASE_DIR / "outputs" / "logs" / "simulation_results.csv"
geojson_file = BASE_DIR / "data" / "raw" / "berlin_map.geojson"
output_dir = BASE_DIR / "outputs" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CHOROPLETH MAP GENERATION")
print("=" * 70)

# ============================================================================
# STEP 1: Load Simulation Results
# ============================================================================
def load_simulation_data():
    """Load and aggregate district-level reuse rates"""
    if not simulation_results.exists():
        print(f"❌ ERROR: {simulation_results} not found")
        print("Run: python src/simulation.py first")
        sys.exit(1)
    
    df = pd.read_csv(simulation_results)
    print(f"✓ Loaded {len(df)} simulation records")
    
    # Focus on realistic scenario
    subset = df[df.scenario == 'realistic']
    
    # Average across 10 runs per district
    district_stats = subset.groupby('district').agg({
        'reuse_rate': ['mean', 'std'],
        'total_co2_kg': 'mean'
    }).reset_index()
    
    district_stats.columns = ['district', 'reuse_rate_mean', 'reuse_rate_std', 'co2_mean']
    
    print(f"✓ Aggregated to {len(district_stats)} districts")
    print("\nDistrict Summary:")
    print(district_stats.sort_values('reuse_rate_mean', ascending=False))
    
    return district_stats

# ============================================================================
# STEP 2: Load GeoJSON
# ============================================================================
def load_geojson():
    """Load Berlin district boundaries"""
    if not geojson_file.exists():
        print(f"❌ ERROR: {geojson_file} not found")
        print("\nExpected file structure:")
        print("  data/raw/berlin_map.geojson")
        sys.exit(1)
    
    try:
        gdf = gpd.read_file(geojson_file)
        print(f"✓ Loaded GeoJSON: {len(gdf)} geometries")
        print(f"  CRS: {gdf.crs}")
        print(f"  Columns: {gdf.columns.tolist()}")
        
        return gdf
    except Exception as e:
        print(f"❌ ERROR loading GeoJSON: {e}")
        sys.exit(1)

# ============================================================================
# STEP 3: Match District Names
# ============================================================================
def match_districts(gdf, district_stats):
    """
    Match simulation district names to GeoJSON geometries.
    
    Problem: GeoJSON has 542 LOR (neighborhoods), simulation has 12 Bezirke (districts)
    Solution: Aggregate LOR geometries to Bezirke using BEZ column
    """
    
    # Check what columns exist
    print("\n" + "-" * 70)
    print("GeoJSON Column Detection:")
    print("-" * 70)
    
    # Look for district identifier column
    possible_cols = ['BEZ', 'Bezirk', 'bezirk', 'spatial_name', 'Gemeinde_name', 
                     'name', 'NAME', 'BEZIRK', 'district']
    
    bez_col = None
    for col in possible_cols:
        if col in gdf.columns:
            bez_col = col
            print(f"✓ Found district column: '{col}'")
            break
    
    if bez_col is None:
        print("❌ No district identifier column found!")
        print(f"Available columns: {gdf.columns.tolist()}")
        print("\n⚠️  Falling back to simple approach...")
        return None, district_stats
    
    # If we have BEZ column (Bezirk code 1-12), aggregate geometries
    if bez_col in ['BEZ', 'bezirk', 'BEZIRK']:
        print(f"✓ Aggregating {len(gdf)} LOR → 12 Bezirke using '{bez_col}' column")
        
        # Dissolve (merge) geometries by Bezirk code
        bezirke_gdf = gdf.dissolve(by=bez_col, as_index=False)
        
        # Map BEZ codes (1-12) to district names
        BEZIRK_NAMES = {
            1:  "Mitte",
            2:  "Friedrichshain-Kreuzberg",
            3:  "Pankow",
            4:  "Charlottenburg-Wilmersdorf",
            5:  "Spandau",
            6:  "Steglitz-Zehlendorf",
            7:  "Tempelhof-Schöneberg",
            8:  "Neukölln",
            9:  "Treptow-Köpenick",
            10: "Marzahn-Hellersdorf",
            11: "Lichtenberg",
            12: "Reinickendorf",
        }
        
        bezirke_gdf['district'] = bezirke_gdf[bez_col].map(BEZIRK_NAMES)
        
        print(f"✓ Created {len(bezirke_gdf)} district geometries")
        print(f"  Districts: {bezirke_gdf['district'].tolist()}")
        
        # Merge with simulation results
        merged = bezirke_gdf.merge(district_stats, on='district', how='left')
        
        print(f"\n✓ Merged simulation data with geometries")
        print(f"  Matches: {merged['reuse_rate_mean'].notna().sum()} / {len(merged)}")
        
        return merged, district_stats
    
    else:
        print("⚠️  Non-standard district column, manual mapping needed")
        return None, district_stats

# ============================================================================
# STEP 4A: Choropleth Map (if geometries available)
# ============================================================================
def plot_choropleth(merged_gdf, district_stats):
    """Create choropleth map with district-level reuse rates"""
    
    if merged_gdf is None:
        print("\n⚠️  Skipping choropleth (no matched geometries)")
        return False
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot choropleth
    merged_gdf.plot(
        column='reuse_rate_mean',
        cmap='RdYlGn',  # Red (low) to Green (high)
        edgecolor='black',
        linewidth=1.5,
        legend=True,
        ax=ax,
        vmin=0.11,  # Min reuse rate from data
        vmax=0.15,  # Max reuse rate from data
        legend_kwds={
            'label': "Reuse Rate (%)",
            'orientation': "horizontal",
            'shrink': 0.8,
            'pad': 0.05
        }
    )
    
    # Add district labels
    for idx, row in merged_gdf.iterrows():
        # Get centroid for label placement
        centroid = row.geometry.centroid
        
        # Add district name
        ax.text(
            centroid.x, centroid.y,
            row['district'],
            fontsize=9,
            ha='center',
            va='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
        
        # Add reuse rate value below name
        if pd.notna(row['reuse_rate_mean']):
            ax.text(
                centroid.x, centroid.y - 0.02,  # Slightly below name
                f"{row['reuse_rate_mean']:.1%}",
                fontsize=8,
                ha='center',
                va='top',
                color='darkblue',
                fontweight='bold'
            )
    
    ax.set_title(
        'Berlin Bulky Waste Reuse Rate by District\n(Realistic Scenario, Average of 10 Simulation Runs)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '7_choropleth_map.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: 7_choropleth_map.png")
    plt.close()
    
    return True

# ============================================================================
# STEP 4B: Alternative - District Ranking Bar Chart
# ============================================================================
def plot_district_ranking(district_stats):
    """Fallback: horizontal bar chart ranked by performance"""
    
    # Sort by reuse rate
    sorted_districts = district_stats.sort_values('reuse_rate_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color bars by value (gradient)
    norm = plt.Normalize(vmin=sorted_districts['reuse_rate_mean'].min(), 
                         vmax=sorted_districts['reuse_rate_mean'].max())
    colors = plt.cm.RdYlGn(norm(sorted_districts['reuse_rate_mean'].values))
    
    bars = ax.barh(
        sorted_districts['district'],
        sorted_districts['reuse_rate_mean'],
        color=colors,
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels
    for i, (bar, rate, std) in enumerate(zip(bars, 
                                              sorted_districts['reuse_rate_mean'], 
                                              sorted_districts['reuse_rate_std'])):
        ax.text(
            rate + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f'{rate:.1%} ± {std:.1%}',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xlabel('Reuse Rate', fontsize=12, fontweight='bold')
    ax.set_title(
        'District Performance Ranking: Reuse Rate (Realistic Scenario)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, sorted_districts['reuse_rate_mean'].max() * 1.15)
    
    # Add ranking numbers
    for i, (idx, row) in enumerate(sorted_districts.iterrows()):
        ax.text(
            -0.005,
            i,
            f"#{i+1}",
            va='center',
            ha='right',
            fontsize=9,
            fontweight='bold',
            color='gray'
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / '7_district_ranking.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 7_district_ranking.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # Step 1: Load simulation results
    district_stats = load_simulation_data()
    
    # Step 2: Load GeoJSON
    gdf = load_geojson()
    
    # Step 3: Match districts
    merged_gdf, district_stats = match_districts(gdf, district_stats)
    
    # Step 4: Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Try choropleth first
    choropleth_success = plot_choropleth(merged_gdf, district_stats)
    
    # Always create ranking chart (works even if choropleth fails)
    plot_district_ranking(district_stats)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    if choropleth_success:
        print("✅ Choropleth map created successfully")
        print("✅ Ranking chart created as alternative view")
    else:
        print("⚠️  Choropleth map skipped (geometry matching issue)")
        print("✅ Ranking chart created as primary visualization")
    
    print(f"\nOutput directory: {output_dir}")