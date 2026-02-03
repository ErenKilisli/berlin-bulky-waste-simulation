"""
FINAL Fixed Choropleth Map for Berlin Bulky Waste Simulation
Type conversion added for BEZ column
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Berlin district name to BEZ code mapping
BEZIRK_MAPPING = {
    "Mitte": 1,
    "Friedrichshain-Kreuzberg": 2,
    "Pankow": 3,
    "Charlottenburg-Wilmersdorf": 4,
    "Spandau": 5,
    "Steglitz-Zehlendorf": 6,
    "Tempelhof-Schöneberg": 7,
    "Neukölln": 8,
    "Treptow-Köpenick": 9,
    "Marzahn-Hellersdorf": 10,
    "Lichtenberg": 11,
    "Reinickendorf": 12,
}

BASE_DIR = Path(__file__).resolve().parent.parent
output_dir = BASE_DIR / "outputs" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CHOROPLETH MAP GENERATION (FINAL FIX)")
print("=" * 70)

# 1. Load simulation results
results_path = BASE_DIR / "outputs" / "logs" / "simulation_results.csv"
df = pd.read_csv(results_path)
print(f"✓ Loaded {len(df)} simulation records")

# 2. Aggregate by district (realistic scenario)
subset = df[df.scenario == 'realistic']
district_stats = subset.groupby('district').agg({
    'reuse_rate': ['mean', 'std'],
    'total_co2_kg': 'mean'
}).reset_index()
district_stats.columns = ['district', 'reuse_rate_mean', 'reuse_rate_std', 'co2_mean']

# 3. Add BEZ codes
district_stats['BEZ'] = district_stats['district'].map(BEZIRK_MAPPING)
print(f"✓ Mapped {district_stats['BEZ'].notna().sum()} districts to BEZ codes")

# 4. Load GeoJSON
geojson_path = BASE_DIR / "data" / "raw" / "berlin_map.geojson"
gdf = gpd.read_file(geojson_path)
print(f"✓ Loaded GeoJSON: {len(gdf)} geometries")
print(f"  BEZ column type: {gdf['BEZ'].dtype}")

# 5. Aggregate LOR to Bezirke
gdf_bezirke = gdf.dissolve(by='BEZ', as_index=False)
print(f"✓ Aggregated to {len(gdf_bezirke)} Bezirke")
print(f"  GeoJSON BEZ type: {gdf_bezirke['BEZ'].dtype}")
print(f"  Stats BEZ type: {district_stats['BEZ'].dtype}")

# 6. **CRITICAL FIX: Convert both to same type**
gdf_bezirke['BEZ'] = gdf_bezirke['BEZ'].astype(int)
district_stats['BEZ'] = district_stats['BEZ'].astype(int)
print(f"✓ Converted both BEZ columns to integer")

# 7. Merge
merged_gdf = gdf_bezirke.merge(district_stats, on='BEZ', how='left')
matches = merged_gdf['reuse_rate_mean'].notna().sum()
print(f"✓ Merged: {matches} / {len(merged_gdf)} matches")

if matches == 0:
    print("❌ ERROR: No matches! Check BEZ mapping.")
    print("\nGeoJSON BEZ values:")
    print(sorted(gdf_bezirke['BEZ'].unique()))
    print("\nStats BEZ values:")
    print(sorted(district_stats['BEZ'].unique()))
    exit(1)

# 8. Create choropleth
fig, ax = plt.subplots(figsize=(12, 10))

merged_gdf.plot(
    column='reuse_rate_mean',
    cmap='RdYlGn',
    linewidth=0.8,
    edgecolor='black',
    legend=True,
    ax=ax,
    vmin=0.12,
    vmax=0.15,
    legend_kwds={
        'label': 'Reuse Rate',
        'orientation': 'horizontal',
        'shrink': 0.6,
        'pad': 0.05,
    }
)

ax.set_title('Berlin Districts: Reuse Rate Distribution\n(Realistic Scenario)', 
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# 9. Add district labels
for idx, row in merged_gdf.iterrows():
    if pd.notna(row['reuse_rate_mean']):
        # Get centroid
        centroid = row.geometry.centroid
        
        # Add percentage text
        ax.text(
            centroid.x, centroid.y,
            f"{row['reuse_rate_mean']:.1%}",
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='black', alpha=0.8, linewidth=1.5)
        )

plt.tight_layout()
output_path = output_dir / '7_choropleth_map.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# 10. Create simple ranking bar chart as backup
fig, ax = plt.subplots(figsize=(10, 8))

# Sort districts by reuse rate
district_stats_sorted = district_stats.sort_values('reuse_rate_mean', ascending=True)

# Color gradient
colors = plt.cm.RdYlGn(
    (district_stats_sorted['reuse_rate_mean'] - 0.12) / (0.14 - 0.12)
)

bars = ax.barh(district_stats_sorted['district'], 
               district_stats_sorted['reuse_rate_mean'],
               color=colors, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Reuse Rate', fontsize=12, fontweight='bold')
ax.set_title('District Performance Ranking (Realistic Scenario)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, rate in zip(bars, district_stats_sorted['reuse_rate_mean']):
    ax.text(rate + 0.002, bar.get_y() + bar.get_height()/2., 
            f'{rate:.1%}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
ranking_path = output_dir / '7b_district_ranking.png'
plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {ranking_path}")
plt.close()

print("\n" + "=" * 70)
print("ALL MAPS GENERATED SUCCESSFULLY")
print(f"  - Choropleth: {output_path}")
print(f"  - Ranking:    {ranking_path}")
print("=" * 70)