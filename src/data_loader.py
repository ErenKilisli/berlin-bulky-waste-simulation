"""
Data Loader Module

Handles loading and preprocessing of all data sources:
- Waste configuration (waste_config.json)
- Population demographics (population_2024.csv)
- Illegal dumping incidents (ordnungsamt_2023.json)
- Berlin district boundaries (berlin_map.geojson)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import geopandas as gpd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve project root and data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class DataLoader:
    """
    Centralized data loader for all simulation data sources.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Optional custom data directory path
        """
        self.data_dir = data_dir or DATA_DIR
        
        # Data containers
        self.waste_config: Optional[Dict] = None
        self.population_data: Optional[pd.DataFrame] = None
        self.ordnungsamt_data: Optional[pd.DataFrame] = None
        self.geo_data: Optional[gpd.GeoDataFrame] = None
        self.district_demographics: Optional[pd.DataFrame] = None
        
    def load_waste_config(self) -> Dict:
        """Load bulky waste category configuration."""
        path = self.data_dir / "waste_config.json"
        if not path.exists():
            path = self.data_dir / "raw" / "waste_config.json"
        
        logger.info(f"Loading waste config from: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            self.waste_config = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(self.waste_config, list):
            # Convert list to dict format
            categories = {}
            for item in self.waste_config:
                if isinstance(item, dict) and 'name' in item:
                    categories[item['name']] = item
            self.waste_config = {'categories': categories}
        
        logger.info(f"Loaded waste config with {len(self.waste_config.get('categories', {}))} categories")
        
        return self.waste_config
    
    def load_population_data(self) -> pd.DataFrame:
        """Load population data and derive youth ratio."""
        path = self.data_dir / "population_2024.csv"
        if not path.exists():
            path = self.data_dir / "raw" / "population_2024.csv"
        
        logger.info(f"Loading population data from: {path}")
        
        df = pd.read_csv(path, sep=";", encoding="utf-8")
        
        # Extract total population
        if "E_E" in df.columns:
            df["total_population"] = pd.to_numeric(df["E_E"], errors="coerce").fillna(0)
        
        # Calculate youth ratio (18-45 age group)
        # Look for age columns in the format E_E01 to E_E18 (age groups)
        age_cols = [col for col in df.columns if col.startswith("E_E") and len(col) > 3]
        
        if age_cols:
            # Simplified: assume columns 3-8 represent youth (18-45)
            youth_cols = age_cols[3:8] if len(age_cols) >= 8 else age_cols[:3]
            df["youth_population"] = df[youth_cols].apply(
                lambda x: pd.to_numeric(x, errors="coerce"), axis=1
            ).sum(axis=1)
            df["youth_ratio"] = (df["youth_population"] / df["total_population"]).fillna(0.30)
            df["youth_ratio"] = df["youth_ratio"].clip(0.10, 0.50)
        else:
            # Default youth ratio
            np.random.seed(42)
            df["youth_ratio"] = np.random.uniform(0.25, 0.45, size=len(df))
        
        # Extract district names
        if "BEZIRK" in df.columns:
            df["bezirk"] = df["BEZIRK"]
        elif "Bezirk" in df.columns:
            df["bezirk"] = df["Bezirk"]
        else:
            # Use first text column as district name
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                df["bezirk"] = df[text_cols[0]]
        
        self.population_data = df
        logger.info(f"Loaded population data: {len(df)} areas")
        
        return df
    
    def load_ordnungsamt_data(self) -> pd.DataFrame:
        """Load illegal bulky waste incident reports."""
        path = self.data_dir / "ordnungsamt_2023.json"
        if not path.exists():
            path = self.data_dir / "raw" / "ordnungsamt_2023.json"
        
        logger.info(f"Loading Ordnungsamt data from: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle nested JSON structure
        if isinstance(data, dict):
            # Try common keys
            for key in ['features', 'data', 'records', 'index']:
                if key in data:
                    data = data[key]
                    break
            else:
                # Use first key
                key = next(iter(data))
                data = data[key]
        
        df = pd.DataFrame(data)
        
        # Parse timestamps if available
        timestamp_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in timestamp_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        self.ordnungsamt_data = df
        logger.info(f"Loaded Ordnungsamt data: {len(df)} incidents")
        
        return df
    
    def load_geo_data(self) -> gpd.GeoDataFrame:
        """Load Berlin district boundaries (GeoJSON)."""
        path = self.data_dir / "berlin_map.geojson"
        if not path.exists():
            path = self.data_dir / "raw" / "berlin_map.geojson"
        
        logger.info(f"Loading geo data from: {path}")
        
        gdf = gpd.read_file(path)
        
        self.geo_data = gdf
        logger.info(f"Loaded geo data: {len(gdf)} geometries")
        
        return gdf
    
    def prepare_district_demographics(self) -> pd.DataFrame:
        """
        Prepare aggregated district-level demographics.
        
        Returns:
            DataFrame with district-level statistics
        """
        if self.population_data is None:
            self.load_population_data()
        
        # Group by district (bezirk)
        if "bezirk" in self.population_data.columns:
            demographics = self.population_data.groupby("bezirk").agg({
                "total_population": "sum",
                "youth_ratio": "mean"
            }).reset_index()
        else:
            # If no district column, use entire dataset
            demographics = pd.DataFrame({
                "bezirk": ["Berlin"],
                "total_population": [self.population_data["total_population"].sum()],
                "youth_ratio": [self.population_data["youth_ratio"].mean()]
            })
        
        self.district_demographics = demographics
        logger.info(f"Prepared demographics for {len(demographics)} districts")
        
        return demographics
    
    def load_all_data(self) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
        """
        Load all data sources.
        
        Returns:
            Tuple of (waste_config, population, ordnungsamt, geo_data, demographics)
        """
        logger.info("Loading all data sources...")
        
        try:
            self.load_waste_config()
            self.load_population_data()
            self.load_ordnungsamt_data()
            self.load_geo_data()
            self.prepare_district_demographics()
            
            logger.info("All data loaded successfully")
            
            return (
                self.waste_config,
                self.population_data,
                self.ordnungsamt_data,
                self.geo_data,
                self.district_demographics
            )
        
        except Exception as e:
            logger.error(f"✗ Error during data loading: {e}")
            raise
    
    def get_district_youth_ratio(self, district: str) -> float:
        """
        Get youth ratio for a specific district.
        
        Args:
            district: District name
            
        Returns:
            Youth ratio (default: 0.30)
        """
        if self.district_demographics is None:
            self.prepare_district_demographics()
        
        match = self.district_demographics[
            self.district_demographics["bezirk"].str.contains(district, case=False, na=False)
        ]
        
        if not match.empty:
            return match.iloc[0]["youth_ratio"]
        
        # Default value
        return 0.30
    
    def get_waste_category_attractiveness(self, category: str) -> float:
        """
        Get attractiveness score for a waste category.
        
        Args:
            category: Waste category name
            
        Returns:
            Attractiveness score (0.0-1.0)
        """
        if self.waste_config is None:
            self.load_waste_config()
        
        categories = self.waste_config.get('categories', {})
        
        if category in categories:
            return categories[category].get('attractiveness', 0.5)
        
        # Default value
        return 0.5


# Standalone functions for backward compatibility
def load_waste_config():
    """Load bulky waste category configuration."""
    path = DATA_DIR / "waste_config.json"
    if not path.exists():
        path = DATA_DIR / "raw" / "waste_config.json"
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_population_data():
    """Load population data and derive a simplified app adoption rate."""
    path = DATA_DIR / "population_2024.csv"
    if not path.exists():
        path = DATA_DIR / "raw" / "population_2024.csv"
    
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    
    if "E_E" in df.columns:
        df["total_population"] = pd.to_numeric(df["E_E"], errors="coerce").fillna(0)
        
        np.random.seed(42)
        df["adoption_rate"] = np.random.uniform(0.1, 0.7, size=len(df))
    
    return df


def load_ordnungsamt_data():
    """Load illegal bulky waste incident reports."""
    path = DATA_DIR / "ordnungsamt_2023.json"
    if not path.exists():
        path = DATA_DIR / "raw" / "ordnungsamt_2023.json"
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        key = next(iter(data))
        return pd.DataFrame(data[key])
    
    return pd.DataFrame(data)


def load_map_data():
    """Load Berlin district boundaries (GeoJSON)."""
    path = DATA_DIR / "berlin_map.geojson"
    if not path.exists():
        path = DATA_DIR / "raw" / "berlin_map.geojson"
    
    return gpd.read_file(path)


if __name__ == "__main__":
    print("="*70)
    print("TESTING DATA LOADER")
    print("="*70)
    
    # Test class-based loader
    loader = DataLoader()
    
    try:
        waste_config, population, ordnungsamt, geo, demographics = loader.load_all_data()
        
        print("\n[OK] Data Loading Summary:")
        print(f"  - Waste categories: {len(waste_config.get('categories', {}))}")
        print(f"  - Population records: {len(population)}")
        print(f"  - Ordnungsamt incidents: {len(ordnungsamt)}")
        print(f"  - Geographic features: {len(geo)}")
        print(f"  - Districts: {len(demographics)}")
        
        print("\n[OK] District Demographics Sample:")
        print(demographics.head())
        
        print("\n[OK] Waste Categories:")
        for cat_name in waste_config.get('categories', {}).keys():
            attractiveness = loader.get_waste_category_attractiveness(cat_name)
            print(f"  - {cat_name}: attractiveness = {attractiveness}")
        
    except Exception as e:
        print(f"\n[ERROR] Error during data loading: {e}")
        import traceback
        traceback.print_exc()
