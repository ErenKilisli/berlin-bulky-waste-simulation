import json
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np


# Resolve project root and data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


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
    print("Loading datasets...")

    waste = load_waste_config()
    print(f"Waste categories loaded: {len(waste)}")

    population = load_population_data()
    print(f"Population data loaded: {population.shape[0]} areas")

    incidents = load_ordnungsamt_data()
    print(f"Incident records loaded: {incidents.shape[0]}")

    berlin_map = load_map_data()
    print(f"District geometries loaded: {len(berlin_map)}")

    print("Data loading completed successfully.")
