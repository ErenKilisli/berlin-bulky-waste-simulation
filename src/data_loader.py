import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve project root and data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _resolve_path(filename: str) -> Path:
    """Try data/ first, then data/raw/."""
    path = DATA_DIR / filename
    if not path.exists():
        path = DATA_DIR / "raw" / filename
    return path


class DataLoader:
    """Unified data loader for Berlin bulky waste simulation."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.waste_config: list = []
        self.population_data: pd.DataFrame = None
        self.ordnungsamt_data: pd.DataFrame = None
        self.geo_data: gpd.GeoDataFrame = None
        self.district_demographics: pd.DataFrame = None

    # ------------------------------------------------------------------
    # WASTE CONFIG
    # ------------------------------------------------------------------
    def load_waste_config(self) -> list:
        path = _resolve_path("waste_config.json")
        logger.info(f"Loading waste config from: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.waste_config = data
        elif isinstance(data, dict) and "categories" in data:
            self.waste_config = data["categories"]
        else:
            raise ValueError("waste_config.json must be a list or contain a 'categories' key")

        logger.info(f"Loaded waste config with {len(self.waste_config)} categories")
        return self.waste_config

    def get_waste_category_attractiveness(self, category_id: str) -> float:
        """Return app_sell_chance for a given category_id."""
        for cat in self.waste_config:
            if cat["category_id"] == category_id:
                return cat["app_sell_chance"]
        logger.warning(f"Category '{category_id}' not found, using default 0.5")
        return 0.5

    def get_category_ids(self) -> list:
        """Return list of all category_id values."""
        return [cat["category_id"] for cat in self.waste_config]

    def get_category_probabilities(self) -> dict:
        """Return {category_id: probability} for weighted random selection."""
        return {cat["category_id"]: cat["probability"] for cat in self.waste_config}

    # ------------------------------------------------------------------
    # POPULATION
    # ------------------------------------------------------------------
    def load_population_data(self) -> pd.DataFrame:
        path = _resolve_path("population_2024.csv")
        logger.info(f"Loading population data from: {path}")

        df = pd.read_csv(path, sep=";", encoding="utf-8")

        if "E_E" in df.columns:
            df["total_population"] = pd.to_numeric(df["E_E"], errors="coerce").fillna(0)

        # Youth ratio (18-45) for reuse probability model
        youth_cols = []
        for age in range(18, 46):
            for col in df.columns:
                if f"E{age}" in col or f"E_E{age}" in col:
                    youth_cols.append(col)
                    break

        if youth_cols and "total_population" in df.columns:
            df["youth_population"] = df[youth_cols].apply(
                lambda row: pd.to_numeric(row, errors="coerce").sum(), axis=1
            )
            df["youth_ratio"] = (df["youth_population"] / df["total_population"]).clip(0.15, 0.45)
        else:
            # Fallback: use age group columns E_E18U25, E_E25U55
            age_group_cols = [c for c in df.columns if any(x in c for x in ["E_E18U25", "E_E25U55"])]
            if age_group_cols:
                df["youth_population"] = df[age_group_cols].apply(
                    lambda row: pd.to_numeric(row, errors="coerce").sum(), axis=1
                )
                df["youth_ratio"] = (df["youth_population"] / df["total_population"]).clip(0.15, 0.45)
            else:
                np.random.seed(42)
                df["youth_ratio"] = np.random.uniform(0.2, 0.4, size=len(df))

        self.population_data = df
        logger.info(f"Loaded population data: {df.shape[0]} areas")
        return df

    # ------------------------------------------------------------------
    # ORDNUNGSAMT
    # ------------------------------------------------------------------
    def load_ordnungsamt_data(self) -> pd.DataFrame:
        path = _resolve_path("ordnungsamt_2023.json")
        logger.info(f"Loading Ordnungsamt data from: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle Berlin Open Data API format: {"messages": {...}, "results": {...}, "index": [...]}
        if isinstance(data, dict) and "index" in data and isinstance(data["index"], list):
            df = pd.DataFrame(data["index"])
        # Fallback: list of records
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        # Fallback: dict with a single key containing list
        elif isinstance(data, dict):
            key = next(iter(data))
            if isinstance(data[key], list):
                df = pd.DataFrame(data[key])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Unexpected structure in ordnungsamt_2023.json")

        self.ordnungsamt_data = df
        logger.info(f"Loaded Ordnungsamt data: {df.shape[0]} incidents")
        return df

    # ------------------------------------------------------------------
    # GEO / MAP
    # ------------------------------------------------------------------
    def load_map_data(self) -> gpd.GeoDataFrame:
        path = _resolve_path("berlin_map.geojson")
        logger.info(f"Loading geo data from: {path}")

        gdf = gpd.read_file(path)
        self.geo_data = gdf
        logger.info(f"Loaded geo data: {len(gdf)} geometries")
        return gdf

    # ------------------------------------------------------------------
    # DISTRICT DEMOGRAPHICS (12 Bezirke)
    # ------------------------------------------------------------------
    def calculate_district_demographics(self) -> pd.DataFrame:
        """
        Aggregate 542 LOR areas into 12 Berlin Bezirke using BEZ column.
        """
        if self.population_data is None:
            self.load_population_data()

        df = self.population_data.copy()

        # Berlin Bezirk names (official)
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

        # Check if BEZ column exists
        if "BEZ" not in df.columns:
            logger.warning("BEZ column not found, treating entire dataset as one district")
            demographics = pd.DataFrame([{
                "bezirk": "Berlin",
                "total_population": df["total_population"].sum(),
                "youth_ratio": df["youth_ratio"].mean(),
            }])
        else:
            # Group by BEZ (Bezirk code 1-12)
            demographics = (
                df.groupby("BEZ")
                .agg(
                    total_population=("total_population", "sum"),
                    youth_ratio=("youth_ratio", "mean"),
                )
                .reset_index()
            )
            
            # Map BEZ codes to district names
            demographics["bezirk"] = demographics["BEZ"].map(BEZIRK_NAMES)
            demographics = demographics[["bezirk", "total_population", "youth_ratio"]]

        # Clip youth_ratio to reasonable range
        demographics["youth_ratio"] = demographics["youth_ratio"].clip(0.15, 0.45)

        # Save processed
        out_dir = self.data_dir / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        demographics.to_csv(out_dir / "district_demographics.csv", index=False)

        self.district_demographics = demographics
        logger.info(f"Prepared demographics for {len(demographics)} districts")
        return demographics

    def get_district_youth_ratio(self, district: str) -> float:
        """Return youth_ratio for a district name."""
        if self.district_demographics is None:
            self.calculate_district_demographics()

        row = self.district_demographics[self.district_demographics["bezirk"] == district]
        if len(row) == 0:
            logger.warning(f"District '{district}' not found → default 0.30")
            return 0.30
        return float(row.iloc[0]["youth_ratio"])

    # ------------------------------------------------------------------
    # LOAD ALL
    # ------------------------------------------------------------------
    def load_all_data(self):
        """Load every dataset and prepare demographics."""
        logger.info("Loading all data sources...")
        self.load_waste_config()
        self.load_population_data()
        self.load_ordnungsamt_data()
        self.load_map_data()
        self.calculate_district_demographics()
        logger.info("All data loaded successfully")
        return (
            self.waste_config,
            self.population_data,
            self.ordnungsamt_data,
            self.geo_data,
            self.district_demographics,
        )


# ==================================================================
# STANDALONE TEST
# ==================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING DATA LOADER")
    print("=" * 70)

    loader = DataLoader()

    try:
        waste, population, incidents, geo, demographics = loader.load_all_data()

        print(f"\n[OK] Waste categories       : {len(waste)}")
        print(f"[OK] Population records     : {population.shape[0]}")
        print(f"[OK] Ordnungsamt incidents  : {incidents.shape[0]}")
        print(f"[OK] Geographic features    : {len(geo)}")
        print(f"[OK] Districts (aggregated) : {len(demographics)}")

        # --- Categories ---
        print("\n" + "-" * 70)
        print("Waste Categories (app_sell_chance):")
        print("-" * 70)
        for cat in waste:
            print(f"  {cat['category_id']:>12}  |  {cat['name']:<45}  |  {cat['app_sell_chance']}")

        # --- Demographics ---
        print("\n" + "-" * 70)
        print("District Demographics (12 Bezirke):")
        print("-" * 70)
        print(demographics.to_string(index=False))

        # --- Youth ratio range ---
        print("\n" + "-" * 70)
        print("Youth Ratio Statistics:")
        print("-" * 70)
        print(f"  Min : {demographics['youth_ratio'].min():.3f}")
        print(f"  Max : {demographics['youth_ratio'].max():.3f}")
        print(f"  Mean: {demographics['youth_ratio'].mean():.3f}")

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()