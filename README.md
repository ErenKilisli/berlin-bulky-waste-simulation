# Berlin Bulky Waste Simulation

## Project Overview
This project develops a **Discrete Event Simulation (DES)** model to analyse
illegal bulky waste (Sperrmüll) incidents in Berlin. The study combines
real-world open municipal data with simulation-based modelling to evaluate
the potential impact of a **reuse-oriented digital platform** (“zu verschenken” app)
on waste accumulation, municipal workload, and environmental performance.

The project is conducted as part of the **MSc Information Technology Management**
programme at **Berlin School of Business & Innovation**, within the module
*Digital Economy & Transformation*.

---

## Problem Context
In Berlin, residents frequently leave bulky items such as furniture,
electronics, and textiles on public streets labelled as “zu verschenken”.
While some items are successfully reused, a significant share remains
uncollected, degrades over time, and must eventually be removed by
Berliner Stadtreinigung (BSR).

This phenomenon creates:
- Visual and spatial pollution in public areas  
- Increased operational pressure on municipal waste services  
- Additional CO₂ emissions from collection logistics  

---

## Objective
The objective of this project is to assess whether a **digital reuse platform**
can reduce illegal bulky waste accumulation by **30–50%**, by increasing reuse
probabilities and decreasing the arrival rate of items requiring municipal
collection.

---

## Data Sources
The simulation integrates multiple official data sources:
- **Ordnungsamt-Online** – incident-level illegal dumping reports  
- **Amt für Statistik Berlin-Brandenburg** – population data by district (LOR)  
- **Berliner Stadtreinigung (BSR)** – Abfallbilanz 2023 (waste composition and factors)  
- **Berlin LOR geospatial boundaries** – spatial context for districts  

All raw datasets are stored unmodified and processed programmatically
to ensure transparency and reproducibility.

---

## Methodology
The modelling approach is based on:
- **Discrete Event Simulation (DES)** using SimPy  
- **Queueing logic (M/M/c)** to represent limited BSR collection capacity  
- **Scenario-based analysis**, comparing:
  - Baseline conditions (no digital intervention)
  - App-supported reuse scenarios  
- **Parameter calibration** using official municipal statistics  

This approach prioritises process flow and operational dynamics over
individual behavioural modelling.

---

## Repository Structure
```text
berlin-bulky-waste-simulation/
│
├── data/
│   ├── raw/                    # Original municipal datasets
│   │   ├── ordnungsamt_2023.json      # Illegal dumping incident reports
│   │   ├── population_2024.csv        # Population data by LOR district
│   │   ├── berlin_map.geojson         # Berlin LOR geospatial boundaries
│   │   └── waste_config.json          # Waste composition and parameters
│   │
│   └── processed/              # Cleaned and derived datasets
│
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data_loader.py          # Data ingestion and preprocessing
│   ├── simulation.py           # SimPy-based DES model
│   ├── analysis.py             # Metrics calculation and statistical analysis
│   ├── visualizations.py       # Visualization generation (charts, plots)
│   ├── choropleth.py           # Geospatial choropleth map generation
│   └── utils.py                # Helper functions and utilities
│
├── notebooks/
│   └── 01_data_exploration.ipynb   # Exploratory data analysis
│
├── outputs/
│   ├── logs/                   # Simulation run outputs (.csv)
│   ├── figures/                # Static plots and charts (.png)
│   └── visualizations/         # Generated visualization outputs
│
├── config/
│   ├── logs/                   # Configuration logs
│   └── figures/                # Configuration-related figures
│
├── berlin_waste_sim.egg-info/  # Package metadata (auto-generated)
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation configuration
└── README.md                   # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd berlin-bulky-waste-simulation
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode** (optional)
   ```bash
   pip install -e .
   ```

---

## Usage

### Running the Simulation

```python
from src.simulation import BulkyWasteSimulation
from src.data_loader import DataLoader

# Load data
loader = DataLoader()
data = loader.load_all_data()

# Initialize and run simulation
sim = BulkyWasteSimulation(data)
results = sim.run(duration=365, num_runs=100)
```

### Generating Visualizations

```python
from src.visualizations import generate_visualizations
from src.choropleth import create_choropleth_map

# Generate standard visualizations
generate_visualizations(results)

# Create geospatial choropleth maps
create_choropleth_map(data, output_path='outputs/visualizations/')
```

### Analyzing Results

```python
from src.analysis import analyze_results

# Perform statistical analysis
analysis = analyze_results(results)
analysis.summary()
```

---

## Key Features

- **Discrete Event Simulation (DES)**: SimPy-based model simulating waste item lifecycle
- **Multi-scenario Analysis**: Compare baseline vs. app-supported reuse scenarios
- **Geospatial Visualization**: Choropleth maps showing district-level waste patterns
- **Statistical Analysis**: Comprehensive metrics and performance indicators
- **Real Municipal Data**: Integration of official Berlin open data sources
- **Reproducible Research**: Transparent data processing and parameter calibration

---

## Simulation Scenarios

The model evaluates three primary scenarios:

1. **Baseline**: Current state without digital intervention
2. **Realistic**: Moderate app adoption with 30% waste reduction
3. **Pessimistic**: Lower app adoption with 15% waste reduction

Each scenario varies in:
- Item arrival rates
- Reuse probabilities
- Collection queue dynamics
- Municipal workload impact

---

## Output Files

### Logs
- `outputs/logs/simulation_results.csv` - Raw simulation event data
- `config/logs/` - Configuration and runtime logs

### Visualizations
- `outputs/figures/` - Statistical charts and plots (PNG)
- `outputs/visualizations/` - Generated visualization outputs
- `config/figures/` - Configuration-related visualizations

---

## Technologies Used

- **Simulation**: SimPy 4.1.1
- **Data Processing**: Pandas, NumPy, GeoPandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Geospatial**: Shapely, GeoJSON
- **Statistical Analysis**: SciPy, Scikit-learn, Statsmodels

---

## Academic Context

**Institution**: Berlin School of Business & Innovation (BSBI)  
**Programme**: MSc Information Technology Management  
**Module**: Digital Economy & Transformation  
**Focus**: Simulation-based policy evaluation for urban waste management

---

## License

This project is developed for academic purposes as part of the MSc IT Management programme.

---

## Contact

For questions or collaboration inquiries, please contact the BSBI MSc IT Management Team.

---

## Acknowledgements

- **Berliner Stadtreinigung (BSR)** - Waste management data
- **Amt für Statistik Berlin-Brandenburg** - Population statistics
- **Berlin Open Data Portal** - Geospatial and incident data