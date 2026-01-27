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
Berlin-Smart-Waste-Sim/
│
├── data/
│   ├── raw/                    # Original municipal datasets
│   │   ├── ordnungsamt_2023.json
│   │   ├── population_2024.csv
│   │   ├── berlin_map.geojson
│   │   └── waste_config.json
│   │
│   └── processed/              # Cleaned and derived datasets
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion and preprocessing
│   ├── simulation.py           # SimPy-based DES model
│   ├── analysis.py             # Metrics calculation and visualisation
│   └── utils.py                # Helper functions
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── outputs/
│   ├── logs/                   # Simulation run outputs (.csv)
│   └── figures/                # Plots and charts (.png)
│
├── requirements.txt
└── README.md