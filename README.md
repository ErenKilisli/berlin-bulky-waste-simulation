# Berlin Bulky Waste Simulation

## Project Overview
This project simulates bulky waste incidents in Berlin using Discrete Event Simulation (DES).
Real-world open data from Ordnungsamt-Online, population statistics, and BSR Abfallbilanz
are combined to analyse the impact of a reuse-focused digital platform.

## Data Sources
- Ordnungsamt-Online (incident-level reports)
- Amt für Statistik Berlin-Brandenburg (population by LOR)
- Berliner Stadtreinigung – Abfallbilanz 2023
- Berlin LOR geospatial boundaries

## Methodology
- Discrete Event Simulation (DES)
- Scenario-based modelling (baseline vs app-supported reuse)
- Parameter calibration using official municipal reports

## Repository Structure
Berlin-Smart-Waste-Sim/
│
├── data/                       # SENİN ELİNDEKİ DOSYALAR BURAYA
│   ├── raw/
│   │   ├── ordnungsamt_2023.json
│   │   ├── population_2024.csv
│   │   ├── berlin_map.geojson
│   │   └── waste_config.json
│   └── processed/              # Python ile temizleyip buraya kaydedeceğiz
│
├── src/                        # KAYNAK KODLAR
│   ├── __init__.py
│   ├── data_loader.py          # Verileri okuyan ve birleştiren script
│   ├── simulation.py           # SimPy mantığının döndüğü yer
│   ├── analysis.py             # Grafikleri çizen kod
│   └── utils.py                # Yardımcı fonksiyonlar (Mesafe hesabı vb.)
│
├── notebooks/                  # JUPYTER NOTEBOOKS (Denemeler için)
│   └── 01_data_exploration.ipynb
│
├── outputs/                    # SİMÜLASYON SONUÇLARI
│   ├── logs/                   # .csv log dosyaları
│   └── figures/                # .png grafikler
│
├── requirements.txt            # Kütüphaneler (pandas, simpy, geopandas...)
└── README.md                   # Projenin vitrini

## How to Run
pip install -r requirements.txt
python src/run_simulation.py
