# 📂 Data Directory

This directory should contain the CEEW smart meter datasets.

## Required Files

| File | Description |
|------|-------------|
| `CEEW - Smart meter data Mathura 2019.csv` | Training data (2019 readings) |
| `CEEW - Smart meter data Mathura 2020.csv` | Testing data (2020 readings) |

## How to Obtain

1. Download the CEEW smart meter data from the official source
2. Place both CSV files in this `data/` directory
3. The filenames must match exactly as listed above

## Data Format

- Each row represents a smart meter reading
- Key column: `t_kWh` — total energy consumption in kilowatt-hours
- Time column is auto-detected during preprocessing

> **Note**: These files are not included in the repository due to their large size (~230 MB total). They are listed in `.gitignore`.
