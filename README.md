# Seasonality-Adjusted Unique Visitor Forecast App

This interactive Streamlit app forecasts monthly unique visitors for the current year (e.g., 2025), using last year's data (e.g., 2024) and adjusting for seasonal patterns.

## Overview

- Upload your own monthly visitor data or use the provided sample.
- Automatically calculates seasonal indices from historical data.
- Applies linear regression on deseasonalized data to forecast missing values.
- Reseasonalizes the predictions to reflect realistic seasonal effects.
- Visualizes actual vs. predicted data.
- Download the adjusted forecast in CSV format.

---

## File Structure
seasonality_forecast_app/
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
└── sample_data.csv # Sample input file


---

## Sample CSV Format

The uploaded file (or the default sample) must follow this format:

```csv
month,last_year_data,this_year_data
Jan,853892,564649
Feb,753225,501947
Mar,1012344,612701
...

