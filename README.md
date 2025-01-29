# Weather-Data-App
Handles weather data 
# Weather Data Analysis Tool

## Overview
This **Weather Data Analysis Tool** is an interactive **Streamlit** application that enables users to analyze, visualize, and model weather data. The tool provides insights into temperature distributions, correlations, and predictive modeling using machine learning.

## Features
- **Dataset Overview**: Displays basic information and statistics about the dataset.
- **Data Cleaning**: Handles missing values by filling them with column means.
- **Visualizations**:
  - Temperature Distribution Histogram
  - Correlation Heatmap for feature relationships
- **Custom Analysis**: Allows users to filter data based on a selected temperature range.
- **Machine Learning Model**:
  - Trains a **Random Forest Regressor** to predict temperature.
  - Evaluates model performance using **Mean Squared Error (MSE)**.
  - Displays **feature importance** using a bar chart.

## Installation
To run the application locally, follow these steps:

```bash
git clone https://github.com/your-repo/weather-data-analysis.git
cd weather-data-analysis
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## Usage
- Upload a dataset (`weather_data.csv` by default) with temperature and relevant features.
- Explore data summaries and visualizations.
- Train and evaluate a predictive model.
- Adjust parameters using interactive sliders in the sidebar.

## Dependencies
- **Python 3.x**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `streamlit`
- `scikit-learn`

## License
This project is open-source and available under the **MIT License**.

## Contribution
Feel free to submit **pull requests** or open **issues** for improvements.
