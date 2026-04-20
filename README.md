# ☀️ Solar Power Trend Analysis

> **Course:** 21CSS303T — Data Science
> **Department:** Electronics & Communication Engineering (W/s in Data Science)  
> **Institution:** SRM Institute of Science & Technology, Vadapalani  
> **Submitted by:** Vishal. M — RA2311053040040 
> **Guide:** Dr. Dinesh Babu


## 📋 Project Overview

This project performs a comprehensive trend analysis of solar power generation data using supervised machine learning and statistical methods. The goal is to uncover seasonal patterns, identify key predictors of energy output, and build a regression model for output forecasting — supporting informed decision-making in renewable energy planning.


## 📁 Repository Structure

solar_power_project/
│
├── dataset/
│   └── solar_dataset.csv          # Monthly solar power dataset (2018–2023)
│
├── notebooks/
│   └── solar_analysis.ipynb       # Jupyter notebook version
│
├── src/
│   └── solar_analysis.py          # Main Python analysis script
│
├── outputs/
│   ├── fig1_monthly_output_line.png
│   ├── fig2_annual_output_bar.png
│   ├── fig3_correlation_heatmap.png
│   ├── fig4_irradiance_vs_output_scatter.png
│   ├── fig5_seasonal_boxplot.png
│   ├── fig6_avg_monthly_pattern.png
│   └── fig7_actual_vs_predicted.png
│
├── README.md
└── requirements.txt


## 📊 Dataset Description

| Feature | Description | Unit |
|---|---|---|
| `date` | Month-year timestamp | YYYY-MM |
| `irradiance_kwh_m2` | Solar irradiance received | kWh/m²/day |
| `temperature_c` | Ambient temperature | °C |
| `cloud_cover_pct` | Percentage cloud cover | % |
| `panel_efficiency_pct` | Photovoltaic panel efficiency | % |
| `installed_capacity_kw` | Installed solar capacity | kW |
| `maintenance_flag` | Whether maintenance occurred (0/1) | Binary |
| `energy_output_kwh` | **Target** — monthly energy output | kWh |

- **Records:** 72 (monthly, Jan 2018 – Dec 2023)  
- **Missing values:** ~6 entries (handled via median imputation)


## 🔧 Steps Performed

1. **Data Loading & Generation** — Synthetic dataset with realistic physics-based formula
2. **Preprocessing** — Median imputation, Z-score normalisation, feature engineering (season, year, month)
3. **EDA** — Descriptive statistics, class distribution, seasonal decomposition
4. **Feature Selection** — Pearson correlation (|r| > 0.2); 4 of 5 features selected
5. **Visualisation** — 7 plots: line, bar, heatmap, scatter, box plot
6. **Regression Model** — Linear Regression with 80/20 train-test split
7. **Evaluation** — R², RMSE, MAE; actual vs predicted plot


## 📈 Results

| Metric | Value |
|---|---|
| R² Score | **0.922** |
| RMSE | 448.87 kWh |
| MAE | 371.25 kWh |

### Top Predictors (Pearson r with output)
| Feature | r |
|---|---|
| Solar Irradiance | +0.77 |
| Temperature | +0.75 |
| Installed Capacity | +0.59 |
| Cloud Cover | −0.41 |


## 💡 Key Insights

1. Total annual output **grew 108.8%** from 2018 to 2023 (32,217 → 67,271 kWh)
2. **Summer** months generate 2.1× more energy than **Winter** months
3. **Solar irradiance** is the single strongest predictor (r = 0.77)
4. **Cloud cover** has a consistent negative impact on generation
5. **Linear Regression** explains 92.2% of variance — near-linear relationships confirmed
6. Installed capacity scaling is the primary driver of long-term growth


## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Analysis
```bash
cd src/
python solar_analysis.py
```
Outputs (graphs + console stats) will be saved to `outputs/`.

### Jupyter Notebook
```bash
jupyter notebook notebooks/solar_analysis.ipynb
```


## 🔮 Future Work

- Incorporate real NREL/NASA weather + irradiance datasets
- Test advanced models: Random Forest, XGBoost, LSTM
- Add SHAP explainability for feature importance
- Build a Streamlit dashboard for interactive analysis
- Include grid-connected vs off-grid comparison


## 📚 References

1. Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. UC Irvine.
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.
3. World Bank (2023). Solar Resource Data. Global Solar Atlas.
4. IEA (2023). Renewables 2023 — Analysis and forecast to 2028.
