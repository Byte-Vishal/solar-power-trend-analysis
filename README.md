# ☀️ Solar Power Trend Analysis

> **Course:** 21CSS303T — Data Science
> **Department:** Electronics & Communication Engineering (W/s in Data Science)  
> **Institution:** SRM Institute of Science & Technology, Vadapalani  
> **Submitted by:** Vishal. M — RA2311053040040 
> **Guide:** Dr. Dinesh Babu


## 📋 Project Overview

This project performs a comprehensive data-driven analysis of solar photovoltaic (PV) energy generation using a monthly dataset spanning January 2018 to December 2023 (72 records, 7 features). By leveraging Python libraries such as Pandas for data manipulation, Seaborn and Matplotlib for visualization, and Scikit-learn for machine learning, the study conducts a full Exploratory Data Analysis (EDA) pipeline to identify the key environmental and infrastructure drivers of solar energy output.

The analysis focuses on the relationships between Solar Irradiance, Temperature, Cloud Cover, and Installed Capacity against monthly energy output, while accounting for seasonal trends (Summer/Winter cycles) and long-term capacity growth. Initial statistical summaries reveal strong positive correlations between irradiance and output (r = 0.77) and temperature and output (r = 0.75), alongside a moderate negative correlation with cloud cover (r = −0.41), suggesting that irradiance and capacity are the dominant drivers of generation volume. Panel efficiency showed a weak correlation (r = 0.187) and was excluded from model features via a |r| > 0.2 threshold.

The workflow includes rigorous data cleaning — handling 6 missing values via median imputation and Z-score normalization — followed by Pearson correlation-based feature selection. A Linear Regression model trained on an 80/20 stratified split achieves an R² of 0.922 and RMSE of 448.87 kWh, confirming near-linear relationships between the selected features and energy output. Annual output grew 108.8% over the study period (32,217 kWh in 2018 → 67,271 kWh in 2023), driven by progressive expansion of installed capacity from 100 kW to 250 kW. This project provides a foundational framework for developing advanced predictive models aimed at optimizing renewable energy planning, grid scheduling, and solar farm investment decisions.
methods. Renewable Energy Reviews.
