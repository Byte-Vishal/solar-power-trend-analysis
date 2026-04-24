import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../dataset', exist_ok=True)

np.random.seed(42)

months = pd.date_range(start='2018-01', end='2023-12', freq='MS')
n = len(months)

seasonal = np.sin((months.month - 3) * np.pi / 6) * 2 + 5   # kWh/m²/day

solar_data = pd.DataFrame({
    'date':               months,
    'irradiance_kwh_m2':  np.clip(seasonal + np.random.normal(0, 0.3, n), 1.5, 7.5).round(2),
    'temperature_c':      (seasonal * 4 + 15 + np.random.normal(0, 1.5, n)).round(1),
    'cloud_cover_pct':    np.clip(40 - seasonal * 3 + np.random.normal(0, 5, n), 5, 85).round(1),
    'panel_efficiency_pct': np.clip(18 + np.random.normal(0, 0.5, n), 16, 20).round(2),
    'installed_capacity_kw': np.linspace(100, 250, n).round(1),   # capacity grows over time
    'maintenance_flag':   np.random.choice([0, 1], n, p=[0.9, 0.1]),
})

solar_data['energy_output_kwh'] = (
    solar_data['irradiance_kwh_m2'] *
    solar_data['installed_capacity_kw'] *
    (solar_data['panel_efficiency_pct'] / 100) *
    (1 - solar_data['cloud_cover_pct'] / 200) *
    30   # ~30 days per month
).round(2)

for col in ['irradiance_kwh_m2', 'temperature_c', 'cloud_cover_pct']:
    idx = np.random.choice(solar_data.index, 2, replace=False)
    solar_data.loc[idx, col] = np.nan

solar_data.to_csv('../dataset/solar_dataset.csv', index=False)
print("✓ Dataset saved to dataset/solar_dataset.csv")
print(f"  Shape  : {solar_data.shape}")
print(f"  Columns: {list(solar_data.columns)}")

print("\n── 2. Preprocessing ─────────────────────────────────────────────────")

df = pd.read_csv('../dataset/solar_dataset.csv', parse_dates=['date'])

print(f"Missing values before imputation:\n{df.isnull().sum()}")

num_cols_all = df.select_dtypes(include=np.number).columns
df[num_cols_all] = df[num_cols_all].apply(lambda c: c.fillna(c.median()))

print(f"\nMissing values after imputation:\n{df.isnull().sum()}")

df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df['season'] = df['month'].map({
    12:'Winter', 1:'Winter', 2:'Winter',
     3:'Spring', 4:'Spring', 5:'Spring',
     6:'Summer', 7:'Summer', 8:'Summer',
     9:'Autumn',10:'Autumn',11:'Autumn'
})

scaler = StandardScaler()
feature_cols = ['irradiance_kwh_m2', 'temperature_c', 'cloud_cover_pct',
                'panel_efficiency_pct', 'installed_capacity_kw']
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

print("\nDescriptive Statistics (raw data):")
print(df[feature_cols + ['energy_output_kwh']].describe().round(2))

print("\n── 3. EDA ───────────────────────────────────────────────────────────")

annual = df.groupby('year')['energy_output_kwh'].sum().reset_index()
annual.columns = ['year', 'total_output_kwh']
print("\nAnnual Total Energy Output (kWh):")
print(annual.to_string(index=False))

monthly_avg = df.groupby('month')['energy_output_kwh'].mean().reset_index()

seasonal_avg = df.groupby('season')['energy_output_kwh'].mean().round(2)
print(f"\nSeasonal Average Output (kWh):\n{seasonal_avg}")

print("\n── 4. Plots ─────────────────────────────────────────────────────────")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold'})

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['date'], df['energy_output_kwh'], color='#E5831A', linewidth=1.5,
        label='Monthly Output')
ax.fill_between(df['date'], df['energy_output_kwh'], alpha=0.15, color='#E5831A')

for yr, grp in df.groupby('year'):
    ax.hlines(grp['energy_output_kwh'].mean(), grp['date'].min(), grp['date'].max(),
              colors='#333', linestyles='--', linewidth=0.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
ax.set_xlabel('Date'); ax.set_ylabel('Energy Output (kWh)')
ax.set_title('Figure 1: Monthly Solar Energy Output (2018–2023)')
ax.legend(); plt.tight_layout()
plt.savefig('../outputs/fig1_monthly_output_line.png', dpi=150); plt.close()
print("  Saved: fig1_monthly_output_line.png")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(annual['year'].astype(str), annual['total_output_kwh'],
              color='#2E75B6', edgecolor='#1a4f80', width=0.6)
for bar, val in zip(bars, annual['total_output_kwh']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val:,.0f}', ha='center', va='bottom', fontsize=10)
ax.set_xlabel('Year'); ax.set_ylabel('Total Energy Output (kWh)')
ax.set_title('Figure 2: Annual Total Solar Energy Output')
plt.tight_layout()
plt.savefig('../outputs/fig2_annual_output_bar.png', dpi=150); plt.close()
print("  Saved: fig2_annual_output_bar.png")

num_cols = ['irradiance_kwh_m2', 'temperature_c', 'cloud_cover_pct',
            'panel_efficiency_pct', 'installed_capacity_kw', 'energy_output_kwh']
fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, center=0, ax=ax,
            xticklabels=['Irradiance','Temp','Cloud','Efficiency','Capacity','Output'],
            yticklabels=['Irradiance','Temp','Cloud','Efficiency','Capacity','Output'])
ax.set_title('Figure 3: Pearson Correlation Heatmap — All Features')
plt.tight_layout()
plt.savefig('../outputs/fig3_correlation_heatmap.png', dpi=150); plt.close()
print("  Saved: fig3_correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df['irradiance_kwh_m2'], df['energy_output_kwh'],
                c=df['month'], cmap='plasma', s=60, alpha=0.8, edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ax=ax, label='Month')
ax.set_xlabel('Solar Irradiance (kWh/m²/day)')
ax.set_ylabel('Energy Output (kWh)')
ax.set_title('Figure 4: Irradiance vs Energy Output (coloured by Month)')
plt.tight_layout()
plt.savefig('../outputs/fig4_irradiance_vs_output_scatter.png', dpi=150); plt.close()
print("  Saved: fig4_irradiance_vs_output_scatter.png")

season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
fig, ax = plt.subplots(figsize=(8, 5))
palette = {'Spring':'#66BB6A','Summer':'#FFA726','Autumn':'#EF5350','Winter':'#42A5F5'}
sns.boxplot(data=df, x='season', y='energy_output_kwh',
            order=season_order, palette=palette, ax=ax, width=0.5)
ax.set_xlabel('Season'); ax.set_ylabel('Energy Output (kWh)')
ax.set_title('Figure 5: Energy Output Distribution by Season')
plt.tight_layout()
plt.savefig('../outputs/fig5_seasonal_boxplot.png', dpi=150); plt.close()
print("  Saved: fig5_seasonal_boxplot.png")

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(month_names, monthly_avg['energy_output_kwh'], marker='o',
        color='#E5831A', linewidth=2, markersize=7)
ax.fill_between(month_names, monthly_avg['energy_output_kwh'], alpha=0.1, color='#E5831A')
ax.set_xlabel('Month'); ax.set_ylabel('Avg Energy Output (kWh)')
ax.set_title('Figure 6: Average Monthly Solar Output Pattern (2018–2023)')
plt.tight_layout()
plt.savefig('../outputs/fig6_avg_monthly_pattern.png', dpi=150); plt.close()
print("  Saved: fig6_avg_monthly_pattern.png")

print("\n── 5. Feature Selection (Pearson |r| > 0.2) ─────────────────────────")
corr_target = df[num_cols].corr()['energy_output_kwh'].drop('energy_output_kwh')
print("\nPearson r with energy_output_kwh:")
print(corr_target.round(3).to_string())

selected_features = corr_target[abs(corr_target) > 0.2].index.tolist()
print(f"\nSelected features: {selected_features}")

print("\n── 6. Model: Linear Regression ──────────────────────────────────────")

X = df_scaled[selected_features]
y = df['energy_output_kwh']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"  R² Score : {r2:.4f}")
print(f"  RMSE     : {rmse:.2f} kWh")
print(f"  MAE      : {mae:.2f} kWh")

coeff_df = pd.DataFrame({
    'Feature':     selected_features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print("\nModel Coefficients:")
print(coeff_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, y_pred, alpha=0.7, color='#2E75B6', edgecolors='k', linewidths=0.3)
lims = [min(y_test.min(), y_pred.min()) - 50, max(y_test.max(), y_pred.max()) + 50]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual Output (kWh)'); ax.set_ylabel('Predicted Output (kWh)')
ax.set_title(f'Figure 7: Actual vs Predicted (R² = {r2:.3f})')
ax.legend(); plt.tight_layout()
plt.savefig('../outputs/fig7_actual_vs_predicted.png', dpi=150); plt.close()
print("  Saved: fig7_actual_vs_predicted.png")

print("\n── 7. Key Insights ──────────────────────────────────────────────────")
insights = [
    f"1. Total output grew from {annual.iloc[0]['total_output_kwh']:,.0f} kWh (2018) to "
    f"{annual.iloc[-1]['total_output_kwh']:,.0f} kWh (2023) — "
    f"{(annual.iloc[-1]['total_output_kwh']/annual.iloc[0]['total_output_kwh']-1)*100:.1f}% increase.",
    f"2. Summer months yield the highest average output; winter the lowest.",
    f"3. Irradiance (r = {corr_target['irradiance_kwh_m2']:.2f}) and installed capacity "
    f"(r = {corr_target['installed_capacity_kw']:.2f}) are the strongest predictors.",
    f"4. Cloud cover shows negative correlation (r = {corr_target['cloud_cover_pct']:.2f}) "
    f"with output.",
    f"5. Linear Regression achieves R² = {r2:.3f}, confirming near-linear relationships.",
    f"6. Installed capacity increased steadily from 100 kW to 250 kW over 6 years.",
]
for i in insights:
    print(f"  {i}")

print("\n✓ All outputs saved to outputs/")
