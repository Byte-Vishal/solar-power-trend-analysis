import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

np.random.seed(42)
months = pd.date_range('2018-01', '2023-12', freq='MS')
n = len(months)
seasonal = np.sin((months.month - 3) * np.pi / 6) * 2 + 5

df = pd.DataFrame({
    'date':               months,
    'irradiance':         np.clip(seasonal + np.random.normal(0, 0.3, n), 1.5, 7.5).round(2),
    'temperature':        (seasonal * 4 + 15 + np.random.normal(0, 1.5, n)).round(1),
    'cloud_cover':        np.clip(40 - seasonal*3 + np.random.normal(0, 5, n), 5, 85).round(1),
    'installed_capacity': np.linspace(100, 250, n).round(1),
    'panel_efficiency':   np.clip(18 + np.random.normal(0, 0.5, n), 16, 20).round(2),
})
df['output'] = (df['irradiance'] * df['installed_capacity'] *
                (df['panel_efficiency']/100) * (1 - df['cloud_cover']/200) * 30).round(2)
df['year']   = df['date'].dt.year
df['month']  = df['date'].dt.month
df['season'] = df['month'].map({12:'Winter',1:'Winter',2:'Winter',
                                 3:'Spring',4:'Spring',5:'Spring',
                                 6:'Summer',7:'Summer',8:'Summer',
                                 9:'Autumn',10:'Autumn',11:'Autumn'})

for col in ['irradiance', 'temperature', 'cloud_cover']:
    df.loc[np.random.choice(df.index, 2), col] = np.nan
df.fillna(df.median(numeric_only=True), inplace=True)

df.to_csv('dataset/solar_dataset.csv', index=False)
print(f"Dataset: {df.shape} | Missing after imputation: {df.isnull().sum().sum()}")
print(df[['irradiance','temperature','cloud_cover','output']].describe().round(2))

plt.figure(figsize=(12, 4))
plt.plot(df['date'], df['output'], color='#E5831A', linewidth=2)
plt.fill_between(df['date'], df['output'], alpha=0.12, color='#E5831A')
plt.title('Monthly Solar Energy Output (2018-2023)')
plt.xlabel('Date'); plt.ylabel('Output (kWh)')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig('outputs/plot1_line_output.png', dpi=150); plt.close()

pivot = df.pivot_table(values='output', index='year', columns='month')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.figure(figsize=(12, 4))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5)
plt.title('Heatmap - Monthly Output by Year')
plt.tight_layout()
plt.savefig('outputs/plot2_heatmap.png', dpi=150); plt.close()

plt.figure(figsize=(7, 5))
sns.regplot(data=df, x='irradiance', y='output',
            scatter_kws={'alpha':0.7, 'color':'#E5831A', 'edgecolors':'k', 'linewidths':0.3},
            line_kws={'color':'#1F3864', 'linewidth':2}, ci=95)
plt.title('Scatter - Irradiance vs Energy Output (95% CI)')
plt.xlabel('Solar Irradiance (kWh/m2/day)'); plt.ylabel('Output (kWh)')
plt.tight_layout()
plt.savefig('outputs/plot3_scatter.png', dpi=150); plt.close()

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='season', y='output',
               order=['Spring','Summer','Autumn','Winter'],
               palette={'Spring':'#66BB6A','Summer':'#FFA726',
                        'Autumn':'#EF5350','Winter':'#42A5F5'},
               inner='quartile')
plt.title('Output Distribution by Season')
plt.xlabel('Season'); plt.ylabel('Output (kWh)')
plt.tight_layout()
plt.savefig('outputs/plot4_violin.png', dpi=150); plt.close()

features = ['irradiance','temperature','cloud_cover','installed_capacity']
corr = df[features + ['output']].corr()['output'].drop('output')
print(f"\nPearson r with output:\n{corr.round(3)}")

X = df[features]; y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\nR2  : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} kWh")
print("\nDone - plots saved to outputs/")
