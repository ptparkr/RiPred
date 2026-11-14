import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/rice.csv')
sns.set_theme(style='whitegrid', palette='muted')

# 1. Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
all_median = df['rice_yield_kgha'].median()
sns.histplot(df['rice_yield_kgha'], bins=40, kde=True, ax=axes[0], color='#2E7D32')
axes[0].set_title('All Observations (N=3744) - Includes Zero-Yield Districts', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Rice Yield (kg/ha)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(all_median, color='red', linestyle='--', label=f'Median: {all_median:.1f}')
axes[0].legend()

positive_yield = df[df['rice_yield_kgha'] > 0]['rice_yield_kgha']
pos_median = positive_yield.median()
sns.histplot(positive_yield, bins=40, kde=True, ax=axes[1], color='#1565C0')
axes[1].set_title('Active Rice-Growing Observations (N=3400, Yield > 0)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Rice Yield (kg/ha)')
axes[1].set_ylabel('Frequency')
axes[1].axvline(pos_median, color='red', linestyle='--', label=f'Median: {pos_median:.1f}')
axes[1].legend()
plt.tight_layout()
plt.savefig('reports/figures/target_distribution.png', dpi=200)
plt.close()

# 2. State-wise Yield Comparison
fig, ax = plt.subplots(figsize=(14, 7))
state_order = df[df['rice_yield_kgha'] > 0].groupby('state')['rice_yield_kgha'].median().sort_values(ascending=False).index
sns.boxplot(data=df[df['rice_yield_kgha'] > 0], x='state', y='rice_yield_kgha', order=state_order, palette='viridis', ax=ax)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.title('Rice Yield Distribution by State (Active Cultivation, Sorted by Median)', fontsize=14, fontweight='bold')
plt.xlabel('State', fontsize=12)
plt.ylabel('Rice Yield (kg/ha)', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/state_yield_comparison.png', dpi=200)
plt.close()

# 3. Yearly Trends
fig, ax = plt.subplots(figsize=(12, 6))
yearly_stats = df[df['rice_yield_kgha'] > 0].groupby('year')['rice_yield_kgha'].agg(
    Mean='mean',
    Median='median',
    Q25=lambda x: np.percentile(x, 25),
    Q75=lambda x: np.percentile(x, 75)
)
ax.plot(yearly_stats.index, yearly_stats['Mean'], marker='o', linewidth=2.5, color='#2E7D32', label='National Mean (kg/ha)')
ax.plot(yearly_stats.index, yearly_stats['Median'], marker='s', linewidth=2, linestyle='--', color='#1565C0', label='National Median (kg/ha)')
ax.fill_between(yearly_stats.index, yearly_stats['Q25'], yearly_stats['Q75'], alpha=0.2, color='#2E7D32', label='IQR (25th - 75th percentile)')
ax.set_title('National Rice Yield Trajectory (2000 - 2015)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Yield (kg/ha)', fontsize=12)
ax.set_xticks(yearly_stats.index)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('reports/figures/yearly_trend.png', dpi=200)
plt.close()

# 4. Correlation Heatmap
key_cols = [
    'rice_yield_kgha', 'n_per_ha_nca', 'p_per_ha_nca', 'k_per_ha_nca',
    'lgp_days', 'rain_monsoon_mm', 'rain_annual_mm', 'water_stress_score',
    'irrig_canal_area_1000ha', 'irrig_tubewell_area_1000ha',
    'tmax_monsoon', 'tmin_monsoon', 'aet_monsoon'
]
corr_matrix = df[key_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.6, vmax=1.0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix: Yield and Key Agro-Climatic Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png', dpi=200)
plt.close()

# 5. Water Stress vs Yield
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df[df['rice_yield_kgha'] > 0], x='stress_group', y='rice_yield_kgha', palette='YlOrRd', ax=ax)
ax.set_title('Rice Yield across Water Stress Quintile Groups (0: Minimal Stress -> 4: Extreme Stress)', fontsize=13, fontweight='bold')
ax.set_xlabel('Stress Group (Quintile)', fontsize=12)
ax.set_ylabel('Rice Yield (kg/ha)', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/water_stress_vs_yield.png', dpi=200)
plt.close()

# 6. Zero Yield Analysis by State
fig, ax = plt.subplots(figsize=(10, 5))
zero_counts = df[df['rice_yield_kgha'] == 0]['state'].value_counts()
zero_counts.plot(kind='bar', color='#D32F2F', ax=ax)
ax.set_title('Observations with Zero Rice Yield by State (Total = 344)', fontsize=13, fontweight='bold')
ax.set_xlabel('State', fontsize=12)
ax.set_ylabel('Number of Zero-Yield Records', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/figures/zero_yield_districts.png', dpi=200)
plt.close()

print('All 6 figures successfully generated in reports/figures/')
