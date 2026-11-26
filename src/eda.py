import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/ai4i2020.csv")
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Target Class Distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Machine failure', hue='Machine failure', palette=['#3498db', '#e74c3c'], legend=False)
plt.title('Target Class Distribution', fontweight='bold')
plt.xlabel('Machine State (0: Normal, 1: Failure)')
plt.ylabel('Count')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')
plt.tight_layout()
plt.savefig('figures/imbalance.png', dpi=300)

# 2. Feature Interaction: Torque vs. RPM
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='Rotational speed [rpm]', y='Torque [Nm]', 
                hue='Machine failure', palette=['#3498db', '#e74c3c'], alpha=0.6, s=50)
plt.title('Feature Interaction: Torque vs. Rotational Speed', fontweight='bold')
plt.xlabel('Rotational Speed [rpm]')
plt.ylabel('Torque [Nm]')
plt.legend(title='State', labels=['0: Normal', '1: Failure'])
plt.tight_layout()
plt.savefig('figures/scatter.png', dpi=300)

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
corr_matrix = df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/heatmap.png', dpi=300)

# 4. Boxplot: Torque Distribution 
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='Machine failure', y='Torque [Nm]', hue='Machine failure', palette=['#3498db', '#e74c3c'], legend=False)
plt.title('Torque Distribution by Machine State', fontweight='bold')
plt.xlabel('Machine State (0: Normal, 1: Failure)')
plt.ylabel('Torque [Nm]')

plt.xticks([0, 1], ['0: Normal', '1: Failure'])
plt.tight_layout()
plt.savefig('figures/boxplot_torque.png', dpi=300)