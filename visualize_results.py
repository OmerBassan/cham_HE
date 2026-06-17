import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set premium aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# Data from the user's summaries
data = {
    'Model': ['Gemini 1.5 Flash', 'Pixtral Latest'],
    'Original Accuracy': [0.72, 0.72],
    'Distorted Accuracy': [0.6667, 0.6056],
    'Original Exact Match': [0.7867, 0.7867],
    'Distorted Exact Match': [0.7534, 0.6831]
}

df = pd.DataFrame(data)

# Calculate downgrade percentage
df['Accuracy Drop (%)'] = ((df['Original Accuracy'] - df['Distorted Accuracy']) / df['Original Accuracy']) * 100

# Prepare data for plotting (long format)
plot_df = df.melt(id_vars='Model', value_vars=['Original Accuracy', 'Distorted Accuracy'], 
                  var_name='Condition', value_name='Accuracy')

# Create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

# --- Plot 1: Grouped Bar Chart (Accuracy Comparison) ---
colors = ["#4A90E2", "#E94E77"]  # Soft Blue and vibrant Pink/Red
sns.barplot(data=plot_df, x='Model', y='Accuracy', hue='Condition', palette=colors, ax=ax1)

# Add value labels on bars
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.2%}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontweight='bold')

ax1.set_title('Model Performance: Original vs Distorted Input', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylim(0, 1.0)
ax1.set_ylabel('VQA Accuracy', fontsize=12)
ax1.set_xlabel('', fontsize=12)
ax1.legend(title='Input Type', frameon=True)

# --- Plot 2: Downgrade Percentage ---
sns.barplot(data=df, x='Model', y='Accuracy Drop (%)', color="#F5A623", ax=ax2)

# Add value labels
for p in ax2.patches:
    ax2.annotate(f'-{p.get_height():.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   color='#D0021B',
                   fontweight='bold')

ax2.set_title('Accuracy Downgrade', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylabel('Percentage Drop (%)', fontsize=12)
ax2.set_ylim(0, max(df['Accuracy Drop (%)']) * 1.3)
ax2.set_xlabel('', fontsize=12)

# Global styling
plt.tight_layout()

# Save the plot
output_path = 'benchmarks/accuracy_downgrade_report.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

# Displaying a summary table
print("\n" + "="*30)
print("BENCHMARK SUMMARY")
print("="*30)
print(df[['Model', 'Original Accuracy', 'Distorted Accuracy', 'Accuracy Drop (%)']].to_string(index=False))
print("="*30)
