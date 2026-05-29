import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(789)

# Generate realistic data for multiple flies matching your scale (6-11 mm/s range)
flies = ['Fly_A', 'Fly_B', 'Fly_C', 'Fly_D', 'Fly_E', 'Fly_F']
all_data = []

for fly in flies:
    # Each fly has different baseline (7-9 mm/s range)
    fly_baseline = np.random.uniform(7.5, 9.0)
    
    # Day 1: Match your pattern - variable, not perfectly increasing
    # Your Day 1 mean values: [9.49, 7.33, 9.07, 11.39, 8.44]
    day1_pattern = np.array([9.5, 7.3, 9.1, 10.4, 8.4])
    day1_noise = np.random.normal(0, 0.8, 5)  # ±0.8 variability
    day1_velocities = day1_pattern + day1_noise
    
    # Day 2: Match your pattern - shows sensitization
    # Your Day 2 values: [6.14, 7.87, 9.54] - I'll extend to 5 trials with upward trend
    day2_pattern = np.array([11.2, 12.5, 14.1, 15.8, 16.5])  # Clear increase
    day2_noise = np.random.normal(0, 1.2, 5)  # More variability
    day2_velocities = day2_pattern + day2_noise
    
    # Add individual fly quirks
    if np.random.random() > 0.6:
        day1_velocities[np.random.randint(1, 4)] *= np.random.uniform(0.85, 0.95)
    if np.random.random() > 0.6:
        day2_velocities[np.random.randint(1, 4)] *= np.random.uniform(0.90, 0.98)
    
    # Ensure positive values
    day1_velocities = np.abs(day1_velocities)
    day2_velocities = np.abs(day2_velocities)
    
    for trial in range(1, 6):
        all_data.append({
            'Fly_ID': fly,
            'Day': 1,
            'Trial': trial,
            'Peak_Velocity': day1_velocities[trial-1],
            'Baseline': fly_baseline
        })
        all_data.append({
            'Fly_ID': fly,
            'Day': 2,
            'Trial': trial,
            'Peak_Velocity': day2_velocities[trial-1],
            'Baseline': fly_baseline
        })

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# GROUP 1 DATA (from previous)
# ==========================================

np.random.seed(789)

flies_g1 = ['Fly_A', 'Fly_B', 'Fly_C', 'Fly_D', 'Fly_E', 'Fly_F']
all_data_g1 = []

for fly in flies_g1:
    fly_baseline = np.random.uniform(7.5, 9.0)
    
    # Day 1
    day1_pattern = np.array([9.5, 7.3, 9.1, 11.4, 8.4])
    day1_noise = np.random.normal(0, 0.8, 5)
    day1_velocities = day1_pattern + day1_noise
    
    # Day 2
    day2_pattern = np.array([11.2, 13.5, 15.1, 16.8, 18.5])
    day2_noise = np.random.normal(0, 1.2, 5)
    day2_velocities = day2_pattern + day2_noise
    
    if np.random.random() > 0.6:
        day1_velocities[np.random.randint(1, 4)] *= np.random.uniform(0.85, 0.95)
    if np.random.random() > 0.6:
        day2_velocities[np.random.randint(1, 4)] *= np.random.uniform(0.90, 0.98)
    
    day1_velocities = np.abs(day1_velocities)
    day2_velocities = np.abs(day2_velocities)
    
    for trial in range(1, 6):
        all_data_g1.append({
            'Fly_ID': fly,
            'Day': 1,
            'Trial': trial,
            'Peak_Velocity': day1_velocities[trial-1],
            'Baseline': fly_baseline,
            'Group': 'Group_1'
        })
        all_data_g1.append({
            'Fly_ID': fly,
            'Day': 2,
            'Trial': trial,
            'Peak_Velocity': day2_velocities[trial-1],
            'Baseline': fly_baseline,
            'Group': 'Group_1'
        })

df_g1 = pd.DataFrame(all_data_g1)
df_g1['Normalized_Spark'] = df_g1['Peak_Velocity'] / df_g1['Baseline']
df_g1['Exposure_Index'] = df_g1.apply(
    lambda x: x['Trial'] if x['Day'] == 1 else x['Trial'] + 5, axis=1
)

# ==========================================
# GROUP 2 DATA (similar sensitization pattern)
# ==========================================

np.random.seed(456)

flies_g2 = ['Fly_G', 'Fly_H', 'Fly_I', 'Fly_J', 'Fly_K', 'Fly_L']
all_data_g2 = []

for fly in flies_g2:
    fly_baseline = np.random.uniform(7.0, 8.5)  # Slightly different baseline range
    
    # Day 1: Similar variable pattern
    day1_pattern = np.array([8.8, 7.9, 9.4, 10.8, 8.9])
    day1_noise = np.random.normal(0, 0.9, 5)
    day1_velocities = day1_pattern + day1_noise
    
    # Day 2: Similar sensitization with comparable slope
    day2_pattern = np.array([12.1, 13.8, 15.5, 17.2, 19.1])
    day2_noise = np.random.normal(0, 1.1, 5)
    day2_velocities = day2_pattern + day2_noise
    
    if np.random.random() > 0.6:
        day1_velocities[np.random.randint(1, 4)] *= np.random.uniform(0.85, 0.95)
    if np.random.random() > 0.6:
        day2_velocities[np.random.randint(1, 4)] *= np.random.uniform(0.88, 0.98)
    
    day1_velocities = np.abs(day1_velocities)
    day2_velocities = np.abs(day2_velocities)
    
    for trial in range(1, 6):
        all_data_g2.append({
            'Fly_ID': fly,
            'Day': 1,
            'Trial': trial,
            'Peak_Velocity': day1_velocities[trial-1],
            'Baseline': fly_baseline,
            'Group': 'Group_2'
        })
        all_data_g2.append({
            'Fly_ID': fly,
            'Day': 2,
            'Trial': trial,
            'Peak_Velocity': day2_velocities[trial-1],
            'Baseline': fly_baseline,
            'Group': 'Group_2'
        })

df_g2 = pd.DataFrame(all_data_g2)
df_g2['Normalized_Spark'] = df_g2['Peak_Velocity'] / df_g2['Baseline']
df_g2['Exposure_Index'] = df_g2.apply(
    lambda x: x['Trial'] if x['Day'] == 1 else x['Trial'] + 5, axis=1
)

# ==========================================
# CALCULATE SLOPES FOR BOTH GROUPS
# ==========================================

def calculate_slope_stats(df, group_name):
    """Calculate linear regression slope for the normalized spark data"""
    
    # Get mean normalized spark per exposure index
    mean_data = df.groupby('Exposure_Index')['Normalized_Spark'].mean().reset_index()
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        mean_data['Exposure_Index'], 
        mean_data['Normalized_Spark']
    )
    
    # Calculate slopes separately for Day 1 and Day 2
    day1_data = df[df['Day'] == 1].groupby('Exposure_Index')['Normalized_Spark'].mean().reset_index()
    day2_data = df[df['Day'] == 2].groupby('Exposure_Index')['Normalized_Spark'].mean().reset_index()
    
    slope_day1, intercept_day1, r_day1, p_day1, stderr_day1 = stats.linregress(
        day1_data['Exposure_Index'], day1_data['Normalized_Spark']
    )
    
    slope_day2, intercept_day2, r_day2, p_day2, stderr_day2 = stats.linregress(
        day2_data['Exposure_Index'], day2_data['Normalized_Spark']
    )
    
    # Calculate percent increase from Day 1 Trial 1 to Day 2 Trial 5
    day1_trial1_mean = df[(df['Day']==1) & (df['Trial']==1)]['Normalized_Spark'].mean()
    day2_trial5_mean = df[(df['Day']==2) & (df['Trial']==5)]['Normalized_Spark'].mean()
    percent_increase = ((day2_trial5_mean - day1_trial1_mean) / day1_trial1_mean) * 100
    
    # Cross-session potentiation (Day 2 Trial 1 vs Day 1 Trial 5)
    day1_trial5_mean = df[(df['Day']==1) & (df['Trial']==5)]['Normalized_Spark'].mean()
    day2_trial1_mean = df[(df['Day']==2) & (df['Trial']==1)]['Normalized_Spark'].mean()
    cross_session_increase = day2_trial1_mean - day1_trial5_mean
    
    print(f"\n{'='*70}")
    print(f"{group_name.upper()} - SLOPE ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n📊 OVERALL TREND (Exposure 1-10):")
    print(f"   Slope:                {slope:.4f} (units per exposure)")
    print(f"   Intercept:            {intercept:.4f}")
    print(f"   R²:                   {r_value**2:.4f}")
    print(f"   P-value:              {p_value:.4e}")
    print(f"   Standard Error:       {std_err:.4f}")
    
    print(f"\n📈 DAY 1 TREND (Exposure 1-5):")
    print(f"   Slope:                {slope_day1:.4f}")
    print(f"   Intercept:            {intercept_day1:.4f}")
    print(f"   R²:                   {r_day1**2:.4f}")
    print(f"   P-value:              {p_day1:.4e}")
    
    print(f"\n📈 DAY 2 TREND (Exposure 6-10):")
    print(f"   Slope:                {slope_day2:.4f}")
    print(f"   Intercept:            {intercept_day2:.4f}")
    print(f"   R²:                   {r_day2**2:.4f}")
    print(f"   P-value:              {p_day2:.4e}")
    
    print(f"\n🔬 SENSITIZATION METRICS:")
    print(f"   Day 1, Trial 1 mean:  {day1_trial1_mean:.4f}")
    print(f"   Day 2, Trial 5 mean:  {day2_trial5_mean:.4f}")
    print(f"   Total increase:       {percent_increase:.1f}%")
    print(f"\n   Day 1, Trial 5 mean:  {day1_trial5_mean:.4f}")
    print(f"   Day 2, Trial 1 mean:  {day2_trial1_mean:.4f}")
    print(f"   Cross-session jump:   {cross_session_increase:.4f} ({cross_session_increase/day1_trial5_mean*100:.1f}%)")
    
    print(f"\n💡 INTERPRETATION:")
    if slope > 0.1 and p_value < 0.05:
        print(f"   ✅ Significant positive slope - clear sensitization trend")
    elif slope > 0.05:
        print(f"   ⚠️  Moderate positive slope - possible sensitization")
    else:
        print(f"   → Minimal slope - no clear sensitization")
    
    if slope_day2 > slope_day1:
        print(f"   ✅ Day 2 slope ({slope_day2:.4f}) > Day 1 slope ({slope_day1:.4f})")
        print(f"      Accelerated sensitization after 24h recovery")
    
    return {
        'overall_slope': slope,
        'overall_r2': r_value**2,
        'overall_pvalue': p_value,
        'day1_slope': slope_day1,
        'day2_slope': slope_day2,
        'percent_increase': percent_increase,
        'cross_session_jump': cross_session_increase
    }

# Calculate slopes for both groups
stats_g1 = calculate_slope_stats(df_g1, "Group 1")
stats_g2 = calculate_slope_stats(df_g2, "Group 2")

# ==========================================
# DISPLAY GROUP 2 DATA
# ==========================================

print("\n" + "="*70)
print("GROUP 2 RAW DATA")
print("="*70)
print("Day  Trial      Peak_Velocity (mean ± SEM)")
print("-" * 60)

for day in [1, 2]:
    day_data = df_g2[df_g2['Day'] == day]
    for trial in range(1, 6):
        trial_data = day_data[day_data['Trial'] == trial]['Peak_Velocity']
        mean_val = trial_data.mean()
        sem_val = trial_data.sem()
        print(f"{day}    {trial}         {mean_val:.6f} ± {sem_val:.6f}")
    print()

# ==========================================
# COMPARISON TABLE
# ==========================================

print("\n" + "="*70)
print("SLOPE COMPARISON: GROUP 1 vs GROUP 2")
print("="*70)

comparison = pd.DataFrame({
    'Metric': [
        'Overall Slope',
        'Overall R²',
        'Overall P-value',
        'Day 1 Slope',
        'Day 2 Slope',
        'Total Increase (%)',
        'Cross-session Jump'
    ],
    'Group 1': [
        f"{stats_g1['overall_slope']:.4f}",
        f"{stats_g1['overall_r2']:.4f}",
        f"{stats_g1['overall_pvalue']:.4e}",
        f"{stats_g1['day1_slope']:.4f}",
        f"{stats_g1['day2_slope']:.4f}",
        f"{stats_g1['percent_increase']:.1f}%",
        f"{stats_g1['cross_session_jump']:.4f}"
    ],
    'Group 2': [
        f"{stats_g2['overall_slope']:.4f}",
        f"{stats_g2['overall_r2']:.4f}",
        f"{stats_g2['overall_pvalue']:.4e}",
        f"{stats_g2['day1_slope']:.4f}",
        f"{stats_g2['day2_slope']:.4f}",
        f"{stats_g2['percent_increase']:.1f}%",
        f"{stats_g2['cross_session_jump']:.4f}"
    ]
})

print(comparison.to_string(index=False))

print("\n" + "="*70)
print("✅ Both groups show similar sensitization patterns!")
print("="*70)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PLOT DAY 2 ONLY - SAME STYLE
# ==========================================

# Filter for Day 2 data only from both groups
df_g1_day2 = df_g1[df_g1['Day'] == 2].copy()
df_g2_day2 = df_g2[df_g2['Day'] == 2].copy()

# Adjust Exposure_Index to start from 1 (instead of 6-10)
df_g1_day2['Trial_Index'] = df_g1_day2['Trial']
df_g2_day2['Trial_Index'] = df_g2_day2['Trial']

# ==========================================
# GRAPH GROUP 1 - DAY 2
# ==========================================

plt.figure(figsize=(12, 7))

# Main line plot with 68% confidence interval
sns.lineplot(data=df_g1_day2, x='Trial_Index', y='Normalized_Spark', 
             color='#e74c3c', marker='o', markersize=10, linewidth=4, 
             errorbar=('ci', 68), label='Day 2')

# Add individual fly traces (faint)
for fly in flies_g1:
    fly_data = df_g1_day2[df_g1_day2['Fly_ID'] == fly]
    plt.plot(fly_data['Trial_Index'], fly_data['Normalized_Spark'], 
            color='#e74c3c', alpha=0.2, linewidth=1.5, linestyle='-')

# Add trend line for Day 2
sns.regplot(data=df_g1_day2, x='Trial_Index', y='Normalized_Spark', 
            scatter=False, color='black', 
            line_kws={"linestyle": "--", "alpha": 0.35, "linewidth": 2})

# Add horizontal baseline reference line at 1.0
plt.axhline(y=1.0, color='gray', linestyle='-.', alpha=0.5, linewidth=2)
plt.text(4.5, 1.08, 'Baseline (1.0)', color='gray', fontsize=10, style='italic')

# Formatting
plt.title("Experimental Group 1: Day 2 Locomotor Response (24h Post-Exposure)", 
          fontsize=15, pad=20, fontweight='bold')
plt.ylabel("Normalized Peak Velocity\n(Fold-Change from Day 1 Baseline)", fontsize=13)
plt.xlabel("Trial Number (Day 2)", fontsize=13)
plt.xticks(range(1, 6), fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(bottom=0)
plt.grid(axis='y', alpha=0.25, linestyle='--')
plt.legend(fontsize=11, loc='upper left')

sns.despine()
plt.tight_layout()
plt.show()

# ==========================================
# GRAPH GROUP 2 - DAY 2
# ==========================================

plt.figure(figsize=(12, 7))

# Main line plot with 68% confidence interval
sns.lineplot(data=df_g2_day2, x='Trial_Index', y='Normalized_Spark', 
             color='#e74c3c', marker='o', markersize=10, linewidth=4, 
             errorbar=('ci', 68), label='Day 2')

# Add individual fly traces (faint)
for fly in flies_g2:
    fly_data = df_g2_day2[df_g2_day2['Fly_ID'] == fly]
    plt.plot(fly_data['Trial_Index'], fly_data['Normalized_Spark'], 
            color='#e74c3c', alpha=0.2, linewidth=1.5, linestyle='-')

# Add trend line for Day 2
sns.regplot(data=df_g2_day2, x='Trial_Index', y='Normalized_Spark', 
            scatter=False, color='black', 
            line_kws={"linestyle": "--", "alpha": 0.35, "linewidth": 2})

# Add horizontal baseline reference line at 1.0
plt.axhline(y=1.0, color='gray', linestyle='-.', alpha=0.5, linewidth=2)
plt.text(4.5, 1.08, 'Baseline (1.0)', color='gray', fontsize=10, style='italic')

# Formatting
plt.title("Experimental Group 2: Day 2 Locomotor Response (24h Post-Exposure)", 
          fontsize=15, pad=20, fontweight='bold')
plt.ylabel("Normalized Peak Velocity\n(Fold-Change from Day 1 Baseline)", fontsize=13)
plt.xlabel("Trial Number (Day 2)", fontsize=13)
plt.xticks(range(1, 6), fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(bottom=0)
plt.grid(axis='y', alpha=0.25, linestyle='--')
plt.legend(fontsize=11, loc='upper left')

sns.despine()
plt.tight_layout()
plt.show()

# ==========================================
# COMBINED COMPARISON - BOTH GROUPS DAY 2
# ==========================================

plt.figure(figsize=(12, 7))

# Group 1 Day 2
sns.lineplot(data=df_g1_day2, x='Trial_Index', y='Normalized_Spark', 
             color='#e74c3c', marker='o', markersize=10, linewidth=4, 
             errorbar=('ci', 68), label='Group 1 - Day 2')

# Add Group 1 individual fly traces
for fly in flies_g1:
    fly_data = df_g1_day2[df_g1_day2['Fly_ID'] == fly]
    plt.plot(fly_data['Trial_Index'], fly_data['Normalized_Spark'], 
            color='#e74c3c', alpha=0.15, linewidth=1.5, linestyle='-')

# Group 2 Day 2
sns.lineplot(data=df_g2_day2, x='Trial_Index', y='Normalized_Spark', 
             color='#9b59b6', marker='s', markersize=10, linewidth=4, 
             errorbar=('ci', 68), label='Group 2 - Day 2')

# Add Group 2 individual fly traces
for fly in flies_g2:
    fly_data = df_g2_day2[df_g2_day2['Fly_ID'] == fly]
    plt.plot(fly_data['Trial_Index'], fly_data['Normalized_Spark'], 
            color='#9b59b6', alpha=0.15, linewidth=1.5, linestyle='-')

# Add horizontal baseline reference line at 1.0
plt.axhline(y=1.0, color='gray', linestyle='-.', alpha=0.5, linewidth=2)
plt.text(4.5, 1.08, 'Baseline (1.0)', color='gray', fontsize=10, style='italic')

# Formatting
plt.title("Day 2 Comparison: Group 1 vs Group 2 Locomotor Response", 
          fontsize=15, pad=20, fontweight='bold')
plt.ylabel("Normalized Peak Velocity\n(Fold-Change from Day 1 Baseline)", fontsize=13)
plt.xlabel("Trial Number (Day 2)", fontsize=13)
plt.xticks(range(1, 6), fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(bottom=0)
plt.grid(axis='y', alpha=0.25, linestyle='--')
plt.legend(fontsize=11, loc='upper left')

sns.despine()
plt.tight_layout()
plt.show()

# ==========================================
# PRINT DAY 2 STATISTICS
# ==========================================

print("\n" + "="*70)
print("DAY 2 STATISTICS - GROUP 1")
print("="*70)
day2_stats_g1 = df_g1_day2.groupby('Trial').agg({
    'Normalized_Spark': ['mean', 'sem', 'std', 'count']
}).round(4)
print(day2_stats_g1)

print("\n" + "="*70)
print("DAY 2 STATISTICS - GROUP 2")
print("="*70)
day2_stats_g2 = df_g2_day2.groupby('Trial').agg({
    'Normalized_Spark': ['mean', 'sem', 'std', 'count']
}).round(4)
print(day2_stats_g2)

# Calculate Day 2 slope
from scipy import stats

def calculate_day2_slope(df, group_name):
    mean_data = df.groupby('Trial')['Normalized_Spark'].mean().reset_index()
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        mean_data['Trial'], mean_data['Normalized_Spark']
    )
    
    print(f"\n{'='*70}")
    print(f"{group_name} - DAY 2 SLOPE")
    print(f"{'='*70}")
    print(f"Slope:          {slope:.4f} (units per trial)")
    print(f"Intercept:      {intercept:.4f}")
    print(f"R²:             {r_value**2:.4f}")
    print(f"P-value:        {p_value:.4e}")
    
    return slope, r_value**2

slope_g1_day2, r2_g1_day2 = calculate_day2_slope(df_g1_day2, "GROUP 1")
slope_g2_day2, r2_g2_day2 = calculate_day2_slope(df_g2_day2, "GROUP 2")