import pandas as pd
import numpy as np

# Set seed for scientific reproducibility
np.random.seed(42)

def generate_long_data():
    all_rows = []
    
    # Define fly cohorts
    groups = {
        'Experimental': ['Fly_A', 'Fly_B', 'Fly_C', 'Fly_D', 'Fly_E', 'Fly_F'],
        'Control': ['Fly_G', 'Fly_H', 'Fly_I', 'Fly_J', 'Fly_K', 'Fly_L']
    }

    for group_name, flies in groups.items():
        for fly in flies:
            # Each fly has a unique baseline (Day 1 Pre-Stimulus average)
            fly_day1_baseline = np.random.uniform(7.5, 9.0)
            
            # 1. SET PATTERNS BASED ON YOUR INPUT
            if group_name == 'Experimental':
                # Day 1: Variable/Habituating [9.5, 7.3, 9.1, 11.4, 8.4]
                d1_base_pattern = np.array([9.5, 7.3, 9.1, 11.4, 8.4])
                # Day 2: Sensitizing [11.2, 13.5, 15.1, 16.8, 18.5]
                d2_base_pattern = np.array([11.2, 13.5, 15.1, 16.8, 18.5])
            else:
                # Control Day 1: Flat [8.2, 7.8, 8.5, 8.1, 8.3]
                d1_base_pattern = np.array([8.2, 7.8, 8.5, 8.1, 8.3])
                # Control Day 2: Flat [8.4, 7.9, 8.6, 8.2, 8.7]
                d2_base_pattern = np.array([8.4, 7.9, 8.6, 8.2, 8.7])

            # 2. GENERATE TRIALS
            for day in [1, 2]:
                pattern = d1_base_pattern if day == 1 else d2_base_pattern
                # Add random biological noise to the pattern
                noise = np.random.normal(0, 0.7, 5)
                trial_velocities = pattern + noise

                for trial_idx, peak_v in enumerate(trial_velocities):
                    trial_num = trial_idx + 1
                    
                    # 3. CALCULATE NORMALIZED SPARK (DV)
                    # Fold-change = current peak / Day 1 Baseline
                    normalized_spark = peak_v / fly_day1_baseline
                    
                    all_rows.append({
                        'Fly_ID': fly,
                        'Group': group_name,
                        'Day': day,
                        'Trial': trial_num,
                        'Peak_Velocity': round(peak_v, 3),
                        'Normalized_Spark': round(normalized_spark, 3)
                    })

    return pd.DataFrame(all_rows)

# Generate and Save
df_long = generate_long_data()
df_long.to_csv("ANOVA_DATA_LONG.csv", index=False)

# Show a preview of the "Long Format"
print("--- PREVIEW OF LONG FORMAT DATA ---")
print(df_long.head(12)) # Shows all trials for Fly A, Day 1 and 2