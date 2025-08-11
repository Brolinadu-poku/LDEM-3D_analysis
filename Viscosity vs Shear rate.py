import pandas as pd
import matplotlib.pyplot as plt

# File path
file_path = '/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 10r/phi_0.52/ar_1.4/Vr_0.5/run_1/data_random_seed_params_stress10r_shear.dat'
# Step 1: Find where the actual data starts
with open(file_path, 'r') as f:
    raw_lines = f.readlines()

data_start_index = next(i for i, line in enumerate(raw_lines) if not line.startswith('#'))

# Step 2: Load data
df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=data_start_index)

# Step 3: Extract columns for shear rate and viscosity
shear_rate = df[2]    # Column 3
viscosity = df[3]     # Column 4

# Step 4: Plot all data (no slicing)
plt.figure(figsize=(10, 6))
plt.plot(shear_rate, viscosity, label='All Data', color='blue')
plt.xlabel('Shear Rate')
plt.ylabel('Viscosity')
plt.title('Viscosity vs. Shear Rate (All Data)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
