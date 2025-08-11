import os
import glob
import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define paths based on platform
system_platform = platform.system()
if system_platform == 'Darwin':
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 2r")
    output_path = Path("/Volumes/T7 Shield/3D Analysis")
elif system_platform == 'Linux':
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 2r")
    output_path = Path("/Users/brolinadu-poku/City College Dropbox/Brolin Adu Poku/3D Analysis")
else:
    raise OSError(f"Unsupported OS: {system_platform}")

# Create output directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

# Simulation parameters with multiple values
phi = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]  # Updated with new phi values
ar = [1.4]           # Example multiple aspect ratios
vr = ['0.5']       # Example multiple velocity ratios
numRun = [1]           # Example multiple run numbers
n_particles = 1000  # From metadata

# File patterns
particleFile = 'par_*.dat'
dataFile = 'data_*.dat'
interactionFile = 'int_*.dat'

def read_data_file(file_path):
    """Read data_*.dat file and extract specified columns."""
    data = np.loadtxt(file_path)
    return {
        'shear_rate': data[:, 2],  # Column 3
        'viscosity': data[:, 3],   # Column 4
        'time': data[:, 0]         # Column 1
    }

def read_particles_file(file_path):
    """Read par_*.dat file and extract particle positions."""
    with open(file_path, 'r') as f:
        lines = f.readlines()[22:]  # Skip initial 22 comment lines
    parList = []
    frame = []
    hashCounter = 0
    for line in lines:
        if line.startswith('#'):
            hashCounter += 1
            if hashCounter == 7 and frame:
                parList.append(np.array(frame))
                frame = []
                hashCounter = 0
        else:
            frame.append([float(x) for x in line.split()])
    if frame:
        parList.append(np.array(frame))
    return [frame[:, [2, 3, 4]] for frame in parList]  # Positions (x, y, z)

def read_interaction_file(file_path):
    """Read int_*.dat file and extract contact states and particle labels."""
    with open(file_path, 'r') as f:
        lines = f.readlines()[27:]  # Skip initial 27 comment lines
    interactions = []
    temp = []
    hashCounter = 0
    for line in lines:
        if line.startswith('#'):
            hashCounter += 1
            if hashCounter == 7 and temp:
                interactions.append(np.array(temp))
                temp = []
                hashCounter = 0
        else:
            temp.append([float(x) for x in line.split()])
    if temp:
        interactions.append(np.array(temp))
    return interactions

# Dictionary to store all data for combined plotting
all_phi_data = {}

# Generate colors for different phi values
colors = plt.cm.viridis(np.linspace(0, 1, len(phi)))

# Process each simulation directory
for i, phii in enumerate(phi):
    phii_str = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    all_phi_data[phii] = {'time': [], 'contacts': [], 'color': colors[i]}
    
    for j, arj in enumerate(ar):
        for k, vrk in enumerate(vr):
            for m, run in enumerate(numRun):
                dataname = f"{topDir}/phi_{phii_str}/ar_{arj}/Vr_{vrk}/run_{run}"
                if not os.path.exists(dataname):
                    print(f"Directory {dataname} not found. Skipping...")
                    continue

                par_files = glob.glob(f'{dataname}/{particleFile}')
                data_files = glob.glob(f'{dataname}/{dataFile}')
                interaction_files = glob.glob(f'{dataname}/{interactionFile}')

                if not par_files or not data_files or not interaction_files:
                    print(f"Missing files in {dataname}. Skipping...")
                    continue

                # Process all files for this phi value
                phi_time_data = []
                phi_contact_data = []
                
                for par_file in par_files:
                    # Find corresponding data and interaction files
                    base_name = os.path.basename(par_file).replace('par_', '').replace('.dat', '')
                    data_file = next((f for f in data_files if os.path.basename(f).replace('data_', '').replace('.dat', '') == base_name), None)
                    interaction_file = next((f for f in interaction_files if os.path.basename(f).replace('int_', '').replace('.dat', '') == base_name), None)

                    if not data_file or not interaction_file:
                        print(f"Missing matching data or interaction file for {par_file}. Skipping...")
                        continue

                    # Read data
                    data_dict = read_data_file(data_file)
                    positions = read_particles_file(par_file)
                    interactions = read_interaction_file(interaction_file)

                    # Store adjacency matrices and total frictional contacts
                    total_frictional_contacts = []
                    timesteps = data_dict['time']

                    # Process each timestep
                    min_length = min(len(positions), len(interactions), len(timesteps))
                    for t in range(min_length):
                        # Identify frictional contacts (contact_state 2 or 3)
                        frictional_count = 0
                        for row in interactions[t]:
                            p1, p2, contact_state = int(row[0]), int(row[1]), int(row[10])
                            if contact_state in [2, 3]:  # Non-sliding or sliding frictional contact
                                frictional_count += 1

                        total_frictional_contacts.append(frictional_count)

                    phi_time_data.extend(timesteps[:min_length])
                    phi_contact_data.extend(total_frictional_contacts)

                # Store data for this phi value
                all_phi_data[phii]['time'] = phi_time_data
                all_phi_data[phii]['contacts'] = phi_contact_data

                # Generate individual plot for this phi value
                if phi_time_data and phi_contact_data:
                    plt.figure(figsize=(10, 6))
                    plt.plot(phi_time_data, phi_contact_data, 'o-', 
                            color=all_phi_data[phii]['color'], alpha=0.7, markersize=3)
                    plt.xlabel('Time')
                    plt.ylabel('Total Number of Frictional Contacts')
                    plt.title(f'Total Frictional Contacts vs Time (φ = {phii})')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Save individual plot
                    plt.savefig(output_path / f'frictional_contacts_phi_{phii_str}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.show()

                print(f"Processed phi={phii} in {dataname}. Individual plot generated.")

# Generate combined plot for all phi values
plt.figure(figsize=(12, 8))
for phii in phi:
    if all_phi_data[phii]['time'] and all_phi_data[phii]['contacts']:
        plt.plot(all_phi_data[phii]['time'], all_phi_data[phii]['contacts'], 
                'o-', color=all_phi_data[phii]['color'], alpha=0.7, 
                label=f'φ = {phii}', markersize=2)

plt.xlabel('Time', fontsize=12)
plt.ylabel('Total Number of Frictional Contacts', fontsize=12)
plt.title('Total Frictional Contacts vs Time for All φ Values', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save combined plot
plt.savefig(output_path / 'frictional_contacts_all_phi.png', 
           dpi=300, bbox_inches='tight')
plt.show()

# Generate summary statistics
print("\nSummary Statistics:")
print("="*50)
for phii in phi:
    if all_phi_data[phii]['contacts']:
        contacts = np.array(all_phi_data[phii]['contacts'])
        print(f"φ = {phii:5.2f}: Mean = {np.mean(contacts):6.1f}, "
              f"Std = {np.std(contacts):6.1f}, "
              f"Max = {np.max(contacts):6.0f}, "
              f"Min = {np.min(contacts):6.0f}")

print(f"\nPlots saved to: {output_path}")
print("Individual plots: frictional_contacts_phi_X.XX.png")
print("Combined plot: frictional_contacts_all_phi.png")