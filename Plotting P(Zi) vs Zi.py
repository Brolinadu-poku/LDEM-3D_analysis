import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys

def read_coordination_analysis_file(filepath):
    """
    Read coordination analysis result files.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the coordination analysis result file
        
    Returns:
    --------
    dict : Contains 'data' (numpy array), 'metadata' (dict), 'columns' (dict)
    """
    filepath = Path(filepath)
    
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'Volume Fraction' in line:
                metadata['phi'] = float(line.split(':')[1].strip())
            elif 'Aspect Ratio' in line:
                metadata['aspect_ratio'] = float(line.split(':')[1].strip())
            elif 'Velocity Ratio' in line:
                metadata['velocity_ratio'] = line.split(':')[1].strip()
            elif 'Run Number' in line:
                metadata['run_number'] = int(line.split(':')[1].strip())
            elif 'Number of Particles' in line:
                metadata['n_particles'] = int(line.split(':')[1].strip())
            elif 'Box Dimensions' in line:
                dims = line.split('-')[1].strip()
                for dim in dims.split(', '):
                    key, value = dim.split(': ')
                    try:
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = value
    
    try:
        data = np.loadtxt(filepath)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return {'data': None, 'metadata': metadata, 'columns': {}}
    
    n_particles = metadata.get('n_particles', 1000)
    columns = {
        'timestep_index': 0,
        'time': 1,
        'shear_rate': 2,
        'viscosity': 3,
        'frictional_contact_number': 4
    }
    
    for i in range(n_particles):
        columns[f'coordination_number_particle_{i}'] = 5 + i
    
    return {
        'data': data,
        'metadata': metadata,
        'columns': columns
    }

output_path = Path("/Volumes/T7 Shield/3D Analysis")

if not output_path.exists():
    print(f"Error: Output path {output_path} does not exist.")
    sys.exit(1)

phi = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]

def calculate_p_zi(file_path):
    """Calculate P(Zi) for a coordination analysis file."""
    result = read_coordination_analysis_file(file_path)
    data = result['data']
    metadata = result['metadata']
    
    if data is None or data.size == 0:
        print(f"No valid data in {file_path}. Skipping...")
        return None, None, None
    
    n_particles = metadata['n_particles']
    n_timesteps = data.shape[0]
    
    zi_columns = data[:, 5:5+n_particles]
    zi_flat = zi_columns.flatten()
    
    max_zi = int(np.max(zi_flat)) if len(zi_flat) > 0 else 0
    bins = np.arange(0, max_zi + 2)
    hist, bin_edges = np.histogram(zi_flat, bins=bins, density=False)
    
    total_samples = n_particles * n_timesteps
    p_zi = hist / total_samples if total_samples > 0 else hist
    
    return p_zi, bin_edges[:-1], metadata

p_zi_data = {}
max_zi_global = 0

print("Scanning for coordination analysis files...")
for phii in phi:
    phii_str = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    file_pattern = f"{output_path}/coordination_analysis_phi_{phii_str}_*.txt"
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found for phi = {phii_str}. Skipping...")
        continue
    
    print(f"Found {len(files)} files for phi = {phii_str}: {files}")
    
    combined_hist = None
    total_samples = 0
    zi_values = None
    for file in files:
        p_zi, zi_vals, metadata = calculate_p_zi(file)
        if p_zi is None:
            continue
        n_particles = metadata['n_particles']
        n_timesteps = read_coordination_analysis_file(file)['data'].shape[0] if read_coordination_analysis_file(file)['data'] is not None else 0
        samples = n_particles * n_timesteps
        total_samples += samples
        
        if combined_hist is None:
            combined_hist = p_zi * samples
            zi_values = zi_vals
        else:
            if len(p_zi) > len(combined_hist):
                combined_hist = np.pad(combined_hist, (0, len(p_zi) - len(combined_hist)), mode='constant')
            elif len(p_zi) < len(combined_hist):
                p_zi = np.pad(p_zi, (0, len(combined_hist) - len(p_zi)), mode='constant')
            combined_hist += p_zi * samples
        
        max_zi_global = max(max_zi_global, len(p_zi))
    
    if total_samples > 0:
        p_zi = combined_hist / total_samples
        p_zi_data[phii] = (p_zi, zi_values)
    else:
        print(f"No valid data for phi = {phii_str}. Skipping...")

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(phi)))

for idx, phii in enumerate(phi):
    if phii not in p_zi_data:
        continue
    p_zi, zi_values = p_zi_data[phii]
    plt.plot(zi_values, p_zi, marker='o', label=f'φ={phii}', color=colors[idx], alpha=0.8)

plt.xlabel('Coordination Number (Zi)')
plt.ylabel('Probability P(Zi)')
plt.title('Probability Distribution of Coordination Numbers for Different Volume Fractions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, max_zi_global + 1, 1))
plt.tight_layout()
plt.savefig(output_path / 'p_zi_distribution.png')
plt.show()

print(f"Plot saved to: {output_path / 'p_zi_distribution.png'}")
