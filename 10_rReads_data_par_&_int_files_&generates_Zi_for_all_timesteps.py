import os  # Import os module for operating system functionalities
import glob  # Import glob module for file pattern matching
import platform  # Import platform module to determine operating system
from pathlib import Path  # Import Path class for handling file paths
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
from datetime import datetime  # Import datetime for timestamp generation

# Define paths based on platform
system_platform = platform.system()  # Get the current operating system
if system_platform == 'Darwin':  # Check if the OS is macOS
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 10r")  # Set top directory path for macOS
    output_path = Path("/Volumes/T7 Shield/3D Analysis 10r")  # Set output directory path for macOS
elif system_platform == 'Linux':  # Check if the OS is Linux
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 10r")  # Set top directory path for Linux
    output_path = Path("/Users/brolinadu-poku/City College Dropbox/Brolin Adu Poku/3D Analysis")  # Set output directory path for Linux
else:  # Handle unsupported operating systems
    raise OSError(f"Unsupported OS: {system_platform}")  # Raise an error for unsupported OS

# Create output directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)  # Create output directory, including parents, if it doesn't exist

# Simulation parameters
phi = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]  # Define list of volume fractions
ar = [1.4]  # Define list of aspect ratios
vr = ['0.5']  # Define list of volume ratios (smaller to larger particles)
numRun = [1]  # Define list of run numbers
n_particles = 1000  # Set number of particles in simulation

# File patterns
particleFile = 'par_*.dat'  # Define pattern for particle data files
dataFile = 'data_*.dat'  # Define pattern for simulation data files
interactionFile = 'int_*.dat'  # Define pattern for interaction data files

def read_data_file(file_path):  # Define function to read data_*.dat files
    """Read data_*.dat file and extract all columns with proper headers."""
    data = np.loadtxt(file_path)  # Load data from file into numpy array
    
    # Column mapping based on provided headers
    columns = {
        'time': data[:, 0],  # Extract time column (index 0)
        'cumulated_shear_strain': data[:, 1],  # Extract cumulated shear strain (index 1)
        'shear_rate': data[:, 2],  # Extract shear rate (index 2)
        'viscosity': data[:, 3],  # Extract viscosity (index 3)
        'viscosity_contact': data[:, 4],  # Extract contact viscosity (index 4)
        'viscosity_dashpot': data[:, 5],  # Extract dashpot viscosity (index 5)
        'viscosity_hydro': data[:, 6],  # Extract hydrodynamic viscosity (index 6)
        'viscosity_repulsion': data[:, 7],  # Extract repulsion viscosity (index 7)
        'particle_pressure': data[:, 8],  # Extract particle pressure (index 8)
        'particle_pressure_contact': data[:, 9],  # Extract contact particle pressure (index 9)
        'N1_viscosity': data[:, 10],  # Extract N1 viscosity (index 10)
        'N2_viscosity': data[:, 11],  # Extract N2 viscosity (index 11)
        'energy': data[:, 12],  # Extract energy (index 12)
        'min_gap': data[:, 13],  # Extract minimum gap (index 13)
        'max_tangential_displacement': data[:, 14],  # Extract max tangential displacement (index 14)
        'contact_number': data[:, 15],  # Extract contact number (index 15)
        'frictional_contact_number': data[:, 16],  # Extract frictional contact number (index 16)
        'avg_sliding_friction_mobilization': data[:, 17],  # Extract average sliding friction mobilization (index 17)
        'number_of_interaction': data[:, 18],  # Extract number of interactions (index 18)
        'max_velocity': data[:, 19],  # Extract maximum velocity (index 19)
        'max_angular_velocity': data[:, 20],  # Extract maximum angular velocity (index 20)
        'dt': data[:, 21],  # Extract time step (index 21)
        'kn': data[:, 22],  # Extract normal stiffness (index 22)
        'kt': data[:, 23],  # Extract tangential stiffness (index 23)
        'kr': data[:, 24],  # Extract rotational stiffness (index 24)
        'shear_strain_26': data[:, 25],  # Extract shear strain component (index 25)
        'shear_strain_27': data[:, 26],  # Extract shear strain component (index 26)
        'shear_strain_28': data[:, 27],  # Extract shear strain component (index 27)
        'shear_stress': data[:, 28],  # Extract shear stress (index 28)
        'theta_shear': data[:, 29]  # Extract shear angle (index 29)
    }
    return columns  # Return dictionary of column data

def read_particles_file(file_path):  # Define function to read par_*.dat files
    """Read par_*.dat file and extract all particle data with proper headers."""
    with open(file_path, 'r') as f:  # Open file in read mode
        lines = f.readlines()  # Read all lines from file
    
    # Extract metadata from header
    metadata = {}  # Initialize dictionary for metadata
    for line in lines[:22]:  # Process first 22 lines (header)
        if line.startswith('# np '):  # Check for number of particles
            metadata['np'] = int(line.split()[-1])  # Store number of particles
        elif line.startswith('# VF '):  # Check for volume fraction
            metadata['VF'] = float(line.split()[-1])  # Store volume fraction
        elif line.startswith('# Lx '):  # Check for box dimension Lx
            metadata['Lx'] = float(line.split()[-1])  # Store Lx dimension
        elif line.startswith('# Ly '):  # Check for box dimension Ly
            metadata['Ly'] = float(line.split()[-1])  # Store Ly dimension
        elif line.startswith('# Lz '):  # Check for box dimension Lz
            metadata['Lz'] = float(line.split()[-1])  # Store Lz dimension
    
    # Process particle data
    lines = lines[22:]  # Skip header lines
    parList = []  # Initialize list to store particle data frames
    frame = []  # Initialize temporary list for current frame
    hashCounter = 0  # Initialize counter for hash lines
    
    for line in lines:  # Process each line
        if line.startswith('#'):  # Check if line is a separator
            hashCounter += 1  # Increment hash counter
            if hashCounter == 7 and frame:  # Check if frame is complete (7 hashes)
                parList.append(np.array(frame))  # Store frame as numpy array
                frame = []  # Reset frame
                hashCounter = 0  # Reset hash counter
        else:  # Process data line
            frame.append([float(x) for x in line.split()])  # Convert line to list of floats
    
    if frame:  # Check if any remaining frame data
        parList.append(np.array(frame))  # Store final frame
    
    # Return structured data with column names
    structured_data = []  # Initialize list for structured data
    for frame in parList:  # Process each frame
        frame_dict = {
            'particle_index': frame[:, 0].astype(int),  # Extract particle index (column 0)
            'radius': frame[:, 1],  # Extract particle radius (column 1)
            'pos_x': frame[:, 2],  # Extract x-position (column 2)
            'pos_y': frame[:, 3],  # Extract y-position (column 3)
            'pos_z': frame[:, 4],  # Extract z-position (column 4)
            'vel_x': frame[:, 5],  # Extract x-velocity (column 5)
            'vel_y': frame[:, 6],  # Extract y-velocity (column 6)
            'vel_z': frame[:, 7],  # Extract z-velocity (column 7)
            'ang_vel_x': frame[:, 8],  # Extract x-angular velocity (column 8)
            'ang_vel_y': frame[:, 9],  # Extract y-angular velocity (column 9)
            'ang_vel_z': frame[:, 10]  # Extract z-angular velocity (column 10)
        }
        structured_data.append(frame_dict)  # Append frame dictionary to list
    
    return structured_data, metadata  # Return structured data and metadata

def read_interaction_file(file_path):  # Define function to read int_*.dat files
    """Read int_*.dat file and extract all interaction data with proper headers."""
    with open(file_path, 'r') as f:  # Open file in read mode
        lines = f.readlines()[27:]  # Skip first 27 lines (header)
    
    interactions = []  # Initialize list for interaction frames
    temp = []  # Initialize temporary list for current frame
    hashCounter = 0  # Initialize counter for hash lines
    
    for line in lines:  # Process each line
        if line.startswith('#'):  # Check if line is a separator
            hashCounter += 1  # Increment hash counter
            if hashCounter == 7 and temp:  # Check if frame is complete
                interactions.append(np.array(temp))  # Store frame as numpy array
                temp = []  # Reset frame
                hashCounter = 0  # Reset hash counter
        else:  # Process data line
            temp.append([float(x) for x in line.split()])  # Convert line to list of floats
    
    if temp:  # Check if any remaining frame data
        interactions.append(np.array(temp))  # Store final frame
    
    # Return structured data with column names
    structured_interactions = []  # Initialize list for structured interactions
    for frame in interactions:  # Process each frame
        if len(frame) > 0:  # Check if frame contains data
            frame_dict = {
                'particle_1_label': frame[:, 0].astype(int),  # Extract particle 1 label (column 0)
                'particle_2_label': frame[:, 1].astype(int),  # Extract particle 2 label (column 1)
                'normal_vector_x': frame[:, 2],  # Extract x-component of normal vector (column 2)
                'normal_vector_y': frame[:, 3],  # Extract y-component of normal vector (column 3)
                'normal_vector_z': frame[:, 4],  # Extract z-component of normal vector (column 4)
                'dimensionless_gap': frame[:, 5],  # Extract dimensionless gap (column 5)
                'normal_lubrication_force': frame[:, 6],  # Extract normal lubrication force (column 6)
                'tangential_lubrication_force_x': frame[:, 7],  # Extract x-tangential lubrication force (column 7)
                'tangential_lubrication_force_y': frame[:, 8],  # Extract y-tangential lubrication force (column 8)
                'tangential_lubrication_force_z': frame[:, 9],  # Extract z-tangential lubrication force (column 9)
                'contact_state': frame[:, 10].astype(int),  # Extract contact state (column 10)
                'normal_contact_force_norm': frame[:, 11],  # Extract normal contact force norm (column 11)
                'tangential_contact_force_x': frame[:, 12],  # Extract x-tangential contact force (column 12)
                'tangential_contact_force_y': frame[:, 13],  # Extract y-tangential contact force (column 13)
                'tangential_contact_force_z': frame[:, 14],  # Extract z-tangential contact force (column 14)
                'sliding_friction_mobilization': frame[:, 15],  # Extract sliding friction mobilization (column 15)
                'normal_repulsive_force_norm': frame[:, 16]  # Extract normal repulsive force norm (column 16)
            }
        else:  # Handle empty frame
            frame_dict = {}  # Create empty dictionary for empty frame
        structured_interactions.append(frame_dict)  # Append frame dictionary to list
    
    return structured_interactions  # Return list of structured interactions

def calculate_coordination_numbers(interactions, n_particles):  # Define function to calculate coordination numbers
    """Calculate Zi = sum(Aij) for each particle i at each timestep."""
    coordination_data = []  # Initialize list for coordination numbers
    
    for t, interaction_frame in enumerate(interactions):  # Iterate through interaction frames
        # Initialize adjacency matrix
        adj_matrix = np.zeros((n_particles, n_particles), dtype=int)  # Create zero matrix for particle interactions
        
        # Build adjacency matrix for frictional contacts
        if interaction_frame:  # Check if frame has data
            for i in range(len(interaction_frame['particle_1_label'])):  # Iterate through interactions
                p1 = interaction_frame['particle_1_label'][i]  # Get particle 1 label
                p2 = interaction_frame['particle_2_label'][i]  # Get particle 2 label
                contact_state = interaction_frame['contact_state'][i]  # Get contact state
                
                # Only frictional contacts (state 2 or 3)
                if contact_state in [2, 3]:  # Check for frictional contact states
                    adj_matrix[p1, p2] = 1  # Mark interaction in adjacency matrix
                    adj_matrix[p2, p1] = 1  # Mark symmetric interaction
        
        # Calculate coordination number Zi for each particle
        coordination_numbers = np.sum(adj_matrix, axis=1)  # Sum each row to get coordination numbers
        coordination_data.append(coordination_numbers)  # Store coordination numbers for this timestep
    
    return np.array(coordination_data)  # Return array of coordination numbers (n_timesteps, n_particles)

# Main processing loop
all_results = {}  # Initialize dictionary to store all results

for i, phii in enumerate(phi):  # Iterate through volume fractions
    phii_str = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)  # Format volume fraction string
    print(f"Processing phi = {phii} ({phii_str})...")  # Print processing status
    
    all_results[phii] = {}  # Initialize results dictionary for this volume fraction
    
    for j, arj in enumerate(ar):  # Iterate through aspect ratios
        for k, vrk in enumerate(vr):  # Iterate through volume ratios
            for m, run in enumerate(numRun):  # Iterate through run numbers
                dataname = f"{topDir}/phi_{phii_str}/ar_{arj}/Vr_{vrk}/run_{run}"  # Construct directory path
                
                if not os.path.exists(dataname):  # Check if directory exists
                    print(f"Directory {dataname} not found. Skipping...")  # Print skip message
                    continue  # Skip to next iteration

                par_files = glob.glob(f'{dataname}/{particleFile}')  # Get list of particle files
                data_files = glob.glob(f'{dataname}/{dataFile}')  # Get list of data files
                interaction_files = glob.glob(f'{dataname}/{interactionFile}')  # Get list of interaction files

                if not par_files or not data_files or not interaction_files:  # Check if any files are missing
                    print(f"Missing files in {dataname}. Skipping...")  # Print skip message
                    continue  # Skip to next iteration

                # Process all files for this configuration
                for par_file in par_files:  # Iterate through particle files
                    base_name = os.path.basename(par_file).replace('par_', '').replace('.dat', '')  # Extract base filename
                    data_file = next((f for f in data_files if os.path.basename(f).replace('data_', '').replace('.dat', '') == base_name), None)  # Find matching data file
                    interaction_file = next((f for f in interaction_files if os.path.basename(f).replace('int_', '').replace('.dat', '') == base_name), None)  # Find matching interaction file

                    if not data_file or not interaction_file:  # Check if matching files exist
                        print(f"Missing matching files for {par_file}. Skipping...")  # Print skip message
                        continue  # Skip to next iteration

                    # Read all data
                    data_dict = read_data_file(data_file)  # Read data file
                    particles_data, metadata = read_particles_file(par_file)  # Read particle file
                    interactions_data = read_interaction_file(interaction_file)  # Read interaction file

                    # Calculate coordination numbers
                    coordination_numbers = calculate_coordination_numbers(interactions_data, n_particles)  # Compute coordination numbers
                    
                    # Prepare comprehensive output
                    min_length = min(len(coordination_numbers), len(data_dict['time']))  # Get minimum length for consistent output
                    
                    # Create output filename
                    output_filename = f"coordination_analysis_phi_{phii_str}_ar_{arj}_vr_{vrk}_run_{run}_{base_name}.txt"  # Construct output filename
                    output_file = output_path / output_filename  # Create full output file path
                    
                    # Write comprehensive data file
                    with open(output_file, 'w') as f:  # Open output file in write mode
                        f.write(f"# Coordination Number Analysis Results\n")  # Write title
                        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # Write generation timestamp
                        f.write(f"# Volume Fraction (VF/phi): {phii}\n")  # Write volume fraction
                        f.write(f"# Aspect Ratio: {arj}\n")  # Write aspect ratio
                        f.write(f"# Velocity Ratio: {vrk}\n")  # Write volume ratio
                        f.write(f"# Run Number: {run}\n")  # Write run number
                        f.write(f"# Base File: {base_name}\n")  # Write base filename
                        f.write(f"# Number of Particles (np): {metadata.get('np', n_particles)}\n")  # Write number of particles
                        f.write(f"# Box Dimensions - Lx: {metadata.get('Lx', 'N/A')}, Ly: {metadata.get('Ly', 'N/A')}, Lz: {metadata.get('Lz', 'N/A')}\n")  # Write box dimensions
                        f.write(f"# Number of Timesteps: {min_length}\n")  # Write number of timesteps
                        f.write(f"#\n")  # Write separator
                        f.write(f"# Column Definitions:\n")  # Write column definitions header
                        f.write(f"# 1: timestep_index\n")  # Define timestep index column
                        f.write(f"# 2: time\n")  # Define time column
                        f.write(f"# 3: shear_rate\n")  # Define shear rate column
                        f.write(f"# 4: viscosity\n")  # Define viscosity column
                        f.write(f"# 5: frictional_contact_number\n")  # Define frictional contact number column
                        f.write(f"# 6-{5+n_particles}: coordination_number_Zi_for_particles_0_to_{n_particles-1}\n")  # Define coordination number columns
                        f.write(f"#\n")  # Write separator
                        
                        # Write data
                        for t in range(min_length):  # Iterate through timesteps
                            line = f"{t}\t{data_dict['time'][t]:.6f}\t{data_dict['shear_rate'][t]:.6f}\t{data_dict['viscosity'][t]:.6f}\t{data_dict['frictional_contact_number'][t]:.0f}"  # Write base data
                            for particle_i in range(n_particles):  # Iterate through particles
                                line += f"\t{coordination_numbers[t, particle_i]}"  # Append coordination number
                            line += "\n"  # Add newline
                            f.write(line)  # Write line to file
                    
                    print(f"  Saved: {output_filename}")  # Print save confirmation
                    
                    # Store summary in memory for potential further analysis
                    all_results[phii][base_name] = {  # Store results in dictionary
                        'coordination_numbers': coordination_numbers,  # Store coordination numbers
                        'time': data_dict['time'][:min_length],  # Store time data
                        'metadata': metadata,  # Store metadata
                        'output_file': output_file  # Store output file path
                    }

print(f"\nAll coordination number analyses completed!")  # Print completion message
print(f"Results saved to: {output_path}")  # Print output directory
print(f"\nTo read any results file, use:")  # Print instructions for reading results
print(f"data = np.loadtxt('filename.txt')")  # Print example code for loading data
print(f"# Then access columns as: data[:, column_index]")  # Print instructions for accessing columns

# Create a utility file reader function
utility_code = '''
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
from pathlib import Path  # Import Path for file path handling

def read_coordination_analysis_file(filepath):  # Define function to read analysis files
    """
    Utility function to read coordination analysis result files.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the coordination analysis result file
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'data': numpy array with all numerical data
        - 'metadata': dict with simulation parameters
        - 'columns': dict with column mappings
    """
    filepath = Path(filepath)  # Convert filepath to Path object
    
    # Read metadata from header
    metadata = {}  # Initialize dictionary for metadata
    with open(filepath, 'r') as f:  # Open file in read mode
        for line in f:  # Read each line
            if not line.startswith('#'):  # Stop at first non-comment line
                break
            if 'Volume Fraction' in line:  # Check for volume fraction
                metadata['phi'] = float(line.split(':')[1].strip())  # Store volume fraction
            elif 'Aspect Ratio' in line:  # Check for aspect ratio
                metadata['aspect_ratio'] = float(line.split(':')[1].strip())  # Store aspect ratio
            elif 'Velocity Ratio' in line:  # Check for volume ratio
                metadata['velocity_ratio'] = line.split(':')[1].strip()  # Store volume ratio
            elif 'Run Number' in line:  # Check for run number
                metadata['run_number'] = int(line.split(':')[1].strip())  # Store run number
            elif 'Number of Particles' in line:  # Check for number of particles
                metadata['n_particles'] = int(line.split(':')[1].strip())  # Store number of particles
            elif 'Box Dimensions' in line:  # Check for box dimensions
                dims = line.split('-')[1].strip()  # Extract dimensions string
                # Parse Lx: X, Ly: Y, Lz: Z
                for dim in dims.split(', '):  # Split dimensions
                    key, value = dim.split(': ')  # Split key-value pair
                    try:
                        metadata[key] = float(value)  # Store dimension as float
                    except ValueError:
                        metadata[key] = value  # Store as string if not float
    
    # Read numerical data
    data = np.loadtxt(filepath)  # Load numerical data into numpy array
    
    # Column mappings
    n_particles = metadata.get('n_particles', 1000)  # Get number of particles (default 1000)
    columns = {  # Define column mappings
        'timestep_index': 0,  # Map timestep index to column 0
        'time': 1,  # Map time to column 1
        'shear_rate': 2,  # Map shear rate to column 2
        'viscosity': 3,  # Map viscosity to column 3
        'frictional_contact_number': 4  # Map frictional contact number to column 4
    }
    
    # Add coordination number columns
    for i in range(n_particles):  # Iterate through particles
        columns[f'coordination_number_particle_{i}'] = 5 + i  # Map coordination number columns
    
    return {  # Return dictionary with results
        'data': data,  # Numerical data
        'metadata': metadata,  # Simulation metadata
        'columns': columns  # Column mappings
    }

def extract_column(analysis_result, column_name):  # Define function to extract specific column
    """Extract a specific column from analysis result."""
    col_idx = analysis_result['columns'][column_name]  # Get column index
    return analysis_result['data'][:, col_idx]  # Return column data

def get_all_coordination_numbers(analysis_result):  # Define function to get all coordination numbers
    """Get coordination numbers for all particles as a 2D array."""
    n_particles = analysis_result['metadata']['n_particles']  # Get number of particles
    coord_cols = [5 + i for i in range(n_particles)]  # Define coordination number columns
    return analysis_result['data'][:, coord_cols]  # Return coordination numbers array

# Example usage:
# result = read_coordination_analysis_file('coordination_analysis_phi_0.52_ar_1.4_vr_0.5_run_1_filename.txt')
# time_data = extract_column(result, 'time')
# all_coord_numbers = get_all_coordination_numbers(result)
# particle_0_coord = extract_column(result, 'coordination_number_particle_0')
'''

# Save utility functions to a separate file
utility_file = output_path / "coordination_analysis_reader.py"  # Define utility file path
with open(utility_file, 'w') as f:  # Open utility file in write mode
    f.write(utility_code)  # Write utility code to file

print(f"Utility reader functions saved to: {utility_file}")  # Print save confirmation for utility file
print("Import this file to easily read and analyze your results!")  # Print instructions for using utility file