import os  # Import os module for operating system functionalities
import glob  # Import glob module for file pattern matching
import platform  # Import platform module to determine operating system
from pathlib import Path  # Import Path class for handling file paths
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
from datetime import datetime  # Import datetime for timestamp generation
import readFiles

'''
Aug 11, 2025; RVP - Added detailed comments and improved structure
Aug 10, 2025; BAP - Intial version

Description: Describe what the code does here

NOTE: add any notes and usage instructions here
'''

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
particleFile    = 'par_*.dat'  # Define pattern for particle data files
dataFile        = 'data_*.dat'  # Define pattern for simulation data files
interactionFile = 'int_*.dat'  # Define pattern for interaction data files

dataArr = np.loadtxt(topDir / dataFile)  # Load data from a specific data file
parList = readFiles.readParFile(open(glob.glob(f'{topDir}/{particleFile}')[0]))
intList = readFiles.readParFile(open(glob.glob(f'{topDir}/{interactionFile}')[0]))

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
                    # data_dict = read_data_file(data_file)  # Read data file
                    # particles_data, metadata = read_particles_file(par_file)  # Read particle file
                    # interactions_data = read_interaction_file(interaction_file)  # Read interaction file

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